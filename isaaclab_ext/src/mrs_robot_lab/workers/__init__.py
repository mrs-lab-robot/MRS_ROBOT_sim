"""Isaac Lab environment factory and worker lifecycle.

Manages the single Isaac Sim/Kit worker with AppLauncher→ManagerBasedEnv lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from openflex_isaac_contract.session_config import SessionConfig, WorkerState
from mrs_robot_lab.recorders.episode_schema import CURRENT_EPISODE_FORMAT


logger = logging.getLogger(__name__)

_CAMERA_SENSOR_STREAMS = {
    "base_camera": "base_d435",
    "head_camera": "head_d435",
    "left_wrist_camera": "left_wrist_d405",
    "right_wrist_camera": "right_wrist_d405",
}


def _tensor_row(values: Any) -> list[float]:
    """Convert a single-batch Isaac tensor/array row to finite Python floats."""
    row = values[0] if getattr(values, "ndim", 0) > 1 else values
    if callable(getattr(row, "detach", None)):
        row = row.detach()
    if callable(getattr(row, "cpu", None)):
        row = row.cpu()
    if callable(getattr(row, "tolist", None)):
        row = row.tolist()
    try:
        result = [float(value) for value in row]
    except (TypeError, ValueError) as error:
        raise RuntimeError("Isaac Lab state tensor is not a numeric vector") from error
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError("Isaac Lab state tensor contains a non-finite value")
    return result


def _extract_joint_state(environment: Any, joint_state_names: tuple[str, ...]) -> dict[str, Any]:
    robot = getattr(environment, "_robot", None)
    data = getattr(robot, "data", None)
    names = tuple(getattr(data, "joint_names", ()))
    if not names or len(names) != len(set(names)):
        raise RuntimeError("Isaac Lab robot does not expose a unique joint-name list")
    missing = set(joint_state_names) - set(names)
    if missing:
        raise RuntimeError("Isaac Lab robot is missing contract joints: " + ", ".join(sorted(missing)))
    positions = _tensor_row(data.joint_pos)
    velocities = _tensor_row(data.joint_vel)
    if len(positions) != len(names) or len(velocities) != len(names):
        raise RuntimeError("Isaac Lab joint-state tensor width does not match its joint names")
    by_name = {name: index for index, name in enumerate(names)}
    order = [by_name[name] for name in joint_state_names]
    return {
        "names": list(joint_state_names),
        "position": [positions[index] for index in order],
        "velocity": [velocities[index] for index in order],
    }


def _capture_camera_frames(
    config: SessionConfig, environment: Any, step_index: int
) -> dict[str, bytes]:
    """Encode due RGB frames at the action timestamp, never after the action."""
    import numpy as np
    from PIL import Image

    scene = getattr(environment, "scene", None)
    sensors = getattr(scene, "sensors", {})
    frames: dict[str, bytes] = {}
    control_hz = float(config.frequency.control_hz)
    for sensor in config.sensors:
        if not sensor.enabled or sensor.sensor_type.lower() != "camera":
            continue
        stream_name = _CAMERA_SENSOR_STREAMS.get(sensor.sensor_id)
        if stream_name is None:
            raise RuntimeError(f"没有 HDF5 流映射的相机：{sensor.sensor_id}")
        rate = float(sensor.frequency_hz or config.frequency.render_hz)
        interval = control_hz / rate
        if rate > control_hz + 1e-6 or abs(interval - round(interval)) > 1e-6:
            raise RuntimeError(
                f"相机 {sensor.sensor_id} 频率不能映射到 Worker 控制 tick：{rate:g} Hz"
            )
        if step_index % int(round(interval)) != 0:
            continue
        camera = sensors.get(stream_name)
        if camera is None:
            raise RuntimeError(f"Isaac Lab 场景中不存在已启用相机 {sensor.sensor_id}")
        output = getattr(getattr(camera, "data", None), "output", {})
        rgb = output.get("rgb") if isinstance(output, dict) else None
        if rgb is None:
            # Sensors may not have produced their first render yet. Episode QC
            # will reject the episode if no frame arrives for an enabled stream.
            continue
        if callable(getattr(rgb, "detach", None)):
            rgb = rgb.detach()
        if callable(getattr(rgb, "cpu", None)):
            rgb = rgb.cpu()
        if callable(getattr(rgb, "numpy", None)):
            rgb = rgb.numpy()
        image = np.asarray(rgb)
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise RuntimeError(f"相机 {sensor.sensor_id} 当前不是单环境图像")
            image = image[0]
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise RuntimeError(f"相机 {sensor.sensor_id} RGB 图像维度无效：{image.shape}")
        image = image[:, :, :3]
        if not np.issubdtype(image.dtype, np.integer):
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=90)
        frames[stream_name] = buffer.getvalue()
    return frames


@dataclass
class WorkerStatus:
    """Current worker status snapshot."""

    state: WorkerState
    message: str = ""
    progress: float = 0.0  # 0.0 to 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class _IsaacLabRuntime:
    """Lazy Isaac Lab adapter; importing this module never boots Kit."""

    _SUPPORTED_TASKS = {"navigation_to_goal", "dual_arm_box_transport"}
    _CAMERA_SENSOR_IDS = frozenset(
        {"base_camera", "head_camera", "left_wrist_camera", "right_wrist_camera"}
    )
    _RUNTIME_SENSOR_IDS = frozenset(
        {"odom", "joint_state", "object_state"} | _CAMERA_SENSOR_IDS
    )
    _SUPPORTED_TELEOP_MODES = frozenset({"vr", "keyboard"})

    def __init__(
        self,
        *,
        process_factory: Callable[..., Any] | None = None,
        api_request: Callable[..., dict[str, Any]] | None = None,
        sleep_fn: Callable[[float], Any] = time.sleep,
        monotonic_fn: Callable[[], float] = time.monotonic,
        sidecar_start_timeout: float = 55.0,
    ) -> None:
        self._process_factory = process_factory or subprocess.Popen
        self._api_request = api_request or self._request_json
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._sidecar_start_timeout = float(sidecar_start_timeout)
        self._control_api_url = ""
        self._sidecar_process: Any | None = None
        self._sidecar_ready = False
        self._sidecar_message = "VR sidecar 尚未启动"
        self._keyboard_ready = False
        self._last_arm_action: tuple[float, ...] | None = None

    @staticmethod
    def _request_json(
        url: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 0.4,
    ) -> dict[str, Any]:
        import json

        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="GET" if body is None else "POST",
        )
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Worker API response must be a JSON object")
        return result

    def configure_teleop_api(self, base_url: str) -> None:
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("VR sidecar must use the Worker API on an explicit loopback port")
        self._control_api_url = base_url.rstrip("/")

    def teleop_state(self) -> dict[str, Any]:
        """Return live teleop health, including the owned sidecar process."""
        if self._keyboard_ready:
            return {"ready": True, "message": "GUI keyboard input channel ready"}
        process = self._sidecar_process
        if process is None:
            return {"ready": False, "message": "VR ROS 2 sidecar is not running"}
        return_code = process.poll()
        if return_code is not None:
            self._sidecar_ready = False
            self._sidecar_message = f"VR ROS 2 sidecar exited with code {return_code}"
            return {"ready": False, "message": self._sidecar_message}
        try:
            state = self._api_request(
                self._control_api_url + "/api/v1/teleop/state", timeout=0.4
            )
        except (URLError, TimeoutError, OSError, ValueError) as error:
            state = {"ready": False, "message": f"VR Worker API unavailable: {error}"}
        if not isinstance(state, dict):
            state = {"ready": False, "message": "VR Worker API returned invalid status"}
        self._sidecar_ready = state.get("ready") is True
        self._sidecar_message = str(state.get("message") or "VR ROS 2 control not ready")
        return {**state, "ready": self._sidecar_ready, "message": self._sidecar_message}

    def preflight(self, config: SessionConfig) -> dict[str, Any]:
        root = Path(config.simulation_repo_root).expanduser().resolve(strict=True)
        if not (root / "isaaclab_ext").is_dir() or not (root / "sim_runtime").is_dir():
            raise ValueError(f"仿真仓库目录缺少 isaaclab_ext 或 sim_runtime：{root}")
        if config.task_id not in self._SUPPORTED_TASKS:
            raise ValueError(f"Isaac Lab Worker 尚未适配任务 {config.task_id!r}")

        task_path = root / "sim_runtime" / "config" / "tasks" / f"{config.task_id}.yaml"
        scene_path = root / "sim_runtime" / "config" / "scenes" / f"{config.scene_id}.yaml"
        if not task_path.is_file():
            raise FileNotFoundError(f"任务配置不存在：{task_path}")
        if not scene_path.is_file():
            raise FileNotFoundError(f"场景配置不存在：{scene_path}")

        from mrs_robot_lab.environments.learning.configuration import load_task_configuration

        task_config = load_task_configuration(task_path, scene_spec_path=scene_path)
        if task_config.task.task_id != config.task_id:
            raise ValueError("SessionConfig 与任务 YAML 的 task_id 不一致")
        if task_config.scene.scene_id != config.scene_id:
            raise ValueError("SessionConfig 与场景 YAML 的 scene_id 不一致")
        if "capture" not in task_config.task.capabilities:
            raise ValueError(f"任务 {config.task_id!r} 未声明 capture 能力")

        enabled_sensor_ids = {sensor.sensor_id for sensor in config.sensors if sensor.enabled}
        missing = set(task_config.task.required_sensors) - enabled_sensor_ids
        if missing:
            raise ValueError("任务必需传感器未启用：" + ", ".join(sorted(missing)))
        if config.metadata.get("localization_source") == "fastlio2":
            missing_localization_inputs = {"lidar", "imu"} - enabled_sensor_ids
            if missing_localization_inputs:
                raise ValueError(
                    "FAST-LIO2 定位必须同时启用雷达和 IMU："
                    + ", ".join(sorted(missing_localization_inputs))
                )
            raise ValueError(
                "当前 Isaac Lab Worker 尚未接入 FAST-LIO2 sidecar；"
                "请改用 Isaac Lab 仿真真值里程计，或先完成 FAST-LIO2 适配。"
            )
        unsupported = enabled_sensor_ids - self._RUNTIME_SENSOR_IDS
        if unsupported:
            raise ValueError(
                "当前 Isaac Lab 环境工厂尚未实现已启用的传感器："
                + ", ".join(sorted(unsupported))
            )
        camera_frequencies = self._camera_frequencies(config)
        if camera_frequencies:
            from mrs_robot_lab.assets.asset_resolver import AssetResolver
            from mrs_robot_lab.sensors.camera_cfg import load_camera_mounts

            load_camera_mounts(
                AssetResolver(root), sensor_frequencies_hz=camera_frequencies
            )
            control_hz = float(config.frequency.control_hz)
            for sensor_id, frequency in camera_frequencies.items():
                ratio = control_hz / frequency
                if frequency > control_hz + 1e-6 or abs(ratio - round(ratio)) > 1e-6:
                    raise ValueError(
                        f"相机 {sensor_id} 的 {frequency:g} Hz 不能由当前 "
                        f"{control_hz:g} Hz Worker 逐帧采集；频率必须不高于控制频率且整除控制频率"
                    )
        if config.teleop_mode not in self._SUPPORTED_TELEOP_MODES:
            raise ValueError(
                f"当前 Isaac Lab Worker 尚未接入 {config.teleop_mode} 遥操作适配器；"
                "为避免空采集会话，Kit 启动已阻止。"
            )
        teleop: dict[str, Any] = {
            "mode": config.teleop_mode,
            "sidecar_required": config.teleop_mode == "vr",
        }
        if config.teleop_mode == "vr":
            from mrs_robot_lab.teleoperation.vr_sidecar import load_vr_sidecar_config

            vr_sidecar = load_vr_sidecar_config(config)
            teleop.update(
                {
                    "ros_domain_id": vr_sidecar.domain_id,
                    "simulated_input": vr_sidecar.synthetic_input_script is not None,
                }
            )
        if config.enable_rviz:
            raise ValueError("Isaac Lab Worker 的 ROS 2/RViz sidecar 尚未接入")

        return {
            "repo_root": root,
            "task_path": task_path,
            "scene_path": scene_path,
            "required_sensors": tuple(task_config.task.required_sensors),
            "teleop": teleop,
        }

    def launch_app(self, config: SessionConfig) -> Any:
        camera_enabled = any(
            sensor.enabled and sensor.sensor_type.lower() == "camera"
            for sensor in config.sensors
        )
        from isaaclab.app import AppLauncher

        return AppLauncher(
            {
                "headless": config.headless,
                "enable_cameras": camera_enabled,
                "device": f"cuda:{config.gpu_id}",
            }
        )

    def build_environment(self, config: SessionConfig, _app_launcher: Any) -> Any:
        root = Path(config.simulation_repo_root).expanduser().resolve(strict=True)
        task_path = root / "sim_runtime" / "config" / "tasks" / f"{config.task_id}.yaml"
        scene_path = root / "sim_runtime" / "config" / "scenes" / f"{config.scene_id}.yaml"
        common = {
            "task_spec_path": task_path,
            "scene_spec_path": scene_path,
            "device": f"cuda:{config.gpu_id}",
            "num_envs": 1,
            "physics_hz": config.frequency.physics_hz,
            "control_hz": config.frequency.control_hz,
            "render_hz": config.frequency.render_hz,
            "seed": config.random_seed,
            "camera_frequencies_hz": self._camera_frequencies(config),
        }
        if config.task_id == "navigation_to_goal":
            from mrs_robot_lab.environments.learning.navigation_task import make_navigation_task_env

            return make_navigation_task_env(**common)
        if config.task_id == "dual_arm_box_transport":
            from mrs_robot_lab.environments.learning.dual_arm_box_task import make_dual_arm_box_task_env

            return make_dual_arm_box_task_env(**common)
        raise ValueError(f"Isaac Lab Worker 尚未适配任务 {config.task_id!r}")

    @staticmethod
    def _camera_frequencies(config: SessionConfig) -> dict[str, float]:
        """Return enabled camera rates keyed by stable SessionConfig sensor ID."""
        return {
            sensor.sensor_id: float(
                sensor.frequency_hz or config.frequency.render_hz
            )
            for sensor in config.sensors
            if sensor.enabled and sensor.sensor_type.lower() == "camera"
        }

    def reset_environment(self, environment: Any, seed: int) -> Any:
        self._last_arm_action = None
        return environment.reset(seed=seed)

    @staticmethod
    def warmup(environment: Any, steps: int) -> None:
        import torch

        action_count = int(environment.cfg.action_space)
        env_count = int(environment.num_envs)
        actions = torch.zeros((env_count, action_count), device=environment.device)
        for _ in range(steps):
            environment.step(actions)

    def start_sidecars(self, config: SessionConfig, _environment: Any) -> None:
        if config.teleop_mode == "keyboard":
            self._keyboard_ready = True
            self._sidecar_message = "GUI keyboard input channel ready"
            return
        if config.teleop_mode != "vr":
            raise RuntimeError(f"Isaac Lab Worker 尚未实现 {config.teleop_mode} 遥操作")
        if not self._control_api_url:
            raise RuntimeError("Worker loopback API 尚未连接，不能启动 VR sidecar")
        from mrs_robot_lab.teleoperation.vr_sidecar import (
            build_vr_sidecar_script,
            load_vr_sidecar_config,
        )

        sidecar_config = load_vr_sidecar_config(config)
        script = build_vr_sidecar_script(config, sidecar_config, self._control_api_url)
        self._sidecar_process = self._process_factory(
            ["bash", "-c", script],
            cwd=str(sidecar_config.repo_root),
            start_new_session=True,
        )
        self._sidecar_ready = False
        self._sidecar_message = "等待 Pico/VR ROS 2 控制节点就绪"
        deadline = self._monotonic_fn() + self._sidecar_start_timeout
        state_url = self._control_api_url + "/api/v1/teleop/state"
        while True:
            return_code = self._sidecar_process.poll()
            if return_code is not None:
                self._sidecar_process = None
                raise RuntimeError(
                    f"VR ROS 2 sidecar exited with code {return_code} before readiness"
                )
            try:
                state = self._api_request(state_url, timeout=0.4)
            except (URLError, TimeoutError, OSError, ValueError) as error:
                state = {"ready": False, "message": str(error)}
            if state.get("ready") is True:
                self._sidecar_ready = True
                self._sidecar_message = str(state.get("message") or "Pico VR ROS 2 已就绪")
                return
            self._sidecar_message = str(state.get("message") or "等待 VR ROS 2 控制节点")
            if self._monotonic_fn() >= deadline:
                message = self._sidecar_message
                self._stop_owned_sidecar()
                raise RuntimeError(f"VR ROS 2 sidecar 启动超时：{message}")
            self._sleep_fn(0.1)

    def health_check(self, config: SessionConfig, environment: Any) -> None:
        if environment is None or getattr(environment, "is_closed", False):
            raise RuntimeError("Isaac Lab 环境未初始化或已关闭")
        if getattr(environment, "num_envs", 0) != 1:
            raise RuntimeError("采集 Worker 当前只支持单环境运行")
        if config.teleop_mode == "keyboard" and not self._keyboard_ready:
            raise RuntimeError("GUI keyboard input channel is not ready")
        if config.teleop_mode == "vr":
            if self._sidecar_process is None or self._sidecar_process.poll() is not None:
                raise RuntimeError("VR ROS 2 sidecar 未运行")
            state = self._api_request(
                self._control_api_url + "/api/v1/teleop/state", timeout=0.4
            )
            if state.get("ready") is not True:
                raise RuntimeError(
                    "VR ROS 2 sidecar 尚未就绪："
                    + str(state.get("message") or "未知错误")
                )
            self._sidecar_ready = True
            self._sidecar_message = str(state.get("message") or "Pico VR ROS 2 已就绪")
        for sensor in config.sensors:
            if not sensor.enabled or sensor.sensor_type.lower() != "camera":
                continue
            mount_name = _CAMERA_SENSOR_STREAMS.get(sensor.sensor_id)
            camera = getattr(environment.scene, "sensors", {}).get(mount_name)
            if camera is None:
                raise RuntimeError(f"已启用相机 {sensor.sensor_id} 未注册到 Isaac Lab 环境")
            output = getattr(getattr(camera, "data", None), "output", {})
            if not isinstance(output, dict) or "rgb" not in output:
                raise RuntimeError(f"相机 {sensor.sensor_id} 尚未输出 RGB 图像")

    def close(self, environment: Any, app_launcher: Any) -> None:
        self._stop_owned_sidecar()
        self._keyboard_ready = False
        if environment is not None:
            environment.close()
        app = getattr(app_launcher, "app", None)
        close = getattr(app, "close", None)
        if callable(close):
            close()

    def _stop_owned_sidecar(self) -> None:
        process = self._sidecar_process
        self._sidecar_process = None
        self._sidecar_ready = False
        if process is not None and process.poll() is None:
            process_group = getattr(process, "pid", None)
            if process_group is None:
                process.terminate()
                process.wait(timeout=5.0)
            else:
                try:
                    os.killpg(process_group, signal.SIGINT)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process_group, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process_group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2.0)
        self._sidecar_message = "VR sidecar 已停止"

    @staticmethod
    def pump_once(app_launcher: Any) -> None:
        app = getattr(app_launcher, "app", None)
        update = getattr(app, "update", None)
        if callable(update):
            update()

    def step_once(
        self,
        config: SessionConfig,
        environment: Any,
        command: dict[str, Any] | None,
        *,
        step_index: int,
        recording: bool,
    ) -> dict[str, Any]:
        """Apply one task action and capture its causal state transition."""
        import torch

        from mrs_robot_lab.assets.robot_interface import (
            ACTION_DIMENSION,
            JOINT_POSITION_LIMITS,
            JOINT_STATE_NAMES,
            LEFT_ARM_JOINTS,
            LEFT_GRIPPER_JOINTS,
            RIGHT_ARM_JOINTS,
            RIGHT_GRIPPER_JOINTS,
            SWERVE_CONFIG,
        )
        from mrs_robot_lab.teleoperation.vr_action_adapter import (
            normalize_base_twist,
            normalize_bilateral_arm_targets,
        )

        raw_command = dict(command) if isinstance(command, dict) else {"kind": "watchdog_neutral"}
        operator_command = [0.0] * ACTION_DIMENSION
        applied_target = [0.0] * ACTION_DIMENSION

        if config.task_id == "navigation_to_goal":
            requested = (
                command.get("values")
                if isinstance(command, dict)
                and command.get("kind") == "base_twist"
                and not bool(command.get("active", False))
                else (0.0, 0.0, 0.0)
            )
            if not isinstance(requested, (list, tuple)):
                raise ValueError("VR 底盘指令必须是三个物理量 (vx, vy, wz)")
            normalized = normalize_base_twist(requested, SWERVE_CONFIG.twist_limits)
            clipped = tuple(
                min(max(float(value), lower), upper)
                for value, (lower, upper) in zip(
                    requested, SWERVE_CONFIG.twist_limits, strict=True
                )
            )
            if isinstance(command, dict) and command.get("kind") == "base_twist":
                operator_command[:3] = [float(value) for value in requested]
            applied_target[:3] = list(clipped)
            task_action = normalized
        elif config.task_id == "dual_arm_box_transport":
            task_joint_names = (
                *LEFT_ARM_JOINTS,
                *RIGHT_ARM_JOINTS,
                *LEFT_GRIPPER_JOINTS,
                *RIGHT_GRIPPER_JOINTS,
            )
            task_joint_limits = tuple(JOINT_POSITION_LIMITS[name] for name in task_joint_names)
            is_arm_command = isinstance(command, dict) and command.get("kind") == "dual_arm"
            left = command.get("left") if is_arm_command else None
            right = command.get("right") if is_arm_command else None
            held_action = self._last_arm_action
            if held_action is None and (left is None or right is None):
                current_targets = _tensor_row(environment._joint_targets)
                if len(current_targets) != 16:
                    raise RuntimeError("双臂环境的当前关节目标维度不是 16")
                held_action = normalize_bilateral_arm_targets(
                    (*current_targets[:7], current_targets[14]),
                    (*current_targets[7:14], current_targets[15]),
                    task_joint_limits,
                )
            task_action = normalize_bilateral_arm_targets(
                left,
                right,
                task_joint_limits,
                hold_action=held_action,
            )
            self._last_arm_action = tuple(task_action)
            if is_arm_command:
                if left is not None:
                    for index, value in enumerate(left[:7]):
                        operator_command[3 + index] = float(value)
                    operator_command[20] = float(left[7])
                if right is not None:
                    for index, value in enumerate(right[:7]):
                        operator_command[10 + index] = float(value)
                    operator_command[21] = float(right[7])
            for index, (value, (lower, upper)) in enumerate(
                zip(task_action, task_joint_limits, strict=True)
            ):
                if index in (14, 15):
                    applied_value = lower if value < 0.0 else upper
                else:
                    applied_value = (lower + upper) / 2.0 + value * (upper - lower) / 2.0
                action_index = (3 + index) if index < 14 else (20 + index - 14)
                applied_target[action_index] = float(applied_value)
        else:
            raise ValueError(f"Isaac Lab Worker 不支持任务 {config.task_id!r} 的遥操作步进")

        joint_state_before = _extract_joint_state(environment, JOINT_STATE_NAMES) if recording else None
        sim_time_ns = int(round(step_index * 1_000_000_000 / config.frequency.control_hz))
        camera_frames = (
            _capture_camera_frames(config, environment, step_index)
            if recording
            else None
        )
        action_tensor = torch.as_tensor(
            task_action, dtype=torch.float32, device=environment.device
        ).reshape((1, -1))
        environment.step(action_tensor)
        joint_state = _extract_joint_state(environment, JOINT_STATE_NAMES)

        result: dict[str, Any] = {"joint_state": joint_state}
        if recording:
            sample = {
                "sim_time_ns": sim_time_ns,
                "step_index": int(step_index),
                "command_seq": int(step_index),
                "operator_command": operator_command,
                "applied_target": applied_target,
                "joint_position": joint_state_before["position"],
                "joint_velocity": joint_state_before["velocity"],
                "next_joint_position": joint_state["position"],
                "raw_command": raw_command,
            }
            if camera_frames:
                sample["camera_frames"] = camera_frames
            result["sample"] = sample
        return result


class IsaacLabWorker:
    """Single Isaac Lab environment worker.

    Responsibilities:
    - Launch AppLauncher (once, with session config)
    - Build ManagerBasedEnv from task configuration
    - Manage state transitions
    - Provide status API (loopback only)
    - Coordinate RecorderManager lifecycle

    Constraints:
    - Only one AppLauncher per process
    - Only one SimulationContext (owned by ManagerBasedEnv)
    - No external SimulationContext creation
    """

    _ALLOWED_TRANSITIONS = {
        WorkerState.IDLE: {WorkerState.PREFLIGHT, WorkerState.STOPPING},
        WorkerState.PREFLIGHT: {WorkerState.LAUNCHING_KIT, WorkerState.STOPPING},
        WorkerState.LAUNCHING_KIT: {WorkerState.BUILDING_ENV, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.BUILDING_ENV: {WorkerState.RESETTING, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.RESETTING: {WorkerState.WARMING_UP, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.WARMING_UP: {
            WorkerState.STARTING_SIDECARS,
            WorkerState.HEALTH_CHECK,
            WorkerState.FAILED,
            WorkerState.STOPPING,
        },
        WorkerState.STARTING_SIDECARS: {WorkerState.HEALTH_CHECK, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.HEALTH_CHECK: {WorkerState.ENV_READY, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.ENV_READY: {
            WorkerState.RECORDING,
            WorkerState.RESETTING,
            WorkerState.EXPORTING,
            WorkerState.STOPPING,
        },
        WorkerState.RECORDING: {
            WorkerState.SAVING,
            WorkerState.ENV_READY,
            WorkerState.STOPPING,
        },
        WorkerState.SAVING: {WorkerState.QC, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.QC: {WorkerState.EXPORTING, WorkerState.ENV_READY, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.EXPORTING: {WorkerState.ENV_READY, WorkerState.FAILED, WorkerState.STOPPING},
        WorkerState.STOPPING: {WorkerState.STOPPING},
        WorkerState.FAILED: {WorkerState.STOPPING},
    }

    def __init__(
        self,
        config: SessionConfig,
        *,
        runtime: Any | None = None,
        recorder_factory: Callable[[], Any] | None = None,
        warmup_steps: int = 5,
        export_callback: Callable[[Path, SessionConfig, dict[str, Any]], Any] | None = None,
    ) -> None:
        """Initialize worker with session configuration.

        Args:
            config: Session configuration (will be validated on launch).
        """
        self._config = config
        self._runtime = runtime if runtime is not None else _IsaacLabRuntime()
        self._recorder_factory = recorder_factory
        self._warmup_steps = int(warmup_steps)
        if self._warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        self._export_callback = export_callback
        self._state = WorkerState.IDLE
        self._env = None
        self._app_launcher = None
        self._recorder = None
        self._status_listeners: list[callable] = []
        self._failure_reason = ""
        self._preflight: dict[str, Any] = {}
        self._active_episode_metadata: dict[str, Any] | None = None
        self._recording_faults: list[str] = []
        self._last_saved_path: Path | None = None
        self._last_saved_metadata: dict[str, Any] | None = None
        self._last_quarantined_path: Path | None = None
        self._last_qc_failure_reasons: tuple[str, ...] = ()
        self._last_export_error = ""
        self._teleop_transport: Any | None = None
        self._next_control_tick = 0.0
        self._worker_step_index = 0

        logger.info(f"Worker initialized: session={config.session_id}")

    def configure(self) -> None:
        """Validate configuration and transition to PREFLIGHT.

        Raises:
            ValueError: If configuration validation fails.
            RuntimeError: If not in IDLE state.
        """
        if self._state != WorkerState.IDLE:
            raise RuntimeError(f"Cannot configure from state: {self._state.value}")

        # Validate configuration
        self._config.validate()
        runtime_preflight = getattr(self._runtime, "preflight", None)
        if callable(runtime_preflight) and self._config.simulation_repo_root.strip():
            self._preflight = runtime_preflight(self._config)

        self._transition_to(WorkerState.PREFLIGHT)
        logger.info("Configuration validated, ready to launch")

    @property
    def state(self) -> WorkerState:
        """Current worker state."""
        return self._state

    @property
    def config(self) -> SessionConfig:
        """Immutable session configuration."""
        return self._config

    def get_status(self) -> WorkerStatus:
        """Get current worker status snapshot."""
        return WorkerStatus(
            state=self._state,
            message=self._get_state_message(),
            progress=self._estimate_progress(),
            metadata={
                "session_id": self._config.session_id,
                "config_hash": self._config.config_hash,
                "environment_id": self._config.task_id,
                "teleop_mode": self._config.teleop_mode,
                "episode_id": (
                    self._active_episode_metadata.get("episode_id")
                    if self._active_episode_metadata
                    else ""
                ),
                "steps": int(getattr(self._recorder, "num_steps", 0)),
                "last_saved_path": str(self._last_saved_path or ""),
                "last_export_error": self._last_export_error,
                "last_quarantined_path": str(self._last_quarantined_path or ""),
                "last_qc_failure_reasons": list(self._last_qc_failure_reasons),
                "recording_faults": list(self._recording_faults),
                **self._teleop_status_snapshot(),
                "preflight": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in self._preflight.items()
                    if key not in {"task_path", "scene_path"}
                },
                **(
                    {"failure_reason": self._failure_reason}
                    if self._failure_reason
                    else {}
                ),
            },
        )

    def _teleop_status_snapshot(self) -> dict[str, Any]:
        state = self._read_teleop_state()
        return {"teleop": dict(state)} if isinstance(state, dict) else {}

    def _read_teleop_state(self) -> dict[str, Any] | None:
        state_reader = getattr(self._runtime, "teleop_state", None)
        if callable(state_reader):
            try:
                state = state_reader()
            except Exception as error:
                logger.warning("Unable to read VR sidecar state: %s", error)
                return {"ready": False, "message": f"VR 状态检查失败：{error}"}
            if isinstance(state, dict):
                return state
        transport_reader = getattr(self._teleop_transport, "teleop_state", None)
        if callable(transport_reader):
            state = transport_reader()
            return state if isinstance(state, dict) else None
        return None

    def _annotate_active_episode(self, metadata: dict[str, Any]) -> None:
        if self._active_episode_metadata is not None:
            self._active_episode_metadata.update(metadata)
        annotate = getattr(self._recorder, "annotate", None)
        if callable(annotate):
            try:
                annotate(metadata)
            except Exception:
                logger.exception("Unable to annotate active Episode metadata")

    def _mark_recording_fault(self, reason: str) -> None:
        if reason in self._recording_faults:
            return
        self._recording_faults.append(reason)
        self._annotate_active_episode(
            {
                "recording_faults": list(self._recording_faults),
                "teleop_ready_at_end": False,
            }
        )

    def _check_vr_recording_health(self) -> None:
        if self._state != WorkerState.RECORDING or self._config.teleop_mode != "vr":
            return
        state = self._read_teleop_state()
        if isinstance(state, dict) and state.get("ready") is True:
            return
        message = (
            str(state.get("message") or "遥操作状态未知")
            if isinstance(state, dict)
            else "遥操作状态通道不可用"
        )
        self._mark_recording_fault(
            f"VR 遥操作在 Episode 录制期间断开：{message}"
        )

    def add_status_listener(self, listener: Callable[[WorkerStatus], Any]) -> None:
        """Register a listener and immediately provide the current snapshot."""
        if not callable(listener):
            raise TypeError("status listener must be callable")
        self._status_listeners.append(listener)
        listener(self.get_status())

    def pump_once(self) -> None:
        """Advance one teleoperation control tick on the Kit-owning thread."""
        stepper = getattr(self._runtime, "step_once", None)
        if (
            callable(stepper)
            and self._state in {WorkerState.ENV_READY, WorkerState.RECORDING}
        ):
            now = time.monotonic()
            if now >= self._next_control_tick:
                self._next_control_tick = now + 1.0 / float(
                    self._config.frequency.control_hz
                )
                self._check_vr_recording_health()
                command = None
                if self._teleop_transport is not None:
                    command = self._teleop_transport.take_latest_teleop_command()
                result = stepper(
                    self._config,
                    self._env,
                    command,
                    step_index=self._worker_step_index,
                    recording=self._state == WorkerState.RECORDING,
                )
                self._worker_step_index += 1
                if result is not None:
                    if self._state == WorkerState.RECORDING:
                        self._check_vr_recording_health()
                        sample = result.get("sample")
                        if not isinstance(sample, dict):
                            raise RuntimeError(
                                "Isaac Lab 遥操作步进没有返回可记录的 Episode 样本"
                            )
                        self.append_sample(**sample)
                    if self._teleop_transport is not None:
                        joint_state = result.get("joint_state")
                        if isinstance(joint_state, dict):
                            self._teleop_transport.update_teleop_state(
                                joint_state=joint_state
                            )
        pump = getattr(self._runtime, "pump_once", None)
        if callable(pump):
            pump(self._app_launcher)

    def attach_teleop_transport(self, transport: Any) -> None:
        """Attach the loopback API mailbox used by the ROS 2 bridge process."""
        take_command = getattr(transport, "take_latest_teleop_command", None)
        update_state = getattr(transport, "update_teleop_state", None)
        if not callable(take_command) or not callable(update_state):
            raise TypeError("teleoperation transport must expose command and state methods")
        self._teleop_transport = transport
        configure_api = getattr(self._runtime, "configure_teleop_api", None)
        base_url = getattr(transport, "base_url", None)
        if callable(configure_api) and isinstance(base_url, str):
            configure_api(base_url)

    def launch(self) -> None:
        """Launch Isaac Sim Kit and build environment.

        State transitions:
        PREFLIGHT → LAUNCHING_KIT → BUILDING_ENV → RESETTING →
        WARMING_UP → STARTING_SIDECARS → HEALTH_CHECK → ENV_READY

        Raises:
            RuntimeError: If already launched or validation fails.
        """
        if self._state != WorkerState.PREFLIGHT:
            raise RuntimeError(f"Worker already launched: current state={self._state.value}")

        try:
            self._transition_to(WorkerState.LAUNCHING_KIT)
            self._launch_app()

            self._transition_to(WorkerState.BUILDING_ENV)
            self._build_environment()

            self._transition_to(WorkerState.RESETTING)
            self._reset_environment()

            self._transition_to(WorkerState.WARMING_UP)
            self._warmup()

            self._transition_to(WorkerState.STARTING_SIDECARS)
            self._start_sidecars()

            self._transition_to(WorkerState.HEALTH_CHECK)
            self._health_check()

            self._transition_to(WorkerState.ENV_READY)
            logger.info("Worker ready for episodes")

        except Exception as e:
            logger.exception("Worker launch failed")
            self._failure_reason = str(e)
            self._transition_to(WorkerState.FAILED)
            raise RuntimeError(f"Worker launch failed: {e}") from e

    def start_recording(self, metadata: dict[str, Any] | None = None) -> None:
        """Start episode recording.

        Transitions: ENV_READY → RECORDING

        Raises:
            RuntimeError: If not in ENV_READY state.
        """
        if self._state != WorkerState.ENV_READY:
            raise RuntimeError(f"Cannot start recording from state: {self._state.value}")

        if not self._config.enable_recording:
            raise RuntimeError("当前 SessionConfig 禁用了 Episode 录制")
        if self._config.teleop_mode == "vr":
            state = self._read_teleop_state()
            if not isinstance(state, dict) or state.get("ready") is not True:
                detail = (
                    str(state.get("message") or "遥操作状态未知")
                    if isinstance(state, dict)
                    else "Worker 遥操作状态通道不可用"
                )
                raise RuntimeError(f"VR sidecar 未就绪，不能开始 Episode：{detail}")
        if self._last_saved_path is not None and self._last_export_error:
            raise RuntimeError("上一个 Episode 的 LeRobot 导出仍未完成；请先重试导出")
        if self._recorder is None:
            if self._recorder_factory is None:
                from mrs_robot_lab.recorders.episode_controller import EpisodeController

                self._recorder = EpisodeController()
            else:
                self._recorder = self._recorder_factory()
        self._recording_faults = []
        episode_metadata = self._episode_metadata(metadata or {})
        if self._config.teleop_mode == "vr":
            episode_metadata["teleop_ready_at_start"] = True
        self._recorder.start(episode_metadata)
        self._active_episode_metadata = episode_metadata
        self._transition_to(WorkerState.RECORDING)
        logger.info("Recording started")

    def stop_recording(self) -> None:
        """Stop episode recording and return to ENV_READY.

        Transitions: RECORDING → ENV_READY

        Raises:
            RuntimeError: If not in RECORDING state.
        """
        if self._state != WorkerState.RECORDING:
            raise RuntimeError(f"Cannot stop recording from state: {self._state.value}")

        if self._recorder is not None and getattr(self._recorder, "num_steps", 0):
            raise RuntimeError("Episode 含有样本；请先保存或丢弃，不能静默停止录制")
        if self._recorder is not None:
            self._recorder.discard()
        self._active_episode_metadata = None
        self._transition_to(WorkerState.ENV_READY)
        logger.info("Recording stopped")

    def save_episode(self, output_path: str | Path | None = None) -> Path:
        """Save current episode (HDF5, QC, export to LeRobot).

        Transitions: RECORDING → SAVING → QC → EXPORTING → ENV_READY

        Returns:
            Path to saved episode HDF5 file.

        Raises:
            RuntimeError: If not in RECORDING state or save fails.
        """
        if self._state != WorkerState.RECORDING:
            raise RuntimeError(f"Cannot save episode from state: {self._state.value}")

        try:
            self._check_vr_recording_health()
            if self._config.teleop_mode == "vr":
                self._annotate_active_episode(
                    {
                        "recording_faults": list(self._recording_faults),
                        "teleop_ready_at_end": not self._recording_faults,
                    }
                )
            # Status paths describe the latest save attempt, not an older episode.
            self._last_saved_path = None
            self._last_saved_metadata = None
            self._last_quarantined_path = None
            self._last_qc_failure_reasons = ()
            self._last_export_error = ""
            self._transition_to(WorkerState.SAVING)
            hdf5_path = self._save_hdf5(output_path)

            self._transition_to(WorkerState.QC)
            qc_passed = self._quality_check(hdf5_path)
            if self._recording_faults:
                qc_passed = False
                self._last_qc_failure_reasons = tuple(
                    dict.fromkeys(
                        (*self._last_qc_failure_reasons, *self._recording_faults)
                    )
                )

            if not qc_passed:
                logger.warning(f"Episode QC failed: {hdf5_path}")
                if not self._last_qc_failure_reasons:
                    self._last_qc_failure_reasons = ("Episode 质量检查未通过",)
                hdf5_path = self._quarantine_hdf5(hdf5_path)
                self._last_quarantined_path = hdf5_path
                self._active_episode_metadata = None
                self._transition_to(WorkerState.ENV_READY)
                return hdf5_path

            self._last_saved_path = hdf5_path
            self._last_saved_metadata = dict(self._active_episode_metadata or {})
            self._active_episode_metadata = None
            self._transition_to(WorkerState.EXPORTING)
            try:
                self._export_lerobot(hdf5_path)
            except Exception as error:
                self._last_export_error = str(error)
                logger.exception("Episode is valid HDF5 but LeRobot export failed: %s", hdf5_path)
                self._transition_to(WorkerState.ENV_READY)
                return hdf5_path

            self._last_export_error = ""
            self._transition_to(WorkerState.ENV_READY)
            logger.info(f"Episode saved and exported: {hdf5_path}")
            return hdf5_path

        except Exception as e:
            logger.exception("Episode save failed")
            self._failure_reason = str(e)
            self._transition_to(WorkerState.FAILED)
            raise RuntimeError(f"Episode save failed: {e}") from e

    def retry_export(self) -> Path:
        """Retry LeRobot export for the most recently QC-approved HDF5 episode."""
        if self._state != WorkerState.ENV_READY:
            raise RuntimeError(f"Cannot retry export from state: {self._state.value}")
        if self._last_saved_path is None or self._last_saved_metadata is None:
            raise RuntimeError("没有可重试导出的已质检 HDF5 Episode")
        if not self._last_export_error:
            raise RuntimeError("最近的 Episode 没有待重试的 LeRobot 导出")

        hdf5_path = self._last_saved_path
        self._transition_to(WorkerState.EXPORTING)
        try:
            self._export_lerobot(hdf5_path)
        except Exception as error:
            self._last_export_error = str(error)
            self._transition_to(WorkerState.ENV_READY)
            raise RuntimeError(f"LeRobot export retry failed: {error}") from error

        self._last_export_error = ""
        self._transition_to(WorkerState.ENV_READY)
        logger.info("LeRobot export retry succeeded: %s", hdf5_path)
        return hdf5_path

    @staticmethod
    def _quarantine_hdf5(hdf5_path: Path) -> Path:
        """Move a QC-rejected HDF5 out of the trainable episode directory."""
        source = Path(hdf5_path)
        if not source.is_file():
            raise FileNotFoundError(f"QC-failed Episode file does not exist: {source}")

        quarantine_dir = source.parent / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / source.name
        if destination.exists():
            destination = quarantine_dir / f"{source.stem}-{uuid4().hex}{source.suffix}"

        # The quarantine directory is a sibling, so this remains an atomic rename
        # on the same filesystem and never exposes a partially copied HDF5 file.
        source.replace(destination)
        return destination

    def discard_episode(self) -> None:
        """Discard current episode without saving.

        Transitions: RECORDING → ENV_READY

        Raises:
            RuntimeError: If not in RECORDING state.
        """
        if self._state != WorkerState.RECORDING:
            raise RuntimeError(f"Cannot discard episode from state: {self._state.value}")

        if self._recorder is not None:
            self._recorder.discard()
        self._active_episode_metadata = None
        self._recording_faults = []
        self._transition_to(WorkerState.ENV_READY)
        logger.info("Episode discarded")

    def shutdown(self) -> None:
        """Shutdown worker and cleanup resources.

        Transitions: * → STOPPING → (terminated)

        Safe to call from any state.
        """
        logger.info(f"Shutting down worker from state: {self._state.value}")
        if (
            self._state == WorkerState.RECORDING
            and self._recorder is not None
            and getattr(self._recorder, "num_steps", 0)
        ):
            raise RuntimeError("当前 Episode 尚有未保存样本；停止环境前必须保存或丢弃")
        self._transition_to(WorkerState.STOPPING)

        if self._recorder is not None and self._active_episode_metadata is not None:
            self._recorder.discard()
            self._active_episode_metadata = None
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close(self._env, self._app_launcher)
        self._env = None
        self._app_launcher = None

        logger.info("Worker shutdown complete")

    def _transition_to(self, new_state: WorkerState) -> None:
        """Transition to new state and notify listeners."""
        old_state = self._state
        allowed = self._ALLOWED_TRANSITIONS.get(old_state, set())
        if new_state != old_state and new_state not in allowed:
            raise RuntimeError(
                "Invalid worker state transition: "
                f"{old_state.value} -> {new_state.value}"
            )
        self._state = new_state
        logger.debug(f"State transition: {old_state.value} → {new_state.value}")

        # Notify status listeners
        status = self.get_status()
        for listener in self._status_listeners:
            try:
                listener(status)
            except Exception:
                logger.exception("Status listener failed")

    def _get_state_message(self) -> str:
        """Get human-readable message for current state."""
        messages = {
            WorkerState.PREFLIGHT: "等待启动",
            WorkerState.LAUNCHING_KIT: "正在启动 Isaac Sim Kit...",
            WorkerState.BUILDING_ENV: "正在构建环境...",
            WorkerState.RESETTING: "正在重置环境...",
            WorkerState.WARMING_UP: "正在预热...",
            WorkerState.STARTING_SIDECARS: "正在启动传感器服务...",
            WorkerState.HEALTH_CHECK: "正在健康检查...",
            WorkerState.ENV_READY: "准备就绪",
            WorkerState.RECORDING: "正在录制",
            WorkerState.SAVING: "正在保存...",
            WorkerState.QC: "正在质量检查...",
            WorkerState.EXPORTING: "正在导出到 LeRobot...",
            WorkerState.STOPPING: "正在停止...",
            WorkerState.FAILED: (
                f"失败：{self._failure_reason}" if self._failure_reason else "失败"
            ),
        }
        return messages.get(self._state, "未知状态")

    def _estimate_progress(self) -> float:
        """Estimate progress for current state (0.0 to 1.0)."""
        # Simplified progress estimation
        progress_map = {
            WorkerState.PREFLIGHT: 0.0,
            WorkerState.LAUNCHING_KIT: 0.1,
            WorkerState.BUILDING_ENV: 0.3,
            WorkerState.RESETTING: 0.5,
            WorkerState.WARMING_UP: 0.6,
            WorkerState.STARTING_SIDECARS: 0.7,
            WorkerState.HEALTH_CHECK: 0.9,
            WorkerState.ENV_READY: 1.0,
            WorkerState.RECORDING: 1.0,
            WorkerState.SAVING: 0.3,
            WorkerState.QC: 0.6,
            WorkerState.EXPORTING: 0.9,
            WorkerState.STOPPING: 0.5,
            WorkerState.FAILED: 0.0,
        }
        return progress_map.get(self._state, 0.0)

    def _launch_app(self) -> None:
        """Launch AppLauncher with session config."""
        self._app_launcher = self._runtime.launch_app(self._config)

    def _build_environment(self) -> None:
        """Create exactly one task environment, which owns SimulationContext."""
        self._env = self._runtime.build_environment(self._config, self._app_launcher)

    def _reset_environment(self) -> None:
        """Reset the environment with the frozen session seed."""
        self._runtime.reset_environment(self._env, self._config.random_seed)

    def _warmup(self) -> None:
        """Run configured warm-up steps before enabling user capture."""
        self._runtime.warmup(self._env, self._warmup_steps)

    def _start_sidecars(self) -> None:
        """Start external input/sensor processes, if the runtime adapter supports them."""
        self._runtime.start_sidecars(self._config, self._env)
        if self._config.teleop_mode == "keyboard" and self._teleop_transport is not None:
            self._teleop_transport.update_teleop_state(
                ready=True,
                message="GUI keyboard input channel ready",
            )

    def _health_check(self) -> None:
        """Confirm environment and enabled runtime services are responsive."""
        self._runtime.health_check(self._config, self._env)

    def append_sample(self, **sample: Any) -> None:
        """Append one already time-aligned observation/action sample."""
        if self._state != WorkerState.RECORDING or self._recorder is None:
            raise RuntimeError("只能在 RECORDING 状态追加采集样本")
        self._recorder.append(**sample)

    def reset_environment(self) -> None:
        """Reset one ready session without replacing its frozen configuration."""
        if self._state != WorkerState.ENV_READY:
            raise RuntimeError(f"Cannot reset environment from state: {self._state.value}")
        try:
            self._transition_to(WorkerState.RESETTING)
            self._reset_environment()
            self._transition_to(WorkerState.WARMING_UP)
            self._warmup()
            self._transition_to(WorkerState.HEALTH_CHECK)
            self._health_check()
            self._transition_to(WorkerState.ENV_READY)
        except Exception as error:
            self._failure_reason = str(error)
            self._transition_to(WorkerState.FAILED)
            raise RuntimeError(f"Environment reset failed: {error}") from error

    def handle_command(self, command: str, payload: dict[str, Any] | None = None) -> Any:
        """Dispatch a validated control-API command on the Kit-owning thread."""
        payload = payload or {}
        if command == "start":
            self.start_recording(payload.get("metadata"))
            return self.get_status()
        if command == "save":
            return self.save_episode(payload.get("output_path"))
        if command == "export":
            return self.retry_export()
        if command == "discard":
            self.discard_episode()
            return self.get_status()
        if command == "reset":
            self.reset_environment()
            return self.get_status()
        if command == "stop":
            self.shutdown()
            return self.get_status()
        raise ValueError(f"Unsupported worker command: {command!r}")

    def _episode_metadata(self, supplied: dict[str, Any]) -> dict[str, Any]:
        from mrs_robot_lab.assets.robot_interface import JOINT_STATE_NAMES, ROBOT_CONTRACT_PATH

        try:
            contract_hash = hashlib.sha256(ROBOT_CONTRACT_PATH.read_bytes()).hexdigest()
        except OSError as error:
            raise RuntimeError(f"无法计算机器人动作/状态契约哈希：{error}") from error
        metadata = dict(supplied)
        metadata.update(
            {
                "episode_id": str(supplied.get("episode_id") or uuid4()),
                "config_hash": self._config.config_hash,
                "contract_hash": contract_hash,
                "random_seed": self._config.random_seed,
                "input_source": (
                    "simulated_vr"
                    if self._config.metadata.get("simulated_vr", False)
                    else self._config.teleop_mode
                ),
                "teleop_mode": self._config.teleop_mode,
                "embodiment_id": self._config.embodiment_id,
                "scene_id": self._config.scene_id,
                "task_id": self._config.task_id,
                "dataset_id": self._config.dataset_id,
                "joint_state_order": list(JOINT_STATE_NAMES),
            }
        )
        return metadata

    def _save_hdf5(self, output_path: str | Path | None = None) -> Path:
        """Save episode to HDF5 (atomic)."""
        if self._recorder is None:
            raise RuntimeError("Episode recorder 尚未初始化")
        if output_path is None:
            if not self._config.episode_root.strip():
                raise ValueError("episode_root 未配置，不能保存 HDF5")
            episode_id = (self._active_episode_metadata or {}).get("episode_id")
            if not episode_id:
                raise RuntimeError("当前 Episode 缺少 episode_id")
            output_path = (
                Path(self._config.episode_root).expanduser()
                / self._config.task_id
                / f"{episode_id}.hdf5"
            )
        return Path(self._recorder.save(output_path))

    def _quality_check(self, hdf5_path: Path) -> bool:
        """Perform quality check on saved episode."""
        issues: list[str] = []
        if not hdf5_path.is_file():
            self._last_qc_failure_reasons = ("HDF5 文件不存在",)
            return False
        if hdf5_path.stat().st_size == 0:
            self._last_qc_failure_reasons = ("HDF5 文件为空",)
            return False
        try:
            import h5py
            import json

            with h5py.File(hdf5_path, "r") as dataset:
                try:
                    metadata = json.loads(dataset.attrs["metadata_json"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    metadata = {}
                    issues.append("HDF5 元数据缺失或 JSON 无效")
                if not isinstance(metadata, dict):
                    metadata = {}
                    issues.append("HDF5 元数据必须是 JSON 对象")
                required = {
                    "episode_id",
                    "config_hash",
                    "contract_hash",
                    "random_seed",
                    "input_source",
                    "embodiment_id",
                    "scene_id",
                    "task_id",
                    "dataset_id",
                    "teleop_mode",
                }
                missing_metadata = sorted(required - set(metadata))
                if missing_metadata:
                    issues.append("缺少溯源字段：" + ", ".join(missing_metadata))
                if metadata.get("input_source") == "simulated_vr":
                    issues.append("模拟 VR 是诊断输入，默认不允许进入训练集")
                recording_faults = metadata.get("recording_faults", [])
                if isinstance(recording_faults, list) and recording_faults:
                    issues.extend(str(reason) for reason in recording_faults)
                if metadata.get("quality_status") == "failed":
                    issues.append("Episode 元数据将质量状态标记为失败")
                if metadata.get("quality_status") != "passed":
                    issues.append("EpisodeController 未将样本标记为质量通过")
                if dataset.attrs.get("format", "") != CURRENT_EPISODE_FORMAT:
                    issues.append("HDF5 格式标识缺失或不受支持")

                required_arrays = (
                    "sim_time_ns",
                    "step_index",
                    "command_seq",
                    "operator_command",
                    "applied_target",
                    "joint_position",
                    "joint_velocity",
                    "next_joint_position",
                    "action",
                )
                missing_arrays = [name for name in required_arrays if name not in dataset]
                if missing_arrays:
                    issues.append("缺少 HDF5 数据集：" + ", ".join(missing_arrays))
                if missing_arrays:
                    self._last_qc_failure_reasons = tuple(issues)
                    return False

                step_count = len(dataset["sim_time_ns"])
                if step_count == 0:
                    issues.append("Episode 不含采集步")
                if len(dataset["step_index"]) != step_count:
                    issues.append("step_index 数量与 sim_time_ns 不一致")
                if len(dataset["action"]) != step_count:
                    issues.append("action 数量与 sim_time_ns 不一致")
                sim_times = dataset["sim_time_ns"][:]
                step_indices = dataset["step_index"][:]
                if len(sim_times) > 1 and (sim_times[1:] <= sim_times[:-1]).any():
                    issues.append("sim_time_ns 必须严格递增")
                if len(step_indices) > 1 and (step_indices[1:] <= step_indices[:-1]).any():
                    issues.append("step_index 必须严格递增")
                import numpy as np

                for array_name in required_arrays[2:]:
                    values = dataset[array_name][:]
                    if values.ndim < 1 or values.shape[0] != step_count:
                        issues.append(f"{array_name} 的样本维度与 sim_time_ns 不一致")
                    elif not np.isfinite(values).all():
                        issues.append(f"{array_name} 含有非有限数值")

                cameras = dataset.get("camera_frames")
                if cameras is None or not len(cameras):
                    issues.append("相机帧缺失")
                else:
                    enabled_camera_ids = {
                        sensor.sensor_id
                        for sensor in self._config.sensors
                        if sensor.enabled and sensor.sensor_type.lower() == "camera"
                    }
                    unknown_camera_ids = sorted(
                        enabled_camera_ids - set(_CAMERA_SENSOR_STREAMS)
                    )
                    if unknown_camera_ids:
                        issues.append(
                            "没有 HDF5 流映射的相机：" + ", ".join(unknown_camera_ids)
                        )
                    required_streams = {
                        _CAMERA_SENSOR_STREAMS[sensor_id]
                        for sensor_id in enabled_camera_ids
                        if sensor_id in _CAMERA_SENSOR_STREAMS
                    }
                    missing_streams = sorted(required_streams - set(cameras.keys()))
                    if missing_streams:
                        issues.append(
                            "缺少已启用相机帧：" + ", ".join(missing_streams)
                        )
                action_times = dataset["sim_time_ns"][:]
                for camera_name, camera in (cameras.items() if cameras is not None else ()):
                    required_camera_arrays = ("sim_time_ns", "jpeg")
                    missing_camera_arrays = [
                        name for name in required_camera_arrays if name not in camera
                    ]
                    if missing_camera_arrays:
                        issues.append(
                            f"相机 {camera_name} 缺少数据集："
                            + ", ".join(missing_camera_arrays)
                        )
                        continue
                    times = camera["sim_time_ns"][:]
                    if len(times) != len(camera["jpeg"]):
                        issues.append(f"相机 {camera_name} 时间戳与图像数量不一致")
                        continue
                    if len(times) > 1 and (times[1:] <= times[:-1]).any():
                        issues.append(f"相机 {camera_name} 时间戳非严格递增")
                    if len(times) and len(action_times) and (
                        times[0] < action_times[0] or times[-1] > action_times[-1]
                    ):
                        issues.append(f"相机 {camera_name} 时间范围超出动作时间轴")
                    for frame_index, frame in enumerate(camera["jpeg"]):
                        if not bytes(frame):
                            issues.append(f"相机 {camera_name} 第 {frame_index} 帧为空")
        except (OSError, ImportError, KeyError, TypeError, ValueError) as error:
            issues.append(f"HDF5 读取失败：{error}")

        self._last_qc_failure_reasons = tuple(dict.fromkeys(issues))
        if self._last_qc_failure_reasons:
            return False
        return True

    def _export_lerobot(self, hdf5_path: Path) -> None:
        """Export HDF5 to LeRobot format."""
        if self._export_callback is None:
            raise RuntimeError(
                "LeRobot 自动追加适配器尚未配置；HDF5 已保留，不能伪报导出成功"
            )
        self._export_callback(
            hdf5_path,
            self._config,
            dict(self._active_episode_metadata or self._last_saved_metadata or {}),
        )
