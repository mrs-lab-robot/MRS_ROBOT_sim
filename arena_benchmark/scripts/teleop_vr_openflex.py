#!/usr/bin/env python3
"""Run OpenFlex Arena with the existing ROS 2 VR teleoperation stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import sys
import threading
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path

_CAMERA_CHUNK_SIZE_DEFAULT = 1_200
_ARENA_DISABLE_FABRIC = False

_CAMERA_OBSERVATION_NAMES = {
    "base_d435": "base_d435_rgb",
    "head_d435": "head_d435_rgb",
    "left_wrist_d405": "left_wrist_d405_rgb",
    "right_wrist_d405": "right_wrist_d405_rgb",
}


def _positive_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return parsed


def _add_positive_finite_float_option(
    parser: argparse.ArgumentParser, option: str, *, default: float
) -> argparse.Action:
    return parser.add_argument(
        option, type=_positive_finite_float, default=default
    )


def _frame_interval_for_hz(step_dt: float, hz: float) -> int:
    """Convert a requested sample rate into a positive simulation-step interval."""
    step_dt = float(step_dt)
    hz = float(hz)
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and greater than zero")
    if not math.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and greater than zero")

    product = step_dt * hz
    if product == 0.0:
        # Preserve positive finite inputs even if their product underflows.
        interval = round(
            Fraction(1, 1)
            / Fraction.from_float(step_dt)
            / Fraction.from_float(hz)
        )
    elif math.isinf(product):
        interval = 1
    else:
        # Keep the same rounding used by the historical recording cadence.
        try:
            interval = round(1.0 / product)
        except OverflowError:
            interval = round(
                Fraction(1, 1)
                / Fraction.from_float(step_dt)
                / Fraction.from_float(hz)
            )
    return max(1, int(interval))


def _camera_step_roles(
    sim_step: int, record_interval: int, stream_interval: int
) -> tuple[bool, bool, bool]:
    """Return (capture, record, stream) for independent camera cadences."""
    if record_interval < 1 or stream_interval < 1:
        raise ValueError("camera intervals must be positive")
    record_due = sim_step % record_interval == 0
    stream_due = sim_step % stream_interval == 0
    return record_due or stream_due, record_due, stream_due


def _encode_camera_observations(camera_observations) -> dict[str, bytes]:
    """JPEG-encode one synchronized set of Arena RGB camera observations."""
    import cv2
    import numpy as np

    if not isinstance(camera_observations, dict):
        raise RuntimeError("Arena observations 缺少 camera_obs 图像组")
    encoded_frames = {}
    for camera_name, observation_key in _CAMERA_OBSERVATION_NAMES.items():
        tensor = camera_observations.get(observation_key)
        if tensor is None:
            raise RuntimeError(f"Arena camera_obs 缺少 {observation_key}")
        image = tensor[0].detach().to(device="cpu").numpy()
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(
                f"Arena RGB 相机 {camera_name} 形状无效：{tuple(image.shape)}"
            )
        if image.dtype != np.uint8:
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            if image.size and float(image.max()) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        encoded, jpeg = cv2.imencode(
            ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not encoded:
            raise RuntimeError(f"Arena 相机 {camera_name} JPEG 编码失败")
        encoded_frames[camera_name] = jpeg.tobytes()
    return encoded_frames


def _send_encoded_camera_observations(
    camera_socket,
    camera_target: tuple[str, int],
    camera_frames: dict[str, bytes],
    *,
    frame_id: int,
    sim_time_ns: int,
    chunk_size: int = _CAMERA_CHUNK_SIZE_DEFAULT,
) -> None:
    """Forward encoded camera frames as bounded UDP chunks for optional preview."""
    from mrs_teleoperation.camera_stream import encode_camera_frame

    for camera_name, jpeg_bytes in camera_frames.items():
        for packet in encode_camera_frame(
            camera_name,
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            jpeg_bytes=jpeg_bytes,
            chunk_size=chunk_size,
        ):
            camera_socket.sendto(packet, camera_target)


def _joint_state(robot, field: str) -> tuple[list[float], dict[str, float]]:
    import torch

    names = tuple(robot.joint_names)
    values = getattr(robot.data, field)[0].detach().to(device="cpu", dtype=torch.float32).tolist()
    by_name = dict(zip(names, values))
    from mrs_robot_lab.adapters.teleop_adapter import JOINT_STATE_NAMES

    ordered = [float(by_name[name]) for name in JOINT_STATE_NAMES]
    return ordered, by_name


def _lio_state(robot, env) -> tuple[list[float], list[float]]:
    from mrs_arena.adapters.navigation_observation import _torch_view
    from mrs_arena.tasks.navigation_logic import quaternion_xyzw_to_yaw

    data = robot.data
    position = _torch_view(data.root_pos_w, field_name="root_pos_w")
    orientation = _torch_view(data.root_quat_w, field_name="root_quat_w")
    origins = _torch_view(env.unwrapped.scene.env_origins, field_name="env_origins")
    linear_velocity = _torch_view(data.root_lin_vel_b, field_name="root_lin_vel_b")
    angular_velocity = _torch_view(data.root_ang_vel_b, field_name="root_ang_vel_b")
    yaw = quaternion_xyzw_to_yaw(orientation)
    pose = [
        float(position[0, 0] - origins[0, 0]),
        float(position[0, 1] - origins[0, 1]),
        float(yaw[0]),
    ]
    velocity = [
        float(linear_velocity[0, 0]),
        float(linear_velocity[0, 1]),
        float(angular_velocity[0, 2]),
    ]
    return pose, velocity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=24102)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=24103)
    parser.add_argument("--camera-stream-host", default="127.0.0.1")
    _add_positive_finite_float_option(parser, "--watchdog-timeout", default=0.25)
    parser.add_argument("--record-dir", default="outputs/datasets")
    parser.add_argument(
        "--data-mode",
        choices=("ground_truth", "lidar_lio", "ros2_direct"),
        default="ground_truth",
    )
    parser.add_argument("--enable-lidar", action="store_true")
    parser.add_argument("--task-yaml", default="", help="IsaacLab-Arena task graph YAML to load")
    parser.add_argument("--scene-yaml", default="", help="IsaacLab-Arena scene graph YAML to load with NoTask")
    parser.add_argument("--task-id", default="", help="First-party OpenFlex task registry ID")
    parser.add_argument("--scene-id", default="", help="First-party scene ID recorded in session metadata")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic reset/policy seed")
    parser.add_argument("--episodes", type=int, default=1, help="Target number of episodes for this session")
    parser.add_argument("--session-metadata", default="", help="Write session provenance JSON to this path")
    parser.add_argument("--camera-stream-port", type=int, default=24104)
    parser.add_argument("--camera-register-port", type=int, default=24106)
    parser.add_argument("--camera-chunk-size", type=int, default=_CAMERA_CHUNK_SIZE_DEFAULT)
    _add_positive_finite_float_option(parser, "--camera-stream-hz", default=15.0)
    _add_positive_finite_float_option(parser, "--record-image-hz", default=15.0)
    parser.add_argument("--render-hz", type=int, default=30)
    parser.add_argument("--physics-hz", type=int, default=120)
    parser.add_argument(
        "--record-mimic-annotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record the mimic_annotation/ group (object/eef/subtask) into saved episodes",
    )
    parser.add_argument(
        "--reset-randomization-json",
        default="{}",
        help="Enabled reset randomization ranges declared for the selected YAML profile",
    )
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=24105)
    parser.add_argument("--synthetic-vr", action="store_true")

    # Keep help usable before a user accepts the Omniverse EULA: importing
    # Isaac Lab bootstraps Kit, which prompts for the license in a first-run
    # environment. The teleoperation-specific options above are sufficient for
    # this offline help path; normal launches still expose AppLauncher flags.
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        parser.print_help()
        return

    from isaaclab.app import AppLauncher
    from mrs_arena.sensors.mid360_lifecycle import configure_sensor_kit_args

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    from mrs_teleoperation.camera_stream import PreviewPeerRegistry, validate_camera_chunk_size

    try:
        args.reset_randomization = json.loads(args.reset_randomization_json)
        if not isinstance(args.reset_randomization, dict):
            raise ValueError("must decode to a JSON object")
    except (json.JSONDecodeError, ValueError) as error:
        parser.error(f"--reset-randomization-json is invalid: {error}")
    if args.task_yaml and args.scene_yaml:
        parser.error("--task-yaml and --scene-yaml are mutually exclusive")
    if args.task_id and (args.task_yaml or args.scene_yaml):
        parser.error("--task-id cannot be combined with --task-yaml/--scene-yaml")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    for argument_name in ("task_yaml", "scene_yaml"):
        value = getattr(args, argument_name)
        if value:
            yaml_path = Path(value).expanduser()
            if not yaml_path.is_file():
                parser.error(f"Arena YAML does not exist: {yaml_path}")
            setattr(args, argument_name, str(yaml_path.resolve()))
    if not 1 <= args.physics_hz <= 500:
        parser.error("--physics-hz must be between 1 and 500")
    if not 1 <= args.render_hz <= min(120, args.physics_hz):
        parser.error("--render-hz must be between 1 and min(120, physics-hz)")
    if not 1 <= args.camera_stream_port <= 65535:
        parser.error("--camera-stream-port must be between 1 and 65535")
    if not 1 <= args.camera_register_port <= 65535:
        parser.error("--camera-register-port must be between 1 and 65535")
    try:
        args.camera_chunk_size = validate_camera_chunk_size(args.camera_chunk_size)
    except ValueError as error:
        parser.error(str(error))
    if not 1.0 <= args.record_image_hz <= 60.0:
        parser.error("--record-image-hz must be between 1 and 60")
    if not 1 <= args.control_port <= 65535:
        parser.error("--control-port must be between 1 and 65535")
    if args.control_host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("--control-host must be a loopback address")
    if args.headless:
        args.visualizer = []
    elif args.visualizer is None:
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
    elif "kit" not in args.visualizer:
        parser.error("VR teleoperation requires the Kit GUI; include 'kit' in --viz")

    args.kit_args = configure_sensor_kit_args(args.kit_args)
    app_launcher = AppLauncher(args)
    env = None
    link = None
    recorder = None
    watchdog = None
    camera_socket = None
    camera_register_bound = False
    camera_peer_registry = None
    capture_api = None
    sensor_api = None
    sensor_manager = None
    sensor_runtime = None
    sim_step = 0
    stop_requested = False
    first_app_running = None
    first_app_exiting = None
    try:
        import torch
        if args.seed is not None:
            torch.manual_seed(args.seed)
            try:
                import numpy as np
                np.random.seed(args.seed)
            except ImportError:
                pass
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.utils.rate_limiter import RateLimiter

        from mrs_robot_lab.adapters.teleop_adapter import JOINT_STATE_NAMES, action_from_command
        from mrs_robot_lab.adapters.synthetic_vr import SyntheticVrSource
        from mrs_robot_lab.assets.asset_resolver import AssetResolver
        from mrs_robot_lab.recorders.arena_sensor_api import (
            ArenaSensorControlServer,
            ArenaSensorManager,
        )
        from mrs_robot_lab.recorders.capture_api import CaptureControlServer
        from mrs_robot_lab.recorders.mimic_annotation import (
            MimicAnnotationCollector,
            mimic_annotation_metadata,
        )
        from mrs_robot_lab.recorders.teleop_episode import EpisodeState, TeleopEpisodeRecorder
        from mrs_teleoperation.safety import CommandWatchdog
        from mrs_teleoperation.udp_link import UdpTeleopLink
        from mrs_arena.environments.openflex_graph_loader import (
            apply_reset_randomization_overrides,
            apply_simulation_rate_overrides,
        )
        from mrs_arena.sensors.mid360_lifecycle import ArenaMid360Runtime

        task_profile = None
        if args.task_id:
            from mrs_arena.environments.openflex_task_registry import (
                get_openflex_task_profile,
                make_openflex_task_environment,
            )

            try:
                task_profile = get_openflex_task_profile(args.task_id)
            except KeyError as error:
                raise RuntimeError(str(error)) from error
            if args.scene_id and args.scene_id != task_profile.scene_id:
                raise RuntimeError(
                    f"task {args.task_id} belongs to scene {task_profile.scene_id}, "
                    f"not {args.scene_id}"
                )
            environment = make_openflex_task_environment(
                args.task_id,
                dataset_dir=args.record_dir,
                enable_cameras=args.enable_cameras,
            )
            args.scene_id = task_profile.scene_id
            print(
                f"ARENA_OPENFLEX_TASK_LOADED {args.task_id} scene={args.scene_id}",
                flush=True,
            )
        elif args.task_yaml:
            from mrs_arena.environments.openflex_graph_loader import (
                make_openflex_environment_from_yaml,
            )

            environment = make_openflex_environment_from_yaml(
                args.task_yaml, enable_cameras=args.enable_cameras
            )
            print(f"ARENA_TASK_YAML_LOADED {args.task_yaml}", flush=True)
        elif args.scene_yaml:
            from mrs_arena.environments.openflex_graph_loader import (
                make_openflex_scene_environment_from_yaml,
            )

            environment = make_openflex_scene_environment_from_yaml(
                args.scene_yaml, enable_cameras=args.enable_cameras
            )
            print(f"ARENA_SCENE_YAML_LOADED {args.scene_yaml} task=NoTask", flush=True)
        else:
            from mrs_arena.environments.openflex_smoke import make_openflex_smoke_environment

            environment = make_openflex_smoke_environment(
                enable_cameras=args.enable_cameras
            )

        apply_reset_randomization_overrides(environment, args.reset_randomization)
        apply_simulation_rate_overrides(
            environment, physics_hz=args.physics_hz, render_hz=args.render_hz
        )
        builder = ArenaEnvBuilder(
            environment,
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=_ARENA_DISABLE_FABRIC,
                device=args.device,
            ),
        )
        env = builder.make_registered()
        env.reset()
        step_dt = float(env.unwrapped.step_dt)
        record_camera_interval = _frame_interval_for_hz(
            step_dt, args.record_image_hz
        )
        stream_camera_interval = _frame_interval_for_hz(
            step_dt, args.camera_stream_hz
        )
        camera_frame_id = 0
        camera_target = (args.camera_stream_host, int(args.camera_stream_port))
        if args.enable_cameras:
            camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            try:
                camera_socket.bind(("0.0.0.0", int(args.camera_register_port)))
                camera_register_bound = True
            except OSError as error:
                print(
                    f"[WARN] ARENA_CAMERA_REGISTER_BIND_FAILED "
                    f"addr=0.0.0.0:{args.camera_register_port} error={error}; "
                    "continuing with configured fallback target",
                    file=sys.stderr,
                    flush=True,
                )
            camera_socket.setblocking(False)
            camera_peer_registry = PreviewPeerRegistry()
            print(
                f"ARENA_CAMERA_STREAM_READY target={camera_target[0]}:{camera_target[1]} "
                f"fps<={args.camera_stream_hz:g} "
                f"register={'0.0.0.0:' + str(args.camera_register_port) if camera_register_bound else 'disabled'}",
                flush=True,
            )
        contract_path = AssetResolver().resolve("openflex_embodiment_contract")
        contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        session_metadata = {
            "schema_version": 1,
            "backend": "arena",
            "data_mode": args.data_mode,
            "enable_lidar": bool(args.enable_lidar),
            "lio_source": (
                "arena_ground_truth"
                if args.data_mode == "ground_truth"
                else "fastlio2_unavailable"
                if args.data_mode == "lidar_lio"
                else None
            ),
            "task_id": args.task_id or None,
            "scene_id": args.scene_id or None,
            "task_yaml": args.task_yaml or None,
            "scene_yaml": args.scene_yaml or None,
            "random_seed": args.seed,
            "randomization": args.reset_randomization,
            "episodes": args.episodes,
            "enable_cameras": bool(args.enable_cameras),
            "record_image_hz": float(args.record_image_hz),
            "physics_hz": int(args.physics_hz),
            "render_hz": int(args.render_hz),
            "embodiment": "openflex",
            "embodiment_contract_sha256": contract_sha256,
            "record_mimic_annotation": bool(args.record_mimic_annotation),
        }
        session_metadata["session_config_sha256"] = hashlib.sha256(
            json.dumps(session_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if args.session_metadata:
            metadata_path = Path(args.session_metadata).expanduser()
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(session_metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"ARENA_SESSION_METADATA {metadata_path}", flush=True)
        link = UdpTeleopLink(
            command_host=args.command_host,
            command_port=args.command_port,
            state_target=(args.state_host, args.state_port),
        )
        watchdog = CommandWatchdog(timeout_seconds=args.watchdog_timeout)
        recorder = TeleopEpisodeRecorder()
        synthetic_source = SyntheticVrSource() if args.synthetic_vr else None
        synthetic_started = False
        synthetic_seq = 0
        episode_index = 0
        state_seq = 0
        local_estop = False
        robot = env.unwrapped.scene["robot"]
        action_shape = tuple(env.action_space.shape)
        if not action_shape or action_shape[-1] != 22:
            raise RuntimeError(f"VR action mapping expects 22 actions, got {action_shape}")
        action_tensor = torch.zeros(action_shape, device=env.unwrapped.device)
        rate_limiter = RateLimiter(period_seconds=env.unwrapped.step_dt)
        last_saved_path = ""
        current_status = "模拟 VR 已就绪" if synthetic_source else "等待 ROS VR 数据"
        capture_api = CaptureControlServer(host=args.control_host, port=args.control_port)
        capture_api.start()
        capture_api.update_status(ready=True, message=current_status)
        mimic_collector = None
        if args.record_mimic_annotation:
            try:
                mimic_collector = MimicAnnotationCollector(
                    robot,
                    env,
                    task_id=args.task_id,
                )
            except (ImportError, RuntimeError, ValueError) as error:
                raise RuntimeError(
                    f"Mimic 标注采集器初始化失败（--record-mimic-annotation 已开启）: {error}"
                ) from error
            print(
                "ARENA_MIMIC_ANNOTATION_READY "
                f"objects={','.join(mimic_collector.object_names) or '-'} "
                f"subtasks={','.join(mimic_collector.subtask_labels) or '-'}",
                flush=True,
            )

        robot_view = getattr(robot, "root_view", None)
        robot_prim_paths = getattr(robot_view, "prim_paths", ())
        if not robot_prim_paths:
            robot_prim_paths = getattr(robot, "prim_paths", ())
        if not robot_prim_paths:
            raise RuntimeError("Arena robot does not expose its instantiated prim path")
        sensor_runtime = ArenaMid360Runtime(
            app=app_launcher.app,
            robot_prim_path=str(robot_prim_paths[0]),
            repository_root=AssetResolver().repository_root,
        )
        sensor_manager = ArenaSensorManager()
        sensor_manager.register(
            "lidar",
            label="Livox MID360",
            topic="/livox/lidar",
            create=sensor_runtime.create_lidar,
        )
        sensor_manager.register(
            "imu",
            label="MID360 IMU",
            topic="/livox/imu",
            create=sensor_runtime.create_imu,
        )
        for sensor_id, label, topic in (
            ("camera_base", "底盘相机", "/cam_base/color/image"),
            ("camera_head", "头部相机", "/cam_head/color/image"),
            ("camera_left", "左手相机", "/cam_left/color/image"),
            ("camera_right", "右手相机", "/cam_right/color/image"),
        ):
            sensor_manager.register_readonly(
                sensor_id,
                label=label,
                topic=topic,
                active=bool(args.enable_cameras),
            )
        sensor_preparation_error = ""
        try:
            sensor_runtime.prepare_sensors()
        except Exception as error:
            sensor_preparation_error = str(error)
            print(
                f"ARENA_SENSOR_PREPARE_ERROR {sensor_preparation_error}",
                file=sys.stderr,
                flush=True,
            )
        sensor_manager.mark_ready(error=sensor_preparation_error)
        sensor_api = ArenaSensorControlServer(sensor_manager, host="127.0.0.1", port=8086)
        sensor_api.start()
        if args.enable_lidar:
            for sensor_id in ("lidar", "imu"):
                threading.Thread(
                    target=sensor_manager.request,
                    args=(sensor_id, "create"),
                    daemon=True,
                ).start()
        print(
            f"VR_TELEOP_GUI_READY command={link.command_address[0]}:{link.command_address[1]} "
            f"state={args.state_host}:{args.state_port} action_dim={action_shape[-1]}",
            flush=True,
        )
        print(f"ARENA_CAPTURE_API_READY {capture_api.base_url}/api/v1/status", flush=True)
        print(f"ARENA_SENSOR_API_READY {sensor_api.base_url}/v1/sensors", flush=True)

        def update_capture_status(message: str | None = None) -> None:
            if capture_api is None:
                return
            capture_api.update_status(
                ready=True,
                recording=recorder.state is EpisodeState.RECORDING,
                steps=recorder.num_steps,
                duration_seconds=recorder.num_steps * step_dt,
                last_saved_path=last_saved_path,
                message=message if message is not None else current_status,
            )

        def reset_environment() -> None:
            """机器人/场景复位（旧 Kit 面板『复位』语义，且真正复位本体）。

            每次正式开始采集新一集前自动调用：录制中先丢弃当前 episode；
            env.reset() 处理场景级随机化（物体/灯光）；机器人本体的关节与
            底盘位姿必须显式写回默认值——实测环境级 reset 不会回写机器人
            （视为「本体复位」缺口），写回后推进一步使其生效并刷新数据；
            最后 watchdog.reset() 要求 VR 手柄重新建立控制输入。
            """
            nonlocal current_status, sim_step
            nonlocal joint_position_before, joint_velocity_before, current_by_name
            if recorder.state is EpisodeState.RECORDING:
                recorder.discard()
            env.reset()
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            robot.write_root_pose_to_sim(robot.data.default_root_pose.clone())
            robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.default_root_vel.clone()))
            env.step(torch.zeros_like(action_tensor))
            watchdog.reset()
            sim_step = 0
            # 重新抓取快照，使本帧记录的状态是复位后的（否则帧 0 残留复位前姿态）
            joint_position_before, current_by_name = _joint_state(robot, "joint_pos")
            joint_velocity_before, _ = _joint_state(robot, "joint_vel")
            current_status = "机器人已复位；等待 VR 手柄重新建立控制输入"
            print("ARENA_ENVIRONMENT_RESET", flush=True)
            update_capture_status()

        def start_episode() -> None:
            nonlocal current_status
            if recorder.state is not EpisodeState.RECORDING:
                # 每次开始采集新一集：先复位机器人/环境，再进入录制
                reset_environment()
            try:
                if mimic_collector is not None:
                    mimic_collector.begin_episode()
                recorder.start()
                current_status = "正在采集：操作完成后保存 Episode，或丢弃本次采集"
            except RuntimeError as error:
                current_status = str(error)
            update_capture_status()

        def save_episode() -> None:
            nonlocal episode_index, last_saved_path, current_status
            try:
                episode_index += 1
                saved_steps = recorder.num_steps
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"openflex_vr_demo_{stamp}_{episode_index:04d}.hdf5"
                saved_path = recorder.save(
                    Path(args.record_dir) / filename,
                    metadata={
                        "robot_id": "openflex",
                        "action_dimension": 22,
                        "joint_state_order": list(JOINT_STATE_NAMES),
                        "control_dt": float(env.unwrapped.step_dt),
                        "protocol_version": 1,
                        "runtime": {
                            "isaac_sim": "6.0",
                            "isaac_lab": "3.0",
                            "arena_release": "0.3.0",
                        },
                        "embodiment_contract_sha256": contract_sha256,
                        "synthetic_vr": bool(synthetic_source),
                        "record_mimic_annotation": bool(
                            mimic_collector is not None and saved_steps > 0
                        ),
                        "mimic_annotation": (
                            mimic_annotation_metadata(mimic_collector)
                            if mimic_collector is not None
                            else {"enabled": False}
                        ),
                        "task_id": args.task_id,
                        "scene_id": args.scene_id,
                        "task_yaml": args.task_yaml,
                        "scene_yaml": args.scene_yaml,
                        "data_mode": args.data_mode,
                        "lio_source": session_metadata["lio_source"],
                        "random_seed": args.seed,
                        "randomization": args.reset_randomization,
                        "session_metadata": session_metadata,
                    },
                )
                last_saved_path = str(saved_path.resolve())
                current_status = f"Episode 已保存：{last_saved_path}（{saved_steps} 步）"
            except (RuntimeError, OSError, ImportError) as error:
                current_status = f"保存失败：{error}"
            update_capture_status()

        def discard_episode() -> None:
            nonlocal current_status
            recorder.discard()
            current_status = "当前 Episode 已丢弃"
            update_capture_status()

        while app_launcher.app.is_running():
            if first_app_running is None:
                first_app_running = app_launcher.app.is_running()
                first_app_exiting = app_launcher.app.is_exiting()
            sensor_manager.process_pending()

            if camera_socket is not None and camera_register_bound:
                while True:
                    now = time.monotonic()
                    if camera_peer_registry.expire(now=now):
                        print("ARENA_CAMERA_PEER_RELEASED", flush=True)
                    try:
                        register_packet, peer_address = camera_socket.recvfrom(65_535)
                    except BlockingIOError:
                        break
                    except OSError as error:
                        print(
                            f"[WARN] ARENA_CAMERA_REGISTER_RECEIVE_FAILED error={error}",
                            file=sys.stderr,
                            flush=True,
                        )
                        break
                    if camera_peer_registry.register(
                        register_packet, peer_address, now=time.monotonic()
                    ):
                        print(
                            f"ARENA_CAMERA_PEER_ACTIVE addr={peer_address[0]}:{peer_address[1]}",
                            flush=True,
                        )
                if camera_peer_registry.expire(now=time.monotonic()):
                    print("ARENA_CAMERA_PEER_RELEASED", flush=True)

            joint_position_before, current_by_name = _joint_state(robot, "joint_pos")
            joint_velocity_before, _ = _joint_state(robot, "joint_vel")
            if args.data_mode == "ground_truth":
                lio_pose_before, lio_velocity_before = _lio_state(robot, env)
            else:
                lio_pose_before, lio_velocity_before = None, None
            incoming = link.receive_latest() if synthetic_source is None else None
            if synthetic_source is not None:
                if not synthetic_started:
                    synthetic_source.start(current_by_name)
                    synthetic_started = True
                synthetic_seq += 1
                incoming = synthetic_source.next_frame(
                    seq=synthetic_seq,
                    source_time_ns=int(sim_step * env.unwrapped.step_dt * 1e9),
                )
            if incoming is not None:
                watchdog.accept(incoming, now=time.monotonic())

            for api_command in capture_api.take_commands():
                if api_command == "start":
                    start_episode()
                elif api_command == "save":
                    save_episode()
                elif api_command == "discard":
                    discard_episode()
                elif api_command == "reset":
                    reset_environment()
                elif api_command == "stop":
                    if recorder.state is EpisodeState.RECORDING:
                        recorder.discard()
                    print("ARENA_STOP_REQUESTED", flush=True)
                    stop_requested = True

            if stop_requested:
                break

            command = watchdog.latest
            actions = action_from_command(
                command,
                current_by_name,
                step_dt=float(env.unwrapped.step_dt),
                motion_enabled=watchdog.motion_enabled(time.monotonic()) and not local_estop,
            )
            action_tensor.zero_()
            row = action_tensor[0] if action_tensor.ndim == 2 else action_tensor
            row.copy_(torch.tensor(actions, device=action_tensor.device, dtype=action_tensor.dtype))
            observations, _step_reward, _step_terminated, _step_truncated, step_extras = env.step(action_tensor)
            if not isinstance(step_extras, dict):
                step_extras = {}
            sim_step += 1
            recorded_camera_frames = None
            capture_camera, record_camera, stream_camera = _camera_step_roles(
                sim_step, record_camera_interval, stream_camera_interval
            )
            if args.enable_cameras and capture_camera:
                observations_by_group = observations.get("camera_obs")
                camera_frames = _encode_camera_observations(observations_by_group)
                if stream_camera and camera_socket is not None:
                    camera_send_targets = [camera_target]
                    if camera_peer_registry is not None:
                        if camera_peer_registry.expire(now=time.monotonic()):
                            print("ARENA_CAMERA_PEER_RELEASED", flush=True)
                        if camera_peer_registry.active_peer is not None:
                            camera_send_targets.append(camera_peer_registry.active_peer)
                    for camera_send_target in dict.fromkeys(camera_send_targets):
                        _send_encoded_camera_observations(
                            camera_socket,
                            camera_send_target,
                            camera_frames,
                            frame_id=camera_frame_id,
                            sim_time_ns=int(sim_step * env.unwrapped.step_dt * 1e9),
                            chunk_size=args.camera_chunk_size,
                        )
                    camera_frame_id = (camera_frame_id + 1) & 0xFFFFFFFF
                if record_camera:
                    recorded_camera_frames = camera_frames

            next_joint_position, _ = _joint_state(robot, "joint_pos")
            next_joint_velocity, _ = _joint_state(robot, "joint_vel")
            state_seq += 1
            link.send_state({
                "type": "state",
                "session_id": command.session_id if command else 0,
                "seq": state_seq,
                "sim_time_ns": int(sim_step * env.unwrapped.step_dt * 1e9),
                "joint_position": next_joint_position,
                "joint_velocity": next_joint_velocity,
            })
            if recorder.state is EpisodeState.RECORDING:
                mimic_sample = None
                if mimic_collector is not None:
                    mimic_sample = mimic_collector.collect(
                        actions=actions,
                        current_by_name=current_by_name,
                        extras=step_extras or None,
                    )
                recorder.append(
                    sim_time_ns=int((sim_step - 1) * env.unwrapped.step_dt * 1e9),
                    action=actions,
                    joint_position=joint_position_before,
                    joint_velocity=joint_velocity_before,
                    next_joint_position=next_joint_position,
                    command_seq=command.seq if command else 0,
                    raw_command=command.to_dict() if command else {},
                    camera_frames=recorded_camera_frames,
                    lio_pose=lio_pose_before,
                    lio_velocity=lio_velocity_before,
                    mimic_annotation=mimic_sample,
                )
                if sim_step % 15 == 0:
                    update_capture_status()

            rate_limiter.sleep()
    finally:
        pending_exception = sys.exc_info()[1]
        if pending_exception is not None:
            import traceback

            traceback.print_exception(
                type(pending_exception),
                pending_exception,
                pending_exception.__traceback__,
                file=sys.__stderr__,
            )
            sys.stderr.flush()
        if not args.headless:
            from mrs_teleoperation.lifecycle import format_runtime_exit

            app = app_launcher.app
            print(
                format_runtime_exit(
                    first_app_running=first_app_running,
                    first_app_exiting=first_app_exiting,
                    app_running=app.is_running(),
                    app_exiting=app.is_exiting(),
                    sim_steps=sim_step,
                    received_vr=watchdog is not None and watchdog.received_at is not None,
                    pending_exception_type=(
                        type(pending_exception).__name__ if pending_exception is not None else None
                    ),
                    pending_exception_message=(
                        str(pending_exception) if pending_exception is not None else None
                    ),
                ),
                flush=True,
            )
        if recorder is not None and recorder.state is EpisodeState.RECORDING:
            recorder.discard()
        if sensor_api is not None:
            sensor_api.stop()
        if sensor_manager is not None:
            try:
                sensor_manager.shutdown()
            except Exception as error:
                print(f"ARENA_SENSOR_CLEANUP_ERROR {error}", file=sys.stderr, flush=True)
        if sensor_runtime is not None:
            try:
                sensor_runtime.shutdown()
            except Exception as error:
                print(f"ARENA_SENSOR_RUNTIME_CLEANUP_ERROR {error}", file=sys.stderr, flush=True)
        if capture_api is not None:
            capture_api.close()
        if link is not None:
            link.close()
        if env is not None:
            env.close()
        if camera_socket is not None:
            camera_socket.close()
        app_launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        raise
