"""Preflight and launch-spec helpers for the existing ROS 2 Pico VR stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
from typing import Any


@dataclass(frozen=True)
class VrSidecarConfig:
    repo_root: Path
    ros_setup: Path
    underlay_setup: Path
    workspace_root: Path | None
    bridge_script: Path
    domain_id: int
    task_id: str
    isaac_urdf_path: Path | None
    synthetic_input_script: Path | None


def load_vr_sidecar_config(config: Any) -> VrSidecarConfig:
    """Validate all target-host paths before Kit starts."""

    if getattr(config, "teleop_mode", None) != "vr":
        raise ValueError("Isaac Lab 当前只接入 Pico VR 遥操作 sidecar")
    task_id = str(getattr(config, "task_id", ""))
    if task_id not in {"navigation_to_goal", "dual_arm_box_transport"}:
        raise ValueError(f"VR sidecar 不支持任务 {task_id!r}")
    repo_root = _required_directory(
        getattr(config, "simulation_repo_root", ""), "仿真仓库目录"
    )
    metadata = getattr(config, "metadata", {})
    ros_config = metadata.get("ros2_runtime") if isinstance(metadata, dict) else None
    if not isinstance(ros_config, dict):
        raise ValueError("机器配置未提供 ros2_runtime；请先完成 ROS 2/underlay 路径配置")

    ros_setup = _required_file(ros_config.get("ros_setup"), "ROS 2 setup.bash")
    underlay_setup = _required_file(
        ros_config.get("underlay_setup"), "OpenFleX underlay setup.bash"
    )
    workspace = str(ros_config.get("workspace_root", "")).strip()
    workspace_root = _absolute_path(workspace, "OpenFleX 工作区") if workspace else None
    bridge_script = repo_root / "isaaclab_ext" / "scripts" / "worker_ros2_bridge.py"
    if not bridge_script.is_file():
        raise FileNotFoundError(f"Isaac Lab ROS 2 VR bridge 脚本不存在：{bridge_script}")

    raw_domain = ros_config.get("domain_id", 49)
    if isinstance(raw_domain, bool):
        raise ValueError("VR ROS_DOMAIN_ID 必须是 0 到 232 之间的整数")
    try:
        domain_id = int(raw_domain)
    except (TypeError, ValueError) as error:
        raise ValueError("VR ROS_DOMAIN_ID 必须是 0 到 232 之间的整数") from error
    if not 0 <= domain_id <= 232 or str(raw_domain).strip() not in {
        str(domain_id),
        f"{domain_id}.0",
    }:
        raise ValueError("VR ROS_DOMAIN_ID 必须是 0 到 232 之间的整数")

    urdf_path = None
    if task_id == "dual_arm_box_transport":
        urdf_path = _required_file(
            ros_config.get("isaac_urdf_path"), "双臂 VR IK 所需 robot_control_only.urdf"
        )
    synthetic_input_script = None
    if bool(metadata.get("simulated_vr", False)):
        if workspace_root is None:
            raise ValueError("模拟 VR 输入需要配置 OpenFleX 工作区路径")
        synthetic_input_script = _required_file(
            workspace_root
            / "src/openflex_integrated/openflex_gui/openflex_gui/simulated_vr_input.py",
            "模拟 VR 输入脚本",
        )
    return VrSidecarConfig(
        repo_root=repo_root,
        ros_setup=ros_setup,
        underlay_setup=underlay_setup,
        workspace_root=workspace_root,
        bridge_script=bridge_script,
        domain_id=domain_id,
        task_id=task_id,
        isaac_urdf_path=urdf_path,
        synthetic_input_script=synthetic_input_script,
    )


def build_vr_sidecar_script(
    config: Any, sidecar: VrSidecarConfig, api_url: str
) -> str:
    """Build an owned ROS 2 process that reuses Pico IK and task-specific VR."""

    if not api_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        raise ValueError("VR sidecar may only connect to the Worker loopback API")
    if sidecar.task_id != config.task_id:
        raise ValueError("VR sidecar task does not match the frozen SessionConfig")
    arm_task = sidecar.task_id == "dual_arm_box_transport"
    launch_args = [
        "ros2",
        "launch",
        "openflex_isaac_bringup",
        "vr_teleop.launch.py",
        "listen_address:=0.0.0.0",
        "listen_port:=5100",
        "stop_existing_vr:=false",
        f"enable_arms:={'true' if arm_task else 'false'}",
        "enable_head:=false",
        f"enable_chassis:={'false' if arm_task else 'true'}",
        "enable_lift:=false",
    ]
    if sidecar.isaac_urdf_path is not None:
        launch_args.append(f"isaac_urdf:={sidecar.isaac_urdf_path}")

    repo_setup = sidecar.repo_root / "install" / "local_setup.bash"
    bridge_args = [
        "python3",
        str(sidecar.bridge_script),
        "--api-url",
        api_url,
        "--task-id",
        sidecar.task_id,
        "--control-hz",
        str(float(config.frequency.control_hz)),
    ]
    lines = [
        "set -euo pipefail",
        f"source {shlex.quote(str(sidecar.ros_setup))}",
        f"source {shlex.quote(str(sidecar.underlay_setup))}",
        f"if [ -f {shlex.quote(str(repo_setup))} ]; then source {shlex.quote(str(repo_setup))}; fi",
        f"export ROS_DOMAIN_ID={sidecar.domain_id}",
        "export ROS_LOCALHOST_ONLY=0",
        "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
        "unset ROS_DISCOVERY_SERVER",
        f"export ISAACSIM_ROBOT_ROOT={shlex.quote(str(sidecar.repo_root))}",
        "ros2 pkg prefix openflex_isaac_bringup >/dev/null",
        "VR_LAUNCH_PID=",
        "SYNTHETIC_INPUT_PID=",
        "STARTED_VR_LAUNCH=0",
        "cleanup_vr_launch() {",
        '  trap - EXIT INT TERM',
        '  if [ -n "$SYNTHETIC_INPUT_PID" ]; then kill -INT "$SYNTHETIC_INPUT_PID" 2>/dev/null || true; wait "$SYNTHETIC_INPUT_PID" 2>/dev/null || true; fi',
        '  if [ -n "$VR_LAUNCH_PID" ]; then kill -INT "$VR_LAUNCH_PID" 2>/dev/null || true; wait "$VR_LAUNCH_PID" 2>/dev/null || true; fi',
        "}",
        "trap cleanup_vr_launch EXIT INT TERM",
        "ROS_NODE_LIST=\"$(ros2 node list)\"",
        "HAS_PICO_NODE=0",
        "HAS_TASK_NODE=0",
        "grep -Eq '(^|/)pico_pose_bridge$' <<< \"$ROS_NODE_LIST\" && HAS_PICO_NODE=1 || true",
        (
            "grep -Eq '(^|/)openarmx_teleop_vr_node$' <<< \"$ROS_NODE_LIST\" "
            "&& HAS_TASK_NODE=1 || true"
            if arm_task
            else "grep -Eq '(^|/)vr_teleop_chassis$' <<< \"$ROS_NODE_LIST\" "
            "&& HAS_TASK_NODE=1 || true"
        ),
        'if [ "$HAS_PICO_NODE" = "1" ] && [ "$HAS_TASK_NODE" = "1" ]; then',
        '  echo "Reusing the existing Pico/VR ROS 2 stack"',
        'elif [ "$HAS_PICO_NODE" = "1" ] || [ "$HAS_TASK_NODE" = "1" ]; then',
        '  echo "A partial Pico/VR ROS 2 stack already exists; refusing to start a duplicate" >&2',
        "  exit 8",
        "else",
        "  " + shlex.join(launch_args) + " &",
        "  VR_LAUNCH_PID=$!",
        "  STARTED_VR_LAUNCH=1",
        "fi",
    ]
    if sidecar.synthetic_input_script is not None:
        lines.extend(
            (
                'if [ "$STARTED_VR_LAUNCH" = "1" ]; then',
                shlex.join(
                    [
                        "python3",
                        str(sidecar.synthetic_input_script),
                        "--ros-args",
                        "-p",
                        "publish_rate_hz:=30.0",
                    ]
                )
                + " &",
                "  SYNTHETIC_INPUT_PID=$!",
                "fi",
            )
        )
    lines.append(shlex.join(bridge_args))
    return "\n".join(lines)


def _required_file(value: Any, name: str) -> Path:
    path = _absolute_path(value, name)
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在：{path}")
    return path


def _required_directory(value: Any, name: str) -> Path:
    path = _absolute_path(value, name)
    if not path.is_dir():
        raise FileNotFoundError(f"{name} 不存在：{path}")
    return path


def _absolute_path(value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"缺少{name}绝对路径")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{name}必须是目标机器上的绝对路径")
    return path


__all__ = ["VrSidecarConfig", "build_vr_sidecar_script", "load_vr_sidecar_config"]
