#!/usr/bin/env python3
"""Run OpenFlex Arena with the existing ROS 2 VR teleoperation stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path


_CAMERA_OBSERVATION_NAMES = {
    "base_d435": "base_d435_rgb",
    "head_d435": "head_d435_rgb",
    "left_wrist_d405": "left_wrist_d405_rgb",
    "right_wrist_d405": "right_wrist_d405_rgb",
}


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
) -> None:
    """Forward encoded camera frames as bounded UDP chunks for optional preview."""
    from mrs_teleoperation.camera_stream import encode_camera_frame

    for camera_name, jpeg_bytes in camera_frames.items():
        for packet in encode_camera_frame(
            camera_name,
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            jpeg_bytes=jpeg_bytes,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=24102)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=24103)
    parser.add_argument("--camera-stream-host", default="127.0.0.1")
    parser.add_argument("--watchdog-timeout", type=float, default=0.25)
    parser.add_argument("--record-dir", default="outputs/datasets")
    parser.add_argument("--task-yaml", default="", help="IsaacLab-Arena task graph YAML to load")
    parser.add_argument("--scene-yaml", default="", help="IsaacLab-Arena scene graph YAML to load with NoTask")
    parser.add_argument("--camera-stream-port", type=int, default=24104)
    parser.add_argument("--camera-stream-hz", type=float, default=15.0)
    parser.add_argument("--record-image-hz", type=float, default=15.0)
    parser.add_argument("--render-hz", type=int, default=30)
    parser.add_argument("--physics-hz", type=int, default=120)
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

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        args.reset_randomization = json.loads(args.reset_randomization_json)
        if not isinstance(args.reset_randomization, dict):
            raise ValueError("must decode to a JSON object")
    except (json.JSONDecodeError, ValueError) as error:
        parser.error(f"--reset-randomization-json is invalid: {error}")
    if args.task_yaml and args.scene_yaml:
        parser.error("--task-yaml and --scene-yaml are mutually exclusive")
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
    if args.camera_stream_hz <= 0:
        parser.error("--camera-stream-hz must be greater than zero")
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

    app_launcher = AppLauncher(args)
    env = None
    panel = None
    link = None
    recorder = None
    watchdog = None
    camera_socket = None
    capture_api = None
    sim_step = 0
    first_app_running = None
    first_app_exiting = None
    try:
        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.utils.rate_limiter import RateLimiter

        from mrs_robot_lab.adapters.teleop_adapter import JOINT_STATE_NAMES, action_from_command
        from mrs_robot_lab.adapters.synthetic_vr import SyntheticVrSource
        from mrs_robot_lab.assets.asset_resolver import AssetResolver
        from mrs_robot_lab.recorders.capture_api import CaptureControlServer
        from mrs_robot_lab.recorders.teleop_episode import EpisodeState, TeleopEpisodeRecorder
        from mrs_teleoperation.safety import CommandWatchdog
        from mrs_teleoperation.udp_link import UdpTeleopLink
        if not args.headless:
            from mrs_teleoperation.kit_panel import TeleopPanel

        if args.task_yaml:
            from mrs_arena.environments.openflex_graph_loader import (
                apply_reset_randomization_overrides,
                apply_simulation_rate_overrides,
                make_openflex_environment_from_yaml,
            )

            environment = make_openflex_environment_from_yaml(
                args.task_yaml, enable_cameras=args.enable_cameras
            )
            print(f"ARENA_TASK_YAML_LOADED {args.task_yaml}", flush=True)
        elif args.scene_yaml:
            from mrs_arena.environments.openflex_graph_loader import (
                apply_reset_randomization_overrides,
                apply_simulation_rate_overrides,
                make_openflex_scene_environment_from_yaml,
            )

            environment = make_openflex_scene_environment_from_yaml(
                args.scene_yaml, enable_cameras=args.enable_cameras
            )
            print(f"ARENA_SCENE_YAML_LOADED {args.scene_yaml} task=NoTask", flush=True)
        else:
            from mrs_arena.environments.openflex_graph_loader import (
                apply_reset_randomization_overrides,
                apply_simulation_rate_overrides,
            )
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
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, disable_fabric=True, device=args.device),
        )
        env = builder.make_registered()
        env.reset()
        camera_frame_interval = max(
            1, int(round(1.0 / (float(env.unwrapped.step_dt) * args.record_image_hz)))
        )
        camera_frame_id = 0
        camera_target = (args.camera_stream_host, int(args.camera_stream_port))
        if args.enable_cameras:
            camera_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            camera_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            print(
                f"ARENA_CAMERA_STREAM_READY target={camera_target[0]}:{camera_target[1]} "
                f"fps<={args.camera_stream_hz:g}",
                flush=True,
            )
        contract_path = AssetResolver().resolve("openflex_embodiment_contract")
        contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        panel = TeleopPanel() if not args.headless else None
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
        def update_panel_status(message: str) -> None:
            if panel is not None:
                panel.set_status(message)

        update_panel_status(
            f"Arena 已就绪；监听 {link.command_address[0]}:{link.command_address[1]}，{current_status}"
        )
        capture_api = CaptureControlServer(host=args.control_host, port=args.control_port)
        capture_api.start()
        capture_api.update_status(ready=True, message=current_status)
        print(
            f"VR_TELEOP_GUI_READY command={link.command_address[0]}:{link.command_address[1]} "
            f"state={args.state_host}:{args.state_port} action_dim={action_shape[-1]}",
            flush=True,
        )
        print(f"ARENA_CAPTURE_API_READY {capture_api.base_url}/api/v1/status", flush=True)

        def update_capture_status(message: str | None = None) -> None:
            if capture_api is None:
                return
            capture_api.update_status(
                ready=True,
                recording=recorder.state is EpisodeState.RECORDING,
                steps=recorder.num_steps,
                last_saved_path=last_saved_path,
                message=message if message is not None else current_status,
            )

        def start_episode() -> None:
            nonlocal current_status
            try:
                recorder.start()
                current_status = "正在采集：操作完成后保存 Episode，或丢弃本次采集"
            except RuntimeError as error:
                current_status = str(error)
            update_panel_status(current_status)
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
                        "task_yaml": args.task_yaml,
                        "scene_yaml": args.scene_yaml,
                    },
                )
                last_saved_path = str(saved_path.resolve())
                current_status = f"Episode 已保存：{last_saved_path}（{saved_steps} 步）"
            except (RuntimeError, OSError, ImportError) as error:
                current_status = f"保存失败：{error}"
            update_panel_status(current_status)
            update_capture_status()

        def discard_episode() -> None:
            nonlocal current_status
            recorder.discard()
            current_status = "当前 Episode 已丢弃"
            update_panel_status(current_status)
            update_capture_status()

        while app_launcher.app.is_running():
            if first_app_running is None:
                first_app_running = app_launcher.app.is_running()
                first_app_exiting = app_launcher.app.is_exiting()
            for event in panel.pop_events() if panel is not None else ():
                if event == "record_start":
                    start_episode()
                elif event == "record_save":
                    save_episode()
                elif event == "record_discard":
                    discard_episode()
                elif event == "reset":
                    if recorder.state is EpisodeState.RECORDING:
                        recorder.discard()
                    env.reset()
                    watchdog.reset()
                    sim_step = 0
                    update_panel_status("机器人已复位；等待 VR 手柄重新建立控制输入")
                elif event == "estop_toggle":
                    local_estop = not local_estop
                    update_panel_status("本地急停已触发" if local_estop else "本地急停已解除")

            joint_position_before, current_by_name = _joint_state(robot, "joint_pos")
            joint_velocity_before, _ = _joint_state(robot, "joint_vel")
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
            observations, _, _, _, _ = env.step(action_tensor)
            sim_step += 1
            recorded_camera_frames = None
            if args.enable_cameras and sim_step % camera_frame_interval == 0:
                observations_by_group = observations.get("camera_obs")
                recorded_camera_frames = _encode_camera_observations(observations_by_group)
                if camera_socket is not None:
                    _send_encoded_camera_observations(
                        camera_socket,
                        camera_target,
                        recorded_camera_frames,
                        frame_id=camera_frame_id,
                        sim_time_ns=int(sim_step * env.unwrapped.step_dt * 1e9),
                    )
                camera_frame_id = (camera_frame_id + 1) & 0xFFFFFFFF

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
                recorder.append(
                    sim_time_ns=int((sim_step - 1) * env.unwrapped.step_dt * 1e9),
                    action=actions,
                    joint_position=joint_position_before,
                    joint_velocity=joint_velocity_before,
                    next_joint_position=next_joint_position,
                    command_seq=command.seq if command else 0,
                    raw_command=command.to_dict() if command else {},
                    camera_frames=recorded_camera_frames,
                )
                if sim_step % 15 == 0:
                    update_capture_status()

            if sim_step % 15 == 0:
                age = "无 VR" if watchdog.received_at is None else f"{max(0.0, time.monotonic() - watchdog.received_at):.2f}s"
                mode = "本地急停" if local_estop else ("VR ACTIVE" if watchdog.motion_enabled(time.monotonic()) else "安全保持")
                recording = f"录制 {recorder.num_steps} 步" if recorder.state is EpisodeState.RECORDING else "未录制"
                update_panel_status(f"{mode} | 命令年龄 {age} | {recording} | seq={command.seq if command else '-'}")
            rate_limiter.sleep()
    finally:
        pending_exception = sys.exc_info()[1]
        if panel is not None:
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
            if pending_exception is not None:
                import traceback

                traceback.print_exception(
                    type(pending_exception),
                    pending_exception,
                    pending_exception.__traceback__,
                    file=sys.__stderr__,
                )
        if recorder is not None and recorder.state is EpisodeState.RECORDING:
            recorder.discard()
        if panel is not None:
            panel.close()
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
