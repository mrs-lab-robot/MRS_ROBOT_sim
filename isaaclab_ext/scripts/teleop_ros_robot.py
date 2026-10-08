#!/usr/bin/env python3
"""Run OpenFlex Isaac Lab with ROS 2 commands relayed over localhost UDP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


def _configure_project_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(repo_root))
    for source_root in (
        repo_root / "isaaclab_ext/src",
        repo_root / "sim_runtime/teleoperation/src",
    ):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))


def _joint_state(robot, field: str) -> tuple[list[float], dict[str, float]]:
    values = getattr(robot.data, field)[0].detach().to(device="cpu").tolist()
    names = tuple(robot.joint_names)
    by_name = dict(zip(names, values, strict=True))
    from mrs_robot_lab.assets.robot_interface import JOINT_STATE_NAMES

    return [float(by_name[name]) for name in JOINT_STATE_NAMES], by_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=24102)
    parser.add_argument("--state-host", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=24103)
    parser.add_argument("--watchdog-timeout", type=float, default=0.25)
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        parser.print_help()
        return

    _configure_project_paths()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        parser.error("ROS teleoperation GUI requires Kit; omit --headless")
    if args.visualizer is None:
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
    elif "kit" not in args.visualizer:
        parser.error("ROS teleoperation requires Kit; include 'kit' in --viz")

    app_launcher = AppLauncher(args)
    environment = None
    link = None
    try:
        from mrs_robot_lab.adapters.teleop_adapter import (
            action_from_command,
            joint_targets_from_action,
        )
        from mrs_robot_lab.assets.robot_interface import SWERVE_CONFIG
        from mrs_robot_lab.controllers.swerve import compute_swerve_targets
        from mrs_robot_lab.environments.smoke.openflex_smoke import make_openflex_smoke_environment
        from mrs_teleoperation.safety import CommandWatchdog
        from mrs_teleoperation.udp_link import UdpTeleopLink

        environment = make_openflex_smoke_environment(device=args.device, enable_cameras=False)
        environment.reset()
        robot = environment.robot
        link = UdpTeleopLink(
            command_host=args.command_host,
            command_port=args.command_port,
            state_target=(args.state_host, args.state_port),
        )
        watchdog = CommandWatchdog(timeout_seconds=args.watchdog_timeout)
        dt = environment.step_dt
        state_seq = 0
        next_step = time.perf_counter()
        print(
            f"ROS_TELEOP_GUI_READY backend=isaaclab command={link.command_address[0]}:{link.command_address[1]} "
            f"state={args.state_host}:{args.state_port}",
            flush=True,
        )

        while app_launcher.app.is_running():
            incoming = link.receive_latest()
            if incoming is not None:
                watchdog.accept(incoming, now=time.monotonic())

            _positions, current = _joint_state(robot, "joint_pos")
            command = watchdog.latest
            enabled = watchdog.motion_enabled(time.monotonic())
            action = action_from_command(
                command,
                current,
                step_dt=dt,
                motion_enabled=enabled,
            )
            targets = joint_targets_from_action(action, current)
            twist = tuple(action[:3]) if enabled else (0.0, 0.0, 0.0)
            if any(abs(value) > 1e-9 for value in twist):
                swerve = compute_swerve_targets(twist, SWERVE_CONFIG)
                targets.update(
                    zip(SWERVE_CONFIG.steering_joint_names, swerve.steering_angles, strict=True)
                )
                wheel_targets = dict(
                    zip(SWERVE_CONFIG.wheel_joint_names, swerve.wheel_angular_velocities, strict=True)
                )
            else:
                wheel_targets = dict.fromkeys(SWERVE_CONFIG.wheel_joint_names, 0.0)
            environment.set_joint_positions(targets)
            environment.set_joint_velocities(wheel_targets)
            environment.step()

            state_seq += 1
            joint_position, _ = _joint_state(robot, "joint_pos")
            joint_velocity, _ = _joint_state(robot, "joint_vel")
            link.send_state({
                "type": "state",
                "session_id": command.session_id if command else 0,
                "seq": state_seq,
                "sim_time_ns": state_seq * int(dt * 1e9),
                "joint_position": joint_position,
                "joint_velocity": joint_velocity,
            })

            next_step += dt
            remaining = next_step - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                next_step = time.perf_counter()
    finally:
        if link is not None:
            link.close()
        if environment is not None:
            environment.close()
        app_launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        raise
