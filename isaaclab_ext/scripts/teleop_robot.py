#!/usr/bin/env python3
"""Keyboard teleoperation for OpenFlex using only the runtime and Isaac Lab layers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time


def _configure_project_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(repo_root))
    for source_root in (repo_root / "isaaclab_ext/src", repo_root / "sim_runtime/teleoperation/src"):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))


def main() -> None:
    _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        parser.error("keyboard teleoperation requires the Kit GUI; omit --headless")
    if args.visualizer is None:
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
    elif "kit" not in args.visualizer:
        parser.error("keyboard teleoperation requires the Kit GUI; include 'kit' in --viz")

    app_launcher = AppLauncher(args)
    environment = None
    try:
        import carb
        import omni.appwindow

        from mrs_robot_lab.assets.robot_interface import (
            ACTION_JOINTS_BY_TERM,
            JOINT_POSITION_LIMITS,
            LEFT_ARM_JOINTS,
            RIGHT_ARM_JOINTS,
            SWERVE_CONFIG,
        )
        from mrs_robot_lab.controllers.swerve import compute_swerve_targets
        from mrs_robot_lab.environments.smoke.openflex_smoke import make_openflex_smoke_environment
        from mrs_robot_lab.runners.joint_teleop import clamp_joint_target, physical_jog_delta

        environment = make_openflex_smoke_environment(device=args.device, enable_cameras=False)
        environment.reset()
        robot = environment.robot
        joint_names = tuple(robot.joint_names)
        current_positions = robot.data.joint_pos[0].detach().to(device="cpu").tolist()
        by_name = dict(zip(joint_names, current_positions, strict=True))
        targets = {name: float(by_name[name]) for name in JOINT_POSITION_LIMITS if name in by_name}
        arm_joints = {"left_arm": LEFT_ARM_JOINTS, "right_arm": RIGHT_ARM_JOINTS}

        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        input_interface = carb.input.acquire_input_interface()
        event_type = carb.input.KeyboardEventType
        pressed: set[str] = set()
        selected_arm = ["left_arm"]
        selected_joint = [0]
        quit_requested = [False]
        grippers = {
            "left_gripper": ACTION_JOINTS_BY_TERM["left_gripper_action"][0],
            "right_gripper": ACTION_JOINTS_BY_TERM["right_gripper_action"][0],
        }
        position_jogs = {
            "W": ("arm", 1.0), "S": ("arm", -1.0),
            "R": ("lift_joint", 1.0), "F": ("lift_joint", -1.0),
            "D": ("openarmx_head_yaw_joint", 1.0), "A": ("openarmx_head_yaw_joint", -1.0),
            "Q": ("openarmx_head_pitch_joint", 1.0), "E": ("openarmx_head_pitch_joint", -1.0),
        }
        base_jogs = {
            "I": (0, 0.2), "K": (0, -0.2), "J": (1, 0.2), "L": (1, -0.2),
            "U": (2, 0.35), "O": (2, -0.35),
        }
        position_scales = {"arm": 0.25, "lift_joint": 0.05,
                           "openarmx_head_yaw_joint": 0.15, "openarmx_head_pitch_joint": 0.15}
        number_keys = {name: index - 1 for index in range(1, 8) for name in (str(index), f"KEY_{index}")}

        def on_keyboard_event(event, *_args, **_kwargs) -> bool:
            name = event.input.name
            if event.type == event_type.KEY_PRESS:
                pressed.add(name)
                if name == "TAB":
                    selected_arm[0] = "right_arm" if selected_arm[0] == "left_arm" else "left_arm"
                elif name in number_keys:
                    selected_joint[0] = number_keys[name]
                elif name in {"Z", "X"}:
                    group = "left_gripper" if name == "Z" else "right_gripper"
                    joint = grippers[group]
                    low, high = JOINT_POSITION_LIMITS[joint]
                    targets[joint] = low if targets[joint] > (low + high) / 2 else high
                elif name in {"ESC", "ESCAPE"}:
                    quit_requested[0] = True
            elif event.type == event_type.KEY_RELEASE:
                pressed.discard(name)
            return True

        subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
        print(
            "OpenFlex Isaac Lab teleop ready: I/K forward/back, J/L strafe, U/O rotate; "
            "Tab selects arm, 1-7 selects joint, W/S jog, R/F lift, A/D yaw, Q/E pitch, "
            "Z/X toggle grippers, Esc exits.",
            flush=True,
        )
        dt = 1.0 / 120.0
        try:
            while app_launcher.app.is_running() and not quit_requested[0]:
                start = time.perf_counter()
                deltas: dict[str, float] = {}
                for key, (group, direction) in position_jogs.items():
                    if key not in pressed:
                        continue
                    if group == "arm":
                        joint = arm_joints[selected_arm[0]][selected_joint[0]]
                    else:
                        joint = group
                    delta = physical_jog_delta(direction * 0.02, position_scales[group], control_dt=dt)
                    deltas[joint] = deltas.get(joint, 0.0) + delta
                for joint, delta in deltas.items():
                    targets[joint] = clamp_joint_target(targets[joint], delta, JOINT_POSITION_LIMITS[joint])

                base_twist = [0.0, 0.0, 0.0]
                for key, (index, value) in base_jogs.items():
                    if key in pressed:
                        base_twist[index] += value
                if any(abs(value) > 1e-9 for value in base_twist):
                    swerve = compute_swerve_targets(tuple(base_twist), SWERVE_CONFIG)
                    targets.update(dict(zip(SWERVE_CONFIG.steering_joint_names, swerve.steering_angles, strict=True)))
                    environment.set_joint_velocities(
                        dict(zip(SWERVE_CONFIG.wheel_joint_names, swerve.wheel_angular_velocities, strict=True))
                    )
                else:
                    environment.set_joint_velocities(dict.fromkeys(SWERVE_CONFIG.wheel_joint_names, 0.0))
                environment.set_joint_positions(targets)
                environment.step()
                remaining = dt - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
    finally:
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
