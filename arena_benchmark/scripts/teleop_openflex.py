#!/usr/bin/env python3
"""Keyboard joint-jog teleoperation for the no-task OpenFlex Arena environment."""

from __future__ import annotations

import argparse
import sys

_JOG_MAGNITUDE = 0.02


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.headless:
        parser.error("keyboard teleoperation requires the Kit GUI; omit --headless")
    if args.visualizer is None:
        # Isaac Lab 3.0 defaults to headless unless a visualizer is selected.
        # Keyboard input requires the Kit window and its appwindow extension.
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
    elif "kit" not in args.visualizer:
        parser.error("keyboard teleoperation requires the Kit GUI; include 'kit' in --viz")

    app_launcher = AppLauncher(args)
    env = None
    try:
        print("[TELEOP] Kit GUI initialized; loading Arena teleop components.", file=sys.__stderr__, flush=True)
        import carb
        import omni.appwindow
        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_arena.utils.rate_limiter import RateLimiter

        from mrs_arena.environments.openflex_smoke import make_openflex_smoke_environment
        from mrs_robot_lab.runners.joint_teleop import (
            ACTION_DIMENSION,
            action_vector_for_key,
            toggle_binary_gripper,
        )

        print("[TELEOP] Building the no-task OpenFlex Arena environment.", file=sys.__stderr__, flush=True)
        builder = ArenaEnvBuilder(
            make_openflex_smoke_environment(),
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=True,
                device=args.device,
            ),
        )
        env = builder.make_registered()
        print("[TELEOP] Environment created; resetting robot.", file=sys.__stderr__, flush=True)
        env.reset()
        print("[TELEOP] Robot reset; binding keyboard input.", file=sys.__stderr__, flush=True)
        action_shape = tuple(env.action_space.shape)
        if not action_shape or action_shape[-1] != ACTION_DIMENSION:
            raise RuntimeError(f"keyboard mapping expects {ACTION_DIMENSION} actions, got {action_shape}")

        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        input_interface = carb.input.acquire_input_interface()
        event_type = carb.input.KeyboardEventType
        pressed: set[str] = set()
        selected_arm = ["left_arm"]
        selected_joint = [0]
        grippers = {"left_gripper": 1.0, "right_gripper": 1.0}
        quit_requested = [False]
        held_jogs = {
            "I": ("base_twist", 0, 0.2), "K": ("base_twist", 0, -0.2),
            "J": ("base_twist", 1, 0.2), "L": ("base_twist", 1, -0.2),
            "U": ("base_twist", 2, 0.35), "O": ("base_twist", 2, -0.35),
            "W": ("arm", 0, 1.0), "S": ("arm", 0, -1.0),
            "R": ("lift", 0, 1.0), "F": ("lift", 0, -1.0),
            "A": ("head", 0, -1.0), "D": ("head", 0, 1.0),
            "Q": ("head", 1, 1.0), "E": ("head", 1, -1.0),
        }
        number_keys = {name: i - 1 for i in range(1, 8) for name in (str(i), f"KEY_{i}")}

        def on_keyboard_event(event, *_args, **_kwargs) -> bool:
            input_name = event.input.name
            if event.type == event_type.KEY_PRESS:
                pressed.add(input_name)
                if input_name == "TAB":
                    selected_arm[0] = "right_arm" if selected_arm[0] == "left_arm" else "left_arm"
                    print(f"Selected {selected_arm[0]}", flush=True)
                elif input_name in number_keys:
                    selected_joint[0] = number_keys[input_name]
                    print(f"Selected joint {selected_joint[0] + 1} on {selected_arm[0]}", flush=True)
                elif input_name == "Z":
                    grippers["left_gripper"] = toggle_binary_gripper(grippers["left_gripper"])
                elif input_name == "X":
                    grippers["right_gripper"] = toggle_binary_gripper(grippers["right_gripper"])
                elif input_name in {"ESC", "ESCAPE"}:
                    quit_requested[0] = True
            elif event.type == event_type.KEY_RELEASE:
                pressed.discard(input_name)
            return True

        subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
        print(
            "OpenFlex Arena teleop ready: I/K=base forward/back, J/L=left/right, U/O=rotate, "
            "Tab=select arm, 1-7=joint, W/S=jog, R/F=lift, A/D=head yaw, Q/E=head pitch, "
            "Z/X=grippers, Esc=exit.",
            file=sys.__stderr__,
            flush=True,
        )
        actions = torch.zeros(action_shape, device=env.unwrapped.device)
        action_row = actions[0] if actions.ndim == 2 else actions
        rate_limiter = RateLimiter(period_seconds=env.unwrapped.step_dt)
        try:
            while app_launcher.app.is_running() and not quit_requested[0]:
                actions.zero_()
                for input_key, jog in held_jogs.items():
                    if input_key not in pressed:
                        continue
                    group, index, direction = jog
                    if group == "arm":
                        vector = action_vector_for_key(selected_arm[0], selected_joint[0], direction * _JOG_MAGNITUDE)
                    elif group == "base_twist":
                        vector = action_vector_for_key(group, index, direction)
                    elif group == "lift":
                        vector = action_vector_for_key("lift", 0, direction * _JOG_MAGNITUDE)
                    else:
                        vector = action_vector_for_key(group, index, direction * _JOG_MAGNITUDE)
                    action_row += torch.tensor(vector, device=actions.device)
                action_row[20] = grippers["left_gripper"]
                action_row[21] = grippers["right_gripper"]
                env.step(actions)
                rate_limiter.sleep()
        finally:
            input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
    finally:
        if env is not None:
            env.close()
        app_launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        raise
