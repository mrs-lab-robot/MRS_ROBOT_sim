#!/usr/bin/env python3
"""Run OpenFlex Arena GUI from the existing ROS 2 command topics via UDP."""

from __future__ import annotations

import argparse
import sys
import time


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
        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

        from mrs_arena.environments.openflex_smoke import make_openflex_smoke_environment
        from mrs_robot_lab.adapters.teleop_adapter import JOINT_STATE_NAMES, action_from_command
        from mrs_teleoperation.safety import CommandWatchdog
        from mrs_teleoperation.udp_link import UdpTeleopLink

        builder = ArenaEnvBuilder(
            make_openflex_smoke_environment(),
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, disable_fabric=True, device=args.device),
        )
        environment = builder.make_registered()
        environment.reset()
        robot = environment.unwrapped.scene["robot"]
        shape = tuple(environment.action_space.shape)
        if not shape or shape[-1] != 22:
            raise RuntimeError(f"OpenFlex ROS adapter expects 22 actions, got {shape}")
        action_tensor = torch.zeros(shape, device=environment.unwrapped.device)
        link = UdpTeleopLink(
            command_host=args.command_host,
            command_port=args.command_port,
            state_target=(args.state_host, args.state_port),
        )
        watchdog = CommandWatchdog(timeout_seconds=args.watchdog_timeout)
        step_dt = float(environment.unwrapped.step_dt)
        state_seq = 0
        next_step = time.perf_counter()
        print(
            f"ROS_TELEOP_GUI_READY backend=arena command={link.command_address[0]}:{link.command_address[1]} "
            f"state={args.state_host}:{args.state_port} action_dim={shape[-1]}",
            flush=True,
        )

        while app_launcher.app.is_running():
            incoming = link.receive_latest()
            if incoming is not None:
                watchdog.accept(incoming, now=time.monotonic())

            names = tuple(robot.joint_names)
            positions = robot.data.joint_pos[0].detach().to(device="cpu").tolist()
            current = dict(zip(names, positions, strict=True))
            command = watchdog.latest
            enabled = watchdog.motion_enabled(time.monotonic())
            actions = action_from_command(
                command,
                current,
                step_dt=step_dt,
                motion_enabled=enabled,
            )
            action_tensor.zero_()
            row = action_tensor[0] if action_tensor.ndim == 2 else action_tensor
            row.copy_(torch.tensor(actions, device=action_tensor.device, dtype=action_tensor.dtype))
            environment.step(action_tensor)

            positions = robot.data.joint_pos[0].detach().to(device="cpu").tolist()
            velocities = robot.data.joint_vel[0].detach().to(device="cpu").tolist()
            state_by_name = dict(zip(names, zip(positions, velocities, strict=True), strict=True))
            state_seq += 1
            link.send_state({
                "type": "state",
                "session_id": command.session_id if command else 0,
                "seq": state_seq,
                "sim_time_ns": state_seq * int(step_dt * 1e9),
                "joint_position": [float(state_by_name[name][0]) for name in JOINT_STATE_NAMES],
                "joint_velocity": [float(state_by_name[name][1]) for name in JOINT_STATE_NAMES],
            })

            next_step += step_dt
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
