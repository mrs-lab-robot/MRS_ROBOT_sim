#!/usr/bin/env python3
"""Launch Arena's no-task OpenFlex scene, probe safe actions, and run zero-action steps."""

from __future__ import annotations

import argparse


def _finite_observation(observation) -> bool:
    import torch

    if isinstance(observation, dict):
        return all(_finite_observation(value) for value in observation.values())
    if isinstance(observation, (tuple, list)):
        return all(_finite_observation(value) for value in observation)
    if isinstance(observation, torch.Tensor):
        return bool(torch.isfinite(observation).all().item())
    return True


def _assert_openflex_action_layout(env) -> None:
    manager = env.unwrapped.action_manager
    expected = {
        "base_twist_action": None,
        "left_arm_action": [f"openarmx_left_joint{index}" for index in range(1, 8)],
        "right_arm_action": [f"openarmx_right_joint{index}" for index in range(1, 8)],
        "lift_action": ["lift_joint"],
        "head_action": ["openarmx_head_yaw_joint", "openarmx_head_pitch_joint"],
        "left_gripper_action": ["openarmx_left_finger_joint1"],
        "right_gripper_action": ["openarmx_right_finger_joint1"],
    }
    if manager.active_terms != list(expected):
        raise RuntimeError(f"unexpected OpenFlex action term order: {manager.active_terms}")
    for term_name, joint_names in expected.items():
        if term_name == "base_twist_action":
            term = manager.get_term(term_name)
            descriptor = term.IO_descriptor
            if term.action_dim != 3 or descriptor.action_type != "BaseTwist":
                raise RuntimeError("OpenFlex base term must expose a three-dimensional BaseTwist action")
            if descriptor.extras.get("fields") != ("linear.x", "linear.y", "angular.z"):
                raise RuntimeError(f"unexpected base action fields: {descriptor.extras.get('fields')}")
            continue
        actual_names = list(manager.get_term(term_name).IO_descriptor.joint_names)
        if actual_names != joint_names:
            raise RuntimeError(f"{term_name} joint order mismatch: expected {joint_names}, got {actual_names}")


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="number of zero-action steps after reset")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    app_launcher = AppLauncher(args)
    env = None
    try:
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

        from mrs_arena.environments.openflex_smoke import make_openflex_smoke_environment

        arena_environment = make_openflex_smoke_environment()
        builder = ArenaEnvBuilder(
            arena_environment,
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=True,
                device=args.device,
            ),
        )
        env = builder.make_registered()
        observations, _ = env.reset()
        _assert_openflex_action_layout(env)
        action_shape = tuple(env.action_space.shape)
        if not action_shape or action_shape[-1] != 22:
            raise RuntimeError(f"expected 22 OpenFlex actions per environment, got action shape {action_shape}")
        if not _finite_observation(observations):
            raise RuntimeError("reset returned a non-finite observation")
        import torch

        zero_action = torch.zeros(action_shape, device=env.unwrapped.device)
        probe_action = zero_action.clone()
        # A brief, contract-bounded probe verifies the custom swerve and relative
        # joint terms actually process and apply actions; the environment closes
        # immediately afterward, so no task or persistent state is involved.
        probe_action[0, 0] = 0.05
        probe_action[0, 3] = 0.01
        observations, _, _, _, _ = env.step(probe_action)
        if not _finite_observation(observations):
            raise RuntimeError("safe-action probe returned a non-finite observation")
        base_term = env.unwrapped.action_manager.get_term("base_twist_action")
        arm_term = env.unwrapped.action_manager.get_term("left_arm_action")
        expected_base = torch.tensor([0.05, 0.0, 0.0], device=zero_action.device)
        if not torch.allclose(base_term.processed_actions[0], expected_base):
            raise RuntimeError(f"base action probe was not processed as expected: {base_term.processed_actions[0]}")
        if not torch.isclose(arm_term.processed_actions[0, 0], torch.tensor(0.0025, device=zero_action.device)):
            raise RuntimeError(f"arm action probe was not processed as expected: {arm_term.processed_actions[0]}")
        for _ in range(args.steps):
            observations, _, _, _, _ = env.step(zero_action)
            if not _finite_observation(observations):
                raise RuntimeError("step returned a non-finite observation")
        print(
            f"OPENFLEX_ARENA_SMOKE_OK action_shape={action_shape} probe=base+left_arm "
            f"terms={tuple(env.unwrapped.action_manager.active_terms)} zero_steps={args.steps} "
            f"observation_keys={tuple(observations.keys())}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
