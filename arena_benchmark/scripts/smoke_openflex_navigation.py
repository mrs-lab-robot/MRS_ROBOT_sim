#!/usr/bin/env python3
"""Build Arena's shared-config navigation task and drive one scripted success."""

from __future__ import annotations

import argparse


def _finite_policy_observation(observations) -> bool:
    import torch

    policy = observations.get("policy") if isinstance(observations, dict) else None
    return (
        isinstance(policy, torch.Tensor)
        and policy.ndim == 2
        and policy.shape[-1] == 8
        and bool(torch.isfinite(policy).all().item())
    )


def _as_cpu_list(value):
    """Render Torch tensors and Isaac Lab ProxyArrays for concise smoke telemetry."""
    value = getattr(value, "torch", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _mass_center_of_robot(robot):
    masses = _as_cpu_list(robot.data.body_mass)[0]
    centers = _as_cpu_list(robot.data.body_com_pos_w)[0]
    total_mass = sum(float(mass) for mass in masses)
    center = [
        sum(float(mass) * float(position[axis]) for mass, position in zip(masses, centers, strict=True))
        / total_mass
        for axis in range(3)
    ]
    largest_links = sorted(
        (
            (name, round(float(mass), 3), [round(float(value), 3) for value in position])
            for name, mass, position in zip(robot.body_names, masses, centers, strict=True)
        ),
        key=lambda entry: entry[1],
        reverse=True,
    )[:8]
    return round(total_mass, 3), [round(value, 4) for value in center], largest_links


def _low_body_heights(robot, threshold: float = 0.20):
    positions = getattr(robot.data.body_pos_w, "torch", robot.data.body_pos_w)
    positions = positions[0].detach().cpu()
    return [
        (name, round(float(positions[index, 2].item()), 4))
        for index, name in enumerate(robot.body_names)
        if float(positions[index, 2].item()) < threshold
    ]


def _wheel_ground_clearances(robot, wheel_radius: float):
    positions = getattr(robot.data.body_pos_w, "torch", robot.data.body_pos_w)[0].detach().cpu()
    return {
        name: round(float(positions[index, 2].item()) - wheel_radius, 4)
        for index, name in enumerate(robot.body_names)
        if name.endswith("_wheel_link")
    }


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    print(f"NAV_SMOKE_PARSED_ARGS device={args.device!r} headless={args.headless!r}", flush=True)

    app_launcher = AppLauncher(args)
    print(f"NAV_SMOKE_POST_LAUNCH_ARGS device={args.device!r}", flush=True)
    env = None
    try:
        import torch
        from isaaclab_arena.assets.registries import AssetRegistry, TaskRegistry
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

        from mrs_arena.embodiments.openflex_navigation import OpenFlexNavigationEmbodiment
        from mrs_arena.environments.openflex_navigation import make_openflex_navigation_environment
        from mrs_arena.policies.navigation import policy_actions_to_body_twist
        from mrs_arena.tasks.openflex_navigation_task import OpenFlexNavigationTask

        if AssetRegistry().get_asset_by_name(OpenFlexNavigationEmbodiment.name) is not OpenFlexNavigationEmbodiment:
            raise RuntimeError("OpenFlexNavigationEmbodiment was not registered with Arena")
        if TaskRegistry().get_task_by_name(OpenFlexNavigationTask.__name__) is not OpenFlexNavigationTask:
            raise RuntimeError("OpenFlexNavigationTask was not registered with Arena")

        builder = ArenaEnvBuilder(
            make_openflex_navigation_environment(),
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=True,
                device=args.device,
            ),
        )
        print(
            "NAV_SMOKE_BUILDER_CONFIG "
            f"device={builder.cfg.device!r} disable_fabric={builder.cfg.disable_fabric!r} "
            f"num_envs={builder.cfg.num_envs}",
            flush=True,
        )
        env = builder.make_registered()
        print("NAV_SMOKE_ENV_BUILT", flush=True)
        observations, _ = env.reset(seed=7)
        print("NAV_SMOKE_ENV_RESET", flush=True)
        unwrapped = env.unwrapped
        env_cfg = unwrapped.cfg
        print(
            "NAV_SMOKE_RESOLVED_CONFIG "
            f"sim_device={env_cfg.sim.device!r} use_fabric={env_cfg.sim.use_fabric!r} "
            f"sim_dt={env_cfg.sim.dt} decimation={env_cfg.decimation}",
            flush=True,
        )
        action_manager = unwrapped.action_manager
        robot = unwrapped.scene["robot"]
        action_term = action_manager.get_term("base_twist_action")
        steering_ids = action_term._steering_ids
        wheel_ids = action_term._wheel_ids
        print(
            "NAV_SMOKE_RESET_PHYSICS "
            f"root_pos={_as_cpu_list(robot.data.root_pos_w)} "
            f"mass_center={_mass_center_of_robot(robot)} "
            f"low_body_heights={_low_body_heights(robot)} "
            f"wheel_clearance_m={_wheel_ground_clearances(robot, action_term._wheel_radius)}",
            flush=True,
        )
        print(
            "NAV_SMOKE_CONTACT_API "
            f"{[name for name in dir(robot.root_physx_view) if 'contact' in name.lower()]}",
            flush=True,
        )
        wheel_clearances = _wheel_ground_clearances(robot, action_term._wheel_radius)
        if len(wheel_clearances) != 4 or min(wheel_clearances.values()) < -0.01:
            message = (
                "robot reset starts with wheel collision geometry below the ground: "
                f"clearances_m={wheel_clearances}"
            )
            print(f"NAV_SMOKE_ABORT RuntimeError: {message}", flush=True)
            raise RuntimeError(message)
        if action_manager.active_terms != ["base_twist_action"]:
            raise RuntimeError(f"navigation must expose only base_twist_action: {action_manager.active_terms}")
        if env.action_space.shape[-1] != 3:
            raise RuntimeError(f"navigation checkpoint requires 3 actions, got {env.action_space.shape}")
        if not _finite_policy_observation(observations):
            raise RuntimeError(f"navigation observation must be finite with shape (1, 8): {observations}")
        if "success" not in unwrapped.termination_manager.active_terms:
            raise RuntimeError("Arena did not compose the navigation success termination")
        if "success_rate" not in unwrapped.metrics_manager.active_terms:
            raise RuntimeError("Arena did not compose the navigation success metric")

        normalized_expert_action = torch.zeros((1, 3), device=unwrapped.device)
        body_twist = policy_actions_to_body_twist(normalized_expert_action)
        print(f"NAV_SMOKE_ACTION={body_twist.tolist()}", flush=True)
        max_steps = int(unwrapped.max_episode_length)
        for step in range(1, max_steps + 1):
            if step == 1 or step % 50 == 0:
                print(f"NAV_SMOKE_STEP_BEGIN={step}", flush=True)
            observations, _reward, terminated, truncated, _extras = env.step(body_twist)
            if step in {1, 2, 5, 10, 25, 50, 100, 200, 400}:
                print(
                    "NAV_SMOKE_PHYSICS "
                    f"step={step} root_pos={_as_cpu_list(robot.data.root_pos_w[0])} "
                    f"root_quat={_as_cpu_list(robot.data.root_quat_w[0])} "
                    f"body_lin_vel={_as_cpu_list(robot.data.root_lin_vel_b[0])} "
                    f"body_ang_vel={_as_cpu_list(robot.data.root_ang_vel_b[0])} "
                    f"steering_pos={_as_cpu_list(robot.data.joint_pos[0, steering_ids])} "
                    f"steering_target={_as_cpu_list(action_term._steering_targets[0])} "
                    f"wheel_speed_mps={_as_cpu_list(robot.data.joint_vel[0, wheel_ids] * action_term._wheel_radius)} "
                    f"wheel_target_mps={_as_cpu_list(action_term._wheel_speed_targets[0])} "
                    f"mass_center={_mass_center_of_robot(robot)} "
                    f"low_body_heights={_low_body_heights(robot)}",
                    flush=True,
                )
            if step == 1 or step % 50 == 0 or bool(terminated[0].item()) or bool(truncated[0].item()):
                print(
                    f"NAV_SMOKE_STEP_RESULT={step} policy={observations['policy'][0].tolist()} "
                    f"terminated={terminated.tolist()} truncated={truncated.tolist()}",
                    flush=True,
                )
            if step == 1:
                print("NAV_SMOKE_FIRST_STEP", flush=True)
            if not _finite_policy_observation(observations):
                raise RuntimeError(f"non-finite navigation observation at step {step}")
            if bool(terminated[0].item()) or bool(truncated[0].item()):
                if bool(truncated[0].item()) or not bool(terminated[0].item()):
                    raise RuntimeError(f"scripted forward policy timed out after {step} steps")
                break
        else:
            raise RuntimeError(f"navigation did not terminate after {max_steps} steps")

        print("NAV_SMOKE_BEFORE_METRICS", flush=True)
        metrics = unwrapped.compute_metrics()
        print("NAV_SMOKE_AFTER_METRICS", flush=True)
        success_rate = metrics.metric_data_entries["success_rate"].metric_value
        if metrics.num_episodes != 1 or success_rate != 1.0:
            raise RuntimeError(
                f"expected one successful navigation episode, got episodes={metrics.num_episodes}, "
                f"success_rate={success_rate}"
            )
        print(
            "OPENFLEX_ARENA_NAVIGATION_SMOKE_OK "
            f"task={OpenFlexNavigationTask.__name__} steps={step} "
            f"action_shape={env.action_space.shape} "
            f"observation_shape={tuple(observations['policy'].shape)} success_rate={success_rate}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        app_launcher.app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(f"NAV_SMOKE_ABORT {type(error).__name__}: {error}", flush=True)
        raise
