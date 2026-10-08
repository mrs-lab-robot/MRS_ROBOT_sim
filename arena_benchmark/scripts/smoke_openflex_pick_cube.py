#!/usr/bin/env python3
"""Exercise the OpenFlex table/cube task and its physical lift predicate."""

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory


def main() -> None:
    print("OPENFLEX_PICK_CUBE_SMOKE: entered main", flush=True)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-steps", type=int, default=3)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.idle_steps < 1:
        parser.error("--idle-steps must be at least 1")

    print(f"OPENFLEX_PICK_CUBE_SMOKE: parsed args device={args.device}", flush=True)
    app_launcher = AppLauncher(args)
    print("OPENFLEX_PICK_CUBE_SMOKE: Kit initialized", flush=True)
    env = None
    dataset_dir = None
    failure = None
    try:
        try:
            import torch

            from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
            from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

            from mrs_arena.environments.openflex_smoke import make_openflex_pick_cube_environment

            print("OPENFLEX_PICK_CUBE_SMOKE: imports complete", flush=True)

            dataset_dir = TemporaryDirectory(prefix="mrs-openflex-pick-cube-")
            arena_environment = make_openflex_pick_cube_environment(dataset_dir.name)
            task = arena_environment.task
            if task is None or task.__class__.__name__ != "OpenFlexPickCubeTask":
                raise RuntimeError("the pick-cube environment did not provide OpenFlexPickCubeTask")

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
            unwrapped = env.unwrapped
            cube = unwrapped.scene["pick_cube"]
            table = unwrapped.scene["pick_cube_table"]

            cube_position = cube.data.root_link_pose_w.torch[0, :3]
            table_position = table.data.root_link_pose_w.torch[0, :3]
            table_surface_z = table_position[2] + 0.02
            cube_bottom_z = cube_position[2] - 0.05
            contact_gap = abs(float((cube_bottom_z - table_surface_z).item()))
            if contact_gap > 0.03:
                raise RuntimeError(
                    f"cube did not reset onto the table: table_surface_z={table_surface_z}, "
                    f"cube_bottom_z={cube_bottom_z}"
                )

            if "success" not in unwrapped.termination_manager.active_terms:
                raise RuntimeError("the pick-cube task has no success termination")
            zero_action = torch.zeros(env.action_space.shape, device=unwrapped.device)
            for _ in range(args.idle_steps):
                observations, _, terminated, truncated, _ = env.step(zero_action)
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError("a stationary cube must not count as a successful pick")

            goal_pose = cube.data.root_link_pose_w.torch.clone()
            goal_pose[:, :3] = torch.tensor(task.goal_position_xyz, device=unwrapped.device)
            cube.write_root_link_pose_to_sim_index(root_pose=goal_pose)
            cube.write_root_link_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=unwrapped.device)
            )
            observations, _, terminated, truncated, _ = env.step(zero_action)
            if not bool(terminated[0].item()) or bool(truncated[0].item()):
                raise RuntimeError(
                    "placing the cube at the configured lift goal did not trigger task success: "
                    f"terminated={terminated}, truncated={truncated}"
                )

            metrics = unwrapped.compute_metrics()
            success_rate = metrics.metric_data_entries["success_rate"].metric_value
            if metrics.num_episodes != 1 or success_rate != 1.0:
                raise RuntimeError(
                    f"expected one successful lift-predicate episode, got episodes={metrics.num_episodes}, "
                    f"success_rate={success_rate}"
                )
            print(
                "OPENFLEX_PICK_CUBE_MDP_SMOKE_OK task=OpenFlexPickCubeTask "
                f"table_cube_contact_gap_m={contact_gap:.4f} success_rate={success_rate} "
                "goal_check=state_injection physical_grasp=not_tested",
                flush=True,
            )
        except BaseException as error:
            failure = error
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        if dataset_dir is not None:
            dataset_dir.cleanup()
        try:
            app_launcher.app.close()
        except SystemExit as error:
            if failure is None and error.code not in (None, 0):
                failure = error
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
