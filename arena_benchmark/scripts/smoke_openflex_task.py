#!/usr/bin/env python3
"""Verify OpenFlex task registration, scene composition, reset/step, success, and metric recording."""

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    env = None
    dataset_dir = None
    try:
        from isaaclab_arena.assets.registries import AssetRegistry, TaskRegistry
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

        from mrs_arena.environments.openflex_smoke import make_openflex_task_smoke_environment
        from mrs_arena.embodiments.openflex import OpenFlexEmbodiment
        from mrs_arena.tasks.openflex_smoke_task import OpenFlexSmokeTask

        if AssetRegistry().get_asset_by_name(OpenFlexEmbodiment.name) is not OpenFlexEmbodiment:
            raise RuntimeError("OpenFlexEmbodiment was not registered in Arena's asset registry")
        if TaskRegistry().get_task_by_name(OpenFlexSmokeTask.__name__) is not OpenFlexSmokeTask:
            raise RuntimeError("OpenFlexSmokeTask was not registered in Arena's task registry")

        dataset_dir = TemporaryDirectory(prefix="mrs-openflex-arena-smoke-")
        builder = ArenaEnvBuilder(
            make_openflex_task_smoke_environment(dataset_dir.name),
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=True,
                device=args.device,
            ),
        )
        env = builder.make_registered()
        try:
            observations, _ = env.reset()
            unwrapped = env.unwrapped

            if "success_rate" not in unwrapped.metrics_manager.active_terms:
                raise RuntimeError(f"success_rate metric is not configured: {unwrapped.metrics_manager.active_terms}")
            if "success_rate" not in unwrapped.recorder_manager.active_terms:
                raise RuntimeError(f"success recorder is not configured: {unwrapped.recorder_manager.active_terms}")
            if "success" not in unwrapped.termination_manager.active_terms:
                raise RuntimeError(
                    "success termination is not configured: "
                    f"{unwrapped.termination_manager.active_terms}"
                )

            import torch

            action = torch.zeros(env.action_space.shape, device=unwrapped.device)
            observations, _, terminated, truncated, _ = env.step(action)
            if not bool(terminated[0].item()) or bool(truncated[0].item()):
                raise RuntimeError(
                    "the one-step task should terminate successfully, "
                    f"got terminated={terminated}, truncated={truncated}"
                )
            if unwrapped.get_episode_index(0) != 1:
                raise RuntimeError("Arena did not record and advance the completed smoke-task episode")

            metrics = unwrapped.compute_metrics()
            success_rate = metrics.metric_data_entries["success_rate"].metric_value
            if metrics.num_episodes != 1 or success_rate != 1.0:
                raise RuntimeError(
                    f"expected one recorded successful episode, got episodes={metrics.num_episodes}, "
                    f"success_rate={success_rate}"
                )
            print(
                "OPENFLEX_ARENA_TASK_SMOKE_OK task=OpenFlexSmokeTask episodes=1 "
                f"success_rate={success_rate} observation_keys={tuple(observations.keys())}",
                flush=True,
            )
        finally:
            if env is not None:
                env.close()
                env = None
    finally:
        if env is not None:
            env.close()
        if dataset_dir is not None:
            dataset_dir.cleanup()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
