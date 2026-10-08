#!/usr/bin/env python3
"""Evaluate an Isaac Lab policy checkpoint on a first-party OpenFlex Arena task."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys


def _configure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    for relative in (
        "arena_benchmark/src",
        "isaaclab_ext/src",
        "sim_runtime/teleoperation/src",
        "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract",
    ):
        path = repo_root / relative
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(repo_root))
    return repo_root


def main(argv: list[str] | None = None) -> None:
    _configure_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--result-path", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args(argv)
    if args.num_episodes < 1:
        parser.error("--num-episodes must be at least 1")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    app_launcher = AppLauncher(args)
    environment = None
    try:
        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from mrs_arena.environments.openflex_task_registry import (
            get_openflex_task_profile,
            make_openflex_task_environment,
        )
        from mrs_arena.learning.arena_openflex_evaluator import (
            make_episode_result,
            policy_input,
            single_bool,
        )

        profile = get_openflex_task_profile(args.task_id)
        checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        if checkpoint.is_dir():
            candidates = sorted(checkpoint.rglob("*.pt"))
            if not candidates:
                raise FileNotFoundError(f"no .pt checkpoint under {checkpoint}")
            checkpoint = candidates[-1]
        manifest_path = checkpoint.parent.parent / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        environment_cfg = make_openflex_task_environment(args.task_id, enable_cameras=False)
        environment = ArenaEnvBuilder(
            environment_cfg,
            ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, disable_fabric=True, device=args.device),
        ).make_registered()
        wrapped = RslRlVecEnvWrapper(environment, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapped,
            manifest.get("runner_config", {}),
            log_dir=None,
            device=args.device,
        )
        runner.load(str(checkpoint), map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        records = []
        max_steps = int(environment.max_episode_length)
        for index in range(args.num_episodes):
            seed = args.seed_start + index
            try:
                observations = environment.reset(seed=seed)
            except TypeError:
                observations = environment.reset()
            if isinstance(observations, tuple):
                observations = observations[0]
            success = False
            extras = {}
            for steps in range(1, max_steps + 1):
                with torch.inference_mode():
                    action = policy(policy_input(observations))
                step_result = environment.step(action)
                observations, _reward, terminated, truncated, extras = step_result
                if single_bool(terminated) or single_bool(truncated):
                    success = bool(extras.get("success", single_bool(terminated))) if isinstance(extras, dict) else single_bool(terminated)
                    records.append(make_episode_result(task_id=args.task_id, index=index, seed=seed, steps=steps, success=success, extras=extras if isinstance(extras, dict) else {}))
                    break
            else:
                records.append(make_episode_result(task_id=args.task_id, index=index, seed=seed, steps=max_steps, success=False, extras={"timeout": True}))
        result = {
            "schema_version": 1,
            "evaluation_backend": "arena",
            "task_id": args.task_id,
            "scene_id": profile.scene_id,
            "checkpoint": str(checkpoint),
            "num_episodes": args.num_episodes,
            "seed_start": args.seed_start,
            "episodes": records,
            "success_rate": sum(item["success"] for item in records) / len(records),
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        result_path = args.result_path.expanduser()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"ARENA_EVALUATION_RESULT {result_path}", flush=True)
    finally:
        if environment is not None:
            environment.close()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
