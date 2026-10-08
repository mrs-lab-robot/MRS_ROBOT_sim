#!/usr/bin/env python3
"""Evaluate a completed navigation PPO run on a deterministic seed range."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


def _configure_project_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(repo_root))
    for source_root in (
        repo_root / "isaaclab_ext/src",
        repo_root / "sim_runtime/teleoperation/src",
        repo_root / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract",
    ):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    return repo_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_and_validate_run_manifest(
    checkpoint_path: Path,
    manifest_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], Path, Path, Path, str]:
    from mrs_robot_lab.environments.learning.navigation_evaluation import (
        validate_checkpoint_provenance,
    )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read run_manifest.json: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("run_manifest.json must contain a JSON object")

    required_fields = (
        "task_spec",
        "scene_spec",
        "runner_config",
        "num_envs",
        "iterations_requested",
        "device",
    )
    missing = [field for field in required_fields if field not in manifest]
    if missing:
        raise ValueError("run_manifest.json is missing: " + ", ".join(missing))
    if not isinstance(manifest["runner_config"], dict):
        raise ValueError("run_manifest.json runner_config must be a JSON object")

    task_spec_path = Path(manifest["task_spec"]).expanduser().resolve(strict=True)
    scene_spec_path = Path(manifest["scene_spec"]).expanduser().resolve(strict=True)
    contract_path = (
        repo_root
        / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
    ).resolve(strict=True)

    task_spec_sha256 = _sha256(task_spec_path)
    scene_spec_sha256 = _sha256(scene_spec_path)
    contract_sha256 = _sha256(contract_path)
    runner_config_sha256 = _canonical_hash(manifest["runner_config"])
    training_config_sha256 = _canonical_hash({
        "runner_config": manifest["runner_config"],
        "num_envs": manifest["num_envs"],
        "iterations": manifest["iterations_requested"],
        "device": manifest["device"],
        "task_spec_sha256": task_spec_sha256,
        "scene_spec_sha256": scene_spec_sha256,
    })
    checkpoint_sha256 = validate_checkpoint_provenance(
        checkpoint_path,
        manifest,
        task_spec_sha256=task_spec_sha256,
        scene_spec_sha256=scene_spec_sha256,
        embodiment_contract_sha256=contract_sha256,
        runner_config_sha256=runner_config_sha256,
        training_config_sha256=training_config_sha256,
    )
    return (
        manifest,
        task_spec_path,
        scene_spec_path,
        contract_path,
        checkpoint_sha256,
    )


def main(argv: list[str] | None = None) -> None:
    repo_root = _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="completed training run checkpoint (.pt)",
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args(argv)

    if args.checkpoint is None:
        parser.error("--checkpoint is required")
    if args.num_episodes < 1:
        parser.error("--num-episodes must be at least 1")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    try:
        checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
        manifest_path = checkpoint_path.parent.parent / "run_manifest.json"
        (
            manifest,
            task_spec_path,
            scene_spec_path,
            contract_path,
            checkpoint_sha256,
        ) = _load_and_validate_run_manifest(checkpoint_path, manifest_path, repo_root)
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))

    # Isaac Sim/Kit must be initialized before importing torch and the RL stack.
    app_launcher = AppLauncher(args)
    environment = None
    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        import rsl_rl.algorithms.ppo as rsl_ppo_module
        import rsl_rl.runners.on_policy_runner as rsl_runner_module

        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            aggregate_episode_results,
            run_navigation_episode,
        )
        from mrs_robot_lab.environments.learning.navigation_task import (
            make_navigation_task_env,
        )

        current_source_hashes = {
            "rsl_rl_ppo_source_sha256": _sha256(Path(rsl_ppo_module.__file__).resolve()),
            "rsl_rl_runner_source_sha256": _sha256(Path(rsl_runner_module.__file__).resolve()),
        }
        mismatches = [
            key
            for key, current_hash in current_source_hashes.items()
            if manifest.get(key) != current_hash
        ]
        if mismatches:
            raise ValueError(
                "installed RSL-RL sources differ from the training run: "
                + ", ".join(mismatches)
            )

        environment = make_navigation_task_env(
            task_spec_path=task_spec_path,
            scene_spec_path=scene_spec_path,
            device=args.device,
            num_envs=1,
            seed=args.seed_start,
        )
        wrapped_env = RslRlVecEnvWrapper(environment, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapped_env,
            manifest["runner_config"],
            log_dir=None,
            device=args.device,
        )
        runner.load(str(checkpoint_path), map_location=args.device)
        policy = runner.get_inference_policy(device=environment.device)

        print(
            f"Evaluating {args.num_episodes} episodes, seeds "
            f"{args.seed_start}..{args.seed_start + args.num_episodes - 1}"
        )
        episode_records = []
        for episode_index in range(args.num_episodes):
            episode_seed = args.seed_start + episode_index
            with torch.inference_mode():
                result = run_navigation_episode(
                    environment,
                    policy,
                    index=episode_index,
                    seed=episode_seed,
                    clip_actions=wrapped_env.clip_actions,
                )
            episode_records.append(result)
            status = "success" if result["success"] else "timeout"
            print(f"  episode={episode_index} status={status} steps={result['steps']}")

        aggregate = aggregate_episode_results(episode_records)
        print(
            f"Summary: success={aggregate['success_count']}/{aggregate['total_episodes']} "
            f"({aggregate['success_rate']:.2%}), mean_steps={aggregate['mean_steps']:.1f}"
        )

        evaluations_dir = checkpoint_path.parent.parent / "evaluations"
        evaluations_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result_path = evaluations_dir / f"eval_{timestamp}.json"
        evaluation_output = {
            "schema_version": 2,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "run_manifest": manifest,
            "validated_config_sha256": {
                "task_spec": manifest["task_spec_sha256"],
                "scene_spec": manifest["scene_spec_sha256"],
                "embodiment_contract": manifest["embodiment_contract_sha256"],
                "runner_config": manifest["runner_config_sha256"],
                "training_config": manifest["training_config_sha256"],
            },
            "rsl_rl_source_sha256": current_source_hashes,
            "num_episodes": args.num_episodes,
            "seed_start": args.seed_start,
            "seed_end": args.seed_start + args.num_episodes - 1,
            "episodes": episode_records,
            "aggregate": aggregate,
            "evaluation_notes": {
                "task_initial_state": "fixed by task_spec",
                "policy_mode": "deterministic inference",
                "generalization": "not verified; initial state is fixed",
                "action_clipping": wrapped_env.clip_actions,
            },
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(result_path, evaluation_output)
        print(f"Evaluation results saved to: {result_path}")
    finally:
        if environment is not None:
            environment.close()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
