#!/usr/bin/env python3
"""Train the contract-backed flat-ground navigation task with Isaac Lab PPO."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
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


def build_runner_config(*, seed: int, steps_per_env: int, save_interval: int) -> dict[str, Any]:
    """Return a checkpointable config for the RSL-RL API bundled with Isaac Lab."""
    if isinstance(steps_per_env, bool) or int(steps_per_env) <= 0:
        raise ValueError("steps_per_env must be a positive integer")
    if isinstance(save_interval, bool) or int(save_interval) <= 0:
        raise ValueError("save_interval must be a positive integer")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    model = {
        "class_name": "rsl_rl.models.MLPModel",
        "hidden_dims": [256, 128],
        "activation": "elu",
        "obs_normalization": True,
    }
    actor = dict(model)
    actor["distribution_cfg"] = {
        "class_name": "rsl_rl.modules.distribution.GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
    }
    return {
        "seed": int(seed),
        "runner_class_name": "rsl_rl.runners.OnPolicyRunner",
        "num_steps_per_env": int(steps_per_env),
        "save_interval": int(save_interval),
        "logger": "tensorboard",
        "check_for_nan": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "actor": actor,
        "critic": dict(model),
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.005,
            "learning_rate": 3.0e-4,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_gpu": None,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_runner_config(runner_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Snapshot the config before RSL-RL normalizes its input dictionary in place."""
    frozen = copy.deepcopy(runner_config)
    return frozen, _canonical_hash(frozen)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finalize_training_manifest(
    manifest: dict[str, Any],
    *,
    checkpoint_path: Path,
    iterations_completed: int,
    runner_iteration_index: int,
    previous_runner_iteration_index: int,
    completed_at_utc: str,
) -> dict[str, Any]:
    """Return a completed schema-v2 manifest bound to the saved checkpoint bytes."""
    if manifest.get("status") != "running":
        raise ValueError("only a running training manifest can be finalized")
    resolved_checkpoint = checkpoint_path.expanduser().resolve(strict=True)
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {resolved_checkpoint}")

    completed = dict(manifest)
    completed.update({
        "schema_version": 2,
        "status": "completed",
        "iterations_completed_this_run": int(iterations_completed),
        "runner_iteration_index": int(runner_iteration_index),
        "previous_runner_iteration_index": int(previous_runner_iteration_index),
        "checkpoint": str(resolved_checkpoint),
        "checkpoint_sha256": _sha256(resolved_checkpoint),
        "completed_at_utc": completed_at_utc,
    })
    return completed


def main(argv: list[str] | None = None) -> None:
    repo_root = _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=repo_root / "sim_runtime/config/tasks/navigation_to_goal.yaml",
    )
    parser.add_argument("--scene-spec", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "outputs/training/navigation"
    )
    parser.add_argument("--resume-from", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args(argv)
    if args.num_envs < 1:
        parser.error("--num-envs must be at least 1")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.steps_per_env < 1 or args.save_interval < 1:
        parser.error("--steps-per-env and --save-interval must be positive")

    task_path = args.task_spec.expanduser().resolve(strict=True)
    from mrs_robot_lab.environments.learning.configuration import load_task_configuration
    from openflex_isaac_contract.task_spec import load_task_spec

    task = load_task_spec(task_path)
    if task.task_id != "navigation_to_goal":
        parser.error(
            f"this PPO entrypoint only supports navigation_to_goal, got {task.task_id!r}"
        )
    scene_path = (
        args.scene_spec.expanduser().resolve(strict=True)
        if args.scene_spec is not None
        else (task_path.parent.parent / "scenes" / f"{task.scene_id}.yaml").resolve(strict=True)
    )
    load_task_configuration(task_path, scene_spec_path=scene_path, backend="isaac_lab")
    runner_cfg = build_runner_config(
        seed=args.seed,
        steps_per_env=args.steps_per_env,
        save_interval=args.save_interval,
    )
    runner_cfg_snapshot, runner_cfg_sha256 = _freeze_runner_config(runner_cfg)
    contract_path = (
        repo_root
        / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
    ).resolve(strict=True)
    if args.resume_from is not None:
        args.resume_from = args.resume_from.expanduser().resolve(strict=True)
        resume_manifest_path = args.resume_from.parent.parent / "run_manifest.json"
        if not resume_manifest_path.is_file():
            parser.error(
                "--resume-from requires a checkpoint saved by this entrypoint with its run_manifest.json"
            )
        try:
            resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read resume run manifest: {error}")
        fingerprints = {
            "task_id": task.task_id,
            "task_spec_sha256": _sha256(task_path),
            "scene_spec_sha256": _sha256(scene_path),
            "embodiment_contract_sha256": _sha256(contract_path),
            "runner_config_sha256": runner_cfg_sha256,
        }
        mismatches = [
            name
            for name, expected in fingerprints.items()
            if resume_manifest.get(name) != expected
        ]
        if mismatches:
            parser.error(
                "resume checkpoint provenance does not match the selected task/config: "
                + ", ".join(mismatches)
            )

    app_launcher = AppLauncher(args)
    environment = None
    run_dir = None
    manifest_path = None
    manifest: dict[str, Any] | None = None
    try:
        import numpy as np
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        import rsl_rl.algorithms.ppo as rsl_ppo_module
        import rsl_rl.runners.on_policy_runner as rsl_runner_module

        from mrs_robot_lab.environments.learning.navigation_task import make_navigation_task_env
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = args.output_dir.expanduser().resolve() / f"{timestamp}_{task.task_id}_seed{args.seed}"
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir()
        manifest_path = run_dir / "run_manifest.json"
        manifest = {
            "schema_version": 2,
            "status": "running",
            "task_id": task.task_id,
            "task_spec": str(task_path),
            "scene_id": task.scene_id,
            "scene_spec": str(scene_path),
            "task_spec_sha256": _sha256(task_path),
            "scene_spec_sha256": _sha256(scene_path),
            "embodiment_contract_sha256": _sha256(contract_path),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "iterations_requested": args.iterations,
            "steps_per_env": args.steps_per_env,
            "device": str(args.device),
            "resume_from": str(args.resume_from) if args.resume_from else None,
            "runner_config": copy.deepcopy(runner_cfg_snapshot),
            "runner_config_sha256": runner_cfg_sha256,
            "rsl_rl_ppo_source_sha256": _sha256(Path(rsl_ppo_module.__file__).resolve()),
            "rsl_rl_runner_source_sha256": _sha256(Path(rsl_runner_module.__file__).resolve()),
            "training_config_sha256": _canonical_hash({
                "runner_config": runner_cfg_snapshot,
                "num_envs": args.num_envs,
                "iterations": args.iterations,
                "device": str(args.device),
                "task_spec_sha256": _sha256(task_path),
                "scene_spec_sha256": _sha256(scene_path),
            }),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_manifest(manifest_path, manifest)

        environment = _create_task_environment(
            factory=make_navigation_task_env,
            task_spec_path=task_path,
            scene_spec_path=scene_path,
            device=args.device,
            num_envs=args.num_envs,
            seed=args.seed,
        )
        runner_env = RslRlVecEnvWrapper(environment, clip_actions=1.0)
        runner = OnPolicyRunner(
            runner_env,
            copy.deepcopy(runner_cfg_snapshot),
            log_dir=str(run_dir / "logs"),
            device=args.device,
        )
        if args.resume_from is not None:
            runner.load(str(args.resume_from), map_location=args.device)
        previous_iteration = int(runner.current_learning_iteration)
        runner.learn(num_learning_iterations=args.iterations)

        checkpoint_path = run_dir / "checkpoints" / "model_final.pt"
        runner.save(str(checkpoint_path), infos={"task_id": task.task_id})
        manifest = _finalize_training_manifest(
            manifest,
            checkpoint_path=checkpoint_path,
            iterations_completed=args.iterations,
            runner_iteration_index=int(runner.current_learning_iteration),
            previous_runner_iteration_index=previous_iteration,
            completed_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _write_manifest(manifest_path, manifest)
        print(
            "MRS_NAVIGATION_PPO_TRAINING_COMPLETE "
            f"task={task.task_id} iterations={args.iterations} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )
    except BaseException as error:
        if manifest is not None and manifest_path is not None:
            manifest.update({
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            _write_manifest(manifest_path, manifest)
        raise
    finally:
        if environment is not None:
            environment.close()
        app_launcher.app.close()


def _create_task_environment(*, factory, task_spec_path, scene_spec_path, device, num_envs, seed):
    """Call the environment factory with the specified configuration parameters.

    Args:
        factory: Environment factory callable
        task_spec_path: Path to task specification file
        scene_spec_path: Path to scene specification file
        device: Compute device for the environment
        num_envs: Number of parallel environments
        seed: Random seed for environment initialization

    Returns:
        The environment instance created by the factory
    """
    return factory(
        task_spec_path=task_spec_path,
        scene_spec_path=scene_spec_path,
        device=device,
        num_envs=num_envs,
        seed=seed,
    )


def _task_scene_id(task_path: Path) -> str:
    from openflex_isaac_contract.task_spec import load_task_spec

    return load_task_spec(task_path).scene_id


if __name__ == "__main__":
    main()
