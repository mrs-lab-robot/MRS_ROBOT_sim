#!/usr/bin/env python3
"""在 Arena 注册环境中评估 IsaacLab PPO 导航检查点。

此脚本加载 isaaclab_ext 训练的 PPO 检查点，在 Arena 的 OpenFlexNavigationEmbodiment
环境中运行确定性评估，验证检查点来源，并生成与 isaaclab_ext 兼容的评估报告。

关键特性：
- 完整的来源验证（manifest/checkpoint/task/scene/contract SHA256）
- 使用 Arena 注册的 OpenFlexNavigationTask 和场景
- 8D 导航观测适配（goal-relative odometry）
- 归一化动作到 SI 单位的转换
- 确定性种子范围评估
- 与 isaaclab_ext/scripts/evaluate_navigation_ppo.py 输出格式兼容
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


def _configure_project_paths() -> Path:
    """配置项目路径以导入 arena_benchmark 和 isaaclab_ext 模块。"""
    repo_root = Path(__file__).resolve().parents[3]  # scripts -> arena_benchmark -> MRS_ROBOT_sim -> repo_root
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(repo_root))

    # 添加必要的源码路径
    source_roots = [
        repo_root / "arena_benchmark/src",
        repo_root / "isaaclab_ext/src",
        repo_root / "sim_runtime/teleoperation/src",
        repo_root / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract",
    ]
    for source_root in source_roots:
        if source_root.exists() and str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    return repo_root


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """原子写入 JSON 文件。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    """主评估流程。"""
    repo_root = _configure_project_paths()

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练完成的检查点文件路径 (.pt)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="评估的 episode 数量（默认 10）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="起始随机种子（默认 1000）",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args(argv)

    # 验证参数
    if args.num_episodes < 1:
        parser.error("--num-episodes must be at least 1")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")

    # 加载和验证检查点元数据
    try:
        from mrs_arena.learning.arena_navigation_evaluator import load_checkpoint_metadata

        checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
        print(f"Loading checkpoint metadata from: {checkpoint_path}")

        (
            manifest,
            task_spec_path,
            scene_spec_path,
            contract_path,
            checkpoint_sha256,
        ) = load_checkpoint_metadata(checkpoint_path)

        print(f"  ✓ Manifest validated")
        print(f"  ✓ Task spec: {task_spec_path.name}")
        print(f"  ✓ Scene spec: {scene_spec_path.name}")
        print(f"  ✓ Checkpoint SHA256: {checkpoint_sha256[:16]}...")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(f"Checkpoint validation failed: {error}")

    # 初始化 Isaac Sim/Kit（必须在导入 torch 和 RL 栈之前）
    print(f"\nInitializing Isaac Sim (device={args.device}, headless={args.headless})...")
    app_launcher = AppLauncher(args)
    environment = None

    try:
        import torch
        from isaaclab_arena.assets.registries import AssetRegistry, TaskRegistry
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from mrs_robot_lab.environments.learning.navigation_evaluation import aggregate_episode_results
        from mrs_arena.embodiments.openflex_navigation import OpenFlexNavigationEmbodiment
        from mrs_arena.environments.openflex_navigation import make_openflex_navigation_environment
        from mrs_arena.tasks.openflex_navigation_task import OpenFlexNavigationTask
        from mrs_arena.tasks.navigation_logic import navigation_parameters_from_task
        from mrs_robot_lab.environments.learning.configuration import load_task_configuration
        from mrs_arena.learning.arena_navigation_evaluator import (
            run_arena_navigation_episode,
            make_arena_policy_wrapper,
        )

        # 验证 Arena 注册
        if AssetRegistry().get_asset_by_name(OpenFlexNavigationEmbodiment.name) is not OpenFlexNavigationEmbodiment:
            raise RuntimeError("OpenFlexNavigationEmbodiment was not registered with Arena")
        if TaskRegistry().get_task_by_name(OpenFlexNavigationTask.__name__) is not OpenFlexNavigationTask:
            raise RuntimeError("OpenFlexNavigationTask was not registered with Arena")

        print("  ✓ Arena registrations validated")

        # 构建 Arena 环境
        print("\nBuilding Arena navigation environment...")
        arena_env_cfg = make_openflex_navigation_environment(
            task_spec_path=task_spec_path,
            scene_spec_path=scene_spec_path,
            enable_cameras=False,
        )
        builder = ArenaEnvBuilder(
            arena_env_cfg,
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=True,
                device=args.device,
            ),
        )
        environment = builder.make_registered()
        print(f"  ✓ Environment built (num_envs=1, device={args.device})")

        # 加载任务配置以获取目标位置
        runtime_config = load_task_configuration(
            task_spec_path,
            scene_spec_path=scene_spec_path,
            backend="arena",
        )
        nav_params = navigation_parameters_from_task(runtime_config.task)
        target_xy = tuple(nav_params["target_position"][:2])
        target_yaw_rad = float(nav_params["target_yaw_rad"])
        print(f"  ✓ Navigation target: xy={target_xy}, yaw={target_yaw_rad:.3f} rad")

        # 加载 RSL-RL 检查点
        print("\nLoading RSL-RL checkpoint...")
        wrapped_env = RslRlVecEnvWrapper(environment, clip_actions=1.0)
        runner = OnPolicyRunner(
            wrapped_env,
            manifest["runner_config"],
            log_dir=None,
            device=args.device,
        )
        runner.load(str(checkpoint_path), map_location=args.device)
        rsl_policy = runner.get_inference_policy(device=environment.device)
        arena_policy = make_arena_policy_wrapper(rsl_policy)
        print(f"  ✓ Policy loaded and wrapped for Arena")

        # 获取 asset_cfg（用于访问机器人）
        unwrapped = environment.unwrapped
        asset_cfg = unwrapped.cfg.scene.robot

        # 运行评估
        print(f"\nEvaluating {args.num_episodes} episodes (seeds {args.seed_start}..{args.seed_start + args.num_episodes - 1})...")
        episode_records = []

        for episode_index in range(args.num_episodes):
            episode_seed = args.seed_start + episode_index

            with torch.inference_mode():
                result = run_arena_navigation_episode(
                    environment,
                    arena_policy,
                    target_xy=target_xy,
                    target_yaw_rad=target_yaw_rad,
                    asset_cfg=asset_cfg,
                    index=episode_index,
                    seed=episode_seed,
                    clip_actions=wrapped_env.clip_actions,
                )

            episode_records.append(result)
            status = "✓ success" if result["success"] else "✗ timeout"
            print(f"  Episode {episode_index:2d}: {status} (steps={result['steps']:3d})")

        # 聚合统计
        aggregate = aggregate_episode_results(episode_records)
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"  Success rate: {aggregate['success_count']}/{aggregate['total_episodes']} ({aggregate['success_rate']:.1%})")
        print(f"  Mean steps:   {aggregate['mean_steps']:.1f}")
        print(f"{'='*60}")

        # 保存评估结果
        evaluations_dir = checkpoint_path.parent.parent / "evaluations"
        evaluations_dir.mkdir(exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        result_path = evaluations_dir / f"arena_eval_{timestamp}.json"

        evaluation_output = {
            "schema_version": 2,
            "evaluation_backend": "arena",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "run_manifest": manifest,
            "validated_config_sha256": {
                "task_spec": manifest.get("task_spec_sha256"),
                "scene_spec": manifest.get("scene_spec_sha256"),
                "embodiment_contract": manifest.get("embodiment_contract_sha256"),
                "runner_config": manifest.get("runner_config_sha256"),
                "training_config": manifest.get("training_config_sha256"),
            },
            "arena_environment": {
                "embodiment": OpenFlexNavigationEmbodiment.name,
                "task": OpenFlexNavigationTask.__name__,
                "num_envs": 1,
            },
            "navigation_target": {
                "position_xy": list(target_xy),
                "yaw_rad": target_yaw_rad,
            },
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
                "observation_adapter": "8D goal-relative odometry via navigation_policy_observation",
                "action_adapter": "policy_actions_to_body_twist (normalized to SI)",
            },
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        _write_json_atomic(result_path, evaluation_output)
        print(f"\n✓ Evaluation results saved to: {result_path}")

    finally:
        if environment is not None:
            environment.close()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
