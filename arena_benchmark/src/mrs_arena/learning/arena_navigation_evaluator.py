"""Arena 端的导航 PPO 检查点评估器。

此模块加载 isaaclab_ext 训练生成的 RSL-RL PPO 检查点，
在 Arena 注册的 OpenFlex 导航环境中评估，使用：
- 8D 导航观测（goal-relative odometry）
- 归一化到 SI 的动作适配器
- 确定性种子的多 episode 评估
- 完整的来源验证（manifest/checkpoint/task/scene/contract hash）
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


def run_arena_navigation_episode(
    env,
    policy,
    *,
    target_xy: tuple[float, float],
    target_yaw_rad: float,
    asset_cfg,
    index: int,
    seed: int,
    clip_actions: float | None = 1.0,
) -> dict[str, Any]:
    """在 Arena 环境中运行单个导航 episode。

    Args:
        env: Arena DirectRLEnv 环境实例
        policy: 接受观测并返回归一化动作的策略函数
        target_xy: 目标位置 (x, y) 米
        target_yaw_rad: 目标朝向（弧度）
        asset_cfg: 机器人资产配置（用于访问场景）
        index: Episode 索引
        seed: 随机种子
        clip_actions: 动作裁剪范围（None 则不裁剪）

    Returns:
        Episode 结果字典，包含 index, seed, steps, terminated, truncated, success, timeout

    Raises:
        ValueError: 环境配置不满足单环境要求或最大步数非正
        RuntimeError: Episode 未正常终止
    """
    import math
    import torch

    # 延迟导入共享结果构建器（避免在模块级导入 isaaclab_ext）
    from mrs_robot_lab.environments.learning.navigation_evaluation import make_episode_result
    from mrs_arena.adapters.navigation_observation import navigation_policy_observation
    from mrs_arena.policies.navigation import policy_actions_to_body_twist

    num_envs = int(env.num_envs)
    if num_envs != 1:
        raise ValueError(f"arena navigation evaluation requires exactly one env, got {num_envs}")
    max_episode_length = int(env.max_episode_length)
    if max_episode_length <= 0:
        raise ValueError(f"environment.max_episode_length must be positive, got {max_episode_length}")
    if clip_actions is not None and (
        not math.isfinite(float(clip_actions)) or float(clip_actions) <= 0
    ):
        raise ValueError("clip_actions must be a positive finite value or None")

    env.reset(seed=seed)

    for steps in range(1, max_episode_length + 1):
        # 构建 8D 导航观测
        policy_obs = navigation_policy_observation(
            env,
            target_xy=target_xy,
            target_yaw_rad=target_yaw_rad,
            asset_cfg=asset_cfg,
        )
        observations = {"policy": policy_obs}

        # 策略推理：归一化动作
        normalized_actions = policy(observations)
        if not isinstance(normalized_actions, torch.Tensor):
            raise TypeError("policy must return a torch.Tensor of normalized actions")
        if clip_actions is not None:
            normalized_actions = torch.clamp(
                normalized_actions,
                min=-float(clip_actions),
                max=float(clip_actions),
            )

        # 转换为 SI 单位的 body twist
        body_twist = policy_actions_to_body_twist(normalized_actions)

        # 执行一步
        step_result = env.step(body_twist)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise ValueError("DirectRLEnv.step() must return 5-tuple")
        _observations, _rewards, terminated_values, truncated_values, _extras = step_result

        terminated = _single_bool(terminated_values, "terminated", torch)
        truncated = _single_bool(truncated_values, "truncated", torch)

        if terminated or truncated:
            return make_episode_result(
                index=index,
                seed=seed,
                steps=steps,
                terminated=terminated,
                truncated=truncated,
            )

        if steps == max_episode_length:
            raise RuntimeError("DirectRLEnv did not report terminated or truncated at max_episode_length")

    raise RuntimeError("navigation episode exited without a terminal result")


def load_checkpoint_metadata(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], Path, Path, Path, str]:
    """加载并验证检查点元数据。

    Args:
        checkpoint_path: 检查点文件路径 (.pt)

    Returns:
        (manifest, task_spec_path, scene_spec_path, contract_path, checkpoint_sha256)

    Raises:
        FileNotFoundError: 检查点或 manifest 文件不存在
        ValueError: Manifest 验证失败
    """
    from mrs_robot_lab.environments.learning.navigation_evaluation import validate_checkpoint_provenance

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    manifest_path = checkpoint.parent.parent / "run_manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"run_manifest.json not found at {manifest_path}; "
            f"checkpoint must be in <run_dir>/checkpoints/"
        )

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read run_manifest.json: {error}") from error

    if not isinstance(manifest, dict):
        raise ValueError("run_manifest.json must contain a JSON object")

    if manifest.get("status") != "completed":
        raise ValueError(
            "run manifest status must be 'completed', "
            f"got {manifest.get('status')!r}"
        )

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

    # 解析关键配置路径
    task_spec_path = Path(manifest["task_spec"]).expanduser().resolve(strict=True)
    scene_spec_path = Path(manifest["scene_spec"]).expanduser().resolve(strict=True)

    # isaaclab_ext 的 evaluate_navigation_ppo.py 使用相对路径获取 contract
    # 这里我们需要找到正确的 contract 路径
    repo_root = checkpoint.parents[3]  # checkpoint -> checkpoints -> run_dir -> isaaclab_ext -> repo_root
    contract_path = (
        repo_root / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
    ).resolve(strict=True)

    # 计算配置哈希
    task_spec_sha256 = _sha256_file(task_spec_path)
    scene_spec_sha256 = _sha256_file(scene_spec_path)
    contract_sha256 = _sha256_file(contract_path)
    runner_config_sha256 = _canonical_hash(manifest["runner_config"])
    training_config_sha256 = _canonical_hash({
        "runner_config": manifest["runner_config"],
        "num_envs": manifest["num_envs"],
        "iterations": manifest["iterations_requested"],
        "device": manifest["device"],
        "task_spec_sha256": task_spec_sha256,
        "scene_spec_sha256": scene_spec_sha256,
    })

    # 验证检查点来源
    checkpoint_sha256 = validate_checkpoint_provenance(
        checkpoint,
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


def make_arena_policy_wrapper(rsl_rl_policy):
    """将 RSL-RL 策略包装为 Arena 兼容接口。

    RSL-RL 推理策略期望 TensorDict 输入，Arena 使用普通字典。

    Args:
        rsl_rl_policy: RSL-RL OnPolicyRunner.get_inference_policy() 返回的策略

    Returns:
        接受 dict[str, Tensor] 并返回 Tensor 的策略函数
    """
    from tensordict import TensorDict

    def arena_policy(observations: dict):
        """Arena 策略接口：dict -> Tensor."""
        policy_obs = TensorDict(observations, batch_size=[1])
        return rsl_rl_policy(policy_obs)

    return arena_policy


def _single_bool(value: Any, name: str, torch_module) -> bool:
    """从 Tensor 或 bool 提取单个布尔值。"""
    if isinstance(value, torch_module.Tensor):
        if value.numel() != 1:
            raise ValueError(f"evaluation requires one {name} flag, got shape {tuple(value.shape)}")
        return bool(value.item())
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool or a single-element torch.Tensor")


def _sha256_file(path: Path) -> str:
    """计算文件的 SHA256 哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    """计算 JSON 序列化后的规范哈希。"""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
