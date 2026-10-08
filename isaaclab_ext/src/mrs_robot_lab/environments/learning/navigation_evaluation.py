"""导航策略评估的辅助函数：episode 结果构建和聚合统计。"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any


def make_episode_result(
    index: int,
    seed: int,
    steps: int,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    """构建单个 episode 的评估结果记录。

    Args:
        index: Episode 索引（从 0 开始）
        seed: Episode 使用的随机种子
        steps: Episode 执行的步数
        terminated: 是否成功到达目标（terminated=True 表示成功）
        truncated: 是否超时（truncated=True 表示超时）

    Returns:
        包含 episode 元数据和状态判断的字典，字段包括：
        index, seed, steps, terminated, truncated, success, timeout

    Raises:
        ValueError: 当 terminated 和 truncated 同时为 True 或同时为 False 时
        ValueError: 当 steps 不是正整数时
    """
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    if terminated and truncated:
        raise ValueError("terminated and truncated cannot both be True")

    if not terminated and not truncated:
        raise ValueError("terminated and truncated cannot both be False")

    return {
        "index": index,
        "seed": seed,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "success": terminated,
        "timeout": truncated,
    }


def aggregate_episode_results(records: list[dict[str, Any]]) -> dict[str, float]:
    """聚合多个 episode 结果，计算汇总统计。

    Args:
        records: episode 结果记录列表（由 make_episode_result 生成）

    Returns:
        包含以下字段的统计字典：
        - total_episodes: 总 episode 数
        - success_count: 成功 episode 数
        - timeout_count: 超时 episode 数
        - success_rate: 成功率（0.0 到 1.0）
        - mean_steps: 平均步数

        空列表返回全零统计。
    """
    if not records:
        return {
            "total_episodes": 0,
            "success_count": 0,
            "timeout_count": 0,
            "success_rate": 0.0,
            "mean_steps": 0.0,
        }

    total_episodes = len(records)
    success_count = sum(1 for record in records if record["success"])
    timeout_count = sum(1 for record in records if record["timeout"])
    success_rate = success_count / total_episodes
    mean_steps = sum(record["steps"] for record in records) / total_episodes

    return {
        "total_episodes": total_episodes,
        "success_count": success_count,
        "timeout_count": timeout_count,
        "success_rate": success_rate,
        "mean_steps": mean_steps,
    }


def validate_checkpoint_provenance(
    checkpoint_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    task_spec_sha256: str | None = None,
    scene_spec_sha256: str | None = None,
    embodiment_contract_sha256: str | None = None,
    runner_config_sha256: str | None = None,
    training_config_sha256: str | None = None,
) -> str:
    """Validate that a selected checkpoint is the one produced by a complete run.

    Schema v2 binds the exact checkpoint bytes and path to the training manifest.
    The optional configuration hashes bind evaluation to the selected task and
    runtime contract. The computed checkpoint digest is returned for the
    evaluation record.
    """
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    checkpoint = checkpoint.resolve()

    if not isinstance(manifest, Mapping):
        raise ValueError("run manifest must be a mapping")
    if manifest.get("status") != "completed":
        raise ValueError(
            f"run manifest status must be 'completed', got {manifest.get('status')!r}"
        )
    if manifest.get("task_id") != "navigation_to_goal":
        raise ValueError(
            "run manifest task_id must be 'navigation_to_goal', "
            f"got {manifest.get('task_id')!r}"
        )
    if manifest.get("schema_version") != 2:
        raise ValueError(
            "run manifest schema_version 2 is required for checkpoint hash verification"
        )

    recorded_checkpoint = manifest.get("checkpoint")
    if not isinstance(recorded_checkpoint, str) or not recorded_checkpoint.strip():
        raise ValueError("run manifest is missing its checkpoint path")
    recorded_path = Path(recorded_checkpoint).expanduser().resolve()
    if recorded_path != checkpoint:
        raise ValueError(
            f"selected checkpoint path does not match checkpoint in manifest: "
            f"{checkpoint} != {recorded_path}"
        )

    expected_hashes = {
        "task_spec_sha256": task_spec_sha256,
        "scene_spec_sha256": scene_spec_sha256,
        "embodiment_contract_sha256": embodiment_contract_sha256,
        "runner_config_sha256": runner_config_sha256,
        "training_config_sha256": training_config_sha256,
    }
    for field, expected in expected_hashes.items():
        if expected is None:
            continue
        if manifest.get(field) != expected:
            raise ValueError(
                f"{field} mismatch: manifest={manifest.get(field)!r}, expected={expected!r}"
            )

    expected_checkpoint_hash = manifest.get("checkpoint_sha256")
    if not isinstance(expected_checkpoint_hash, str) or len(expected_checkpoint_hash) != 64:
        raise ValueError("checkpoint_sha256 is required and must be a SHA256 hex digest")
    actual_checkpoint_hash = _sha256_file(checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError(
            "checkpoint_sha256 mismatch: checkpoint contents differ from the completed run manifest"
        )
    return actual_checkpoint_hash


def run_navigation_episode(
    environment: Any,
    policy,
    *,
    index: int,
    seed: int,
    clip_actions: float | None = 1.0,
) -> dict[str, Any]:
    """Run one seeded episode through the public Isaac Lab DirectRLEnv API.

    The raw environment is stepped (rather than manually advancing private
    simulation methods) so episode counters, success checks, timeouts, and
    reset semantics remain active. A TensorDict observation is built to match
    the RSL-RL inference contract; the default action clip matches the training
    wrapper's ``clip_actions=1.0`` setting.
    """
    import math

    import torch
    from tensordict import TensorDict

    num_envs = int(environment.num_envs)
    if num_envs != 1:
        raise ValueError(f"navigation evaluation requires exactly one env, got {num_envs}")
    max_episode_length = int(environment.max_episode_length)
    if max_episode_length <= 0:
        raise ValueError("environment.max_episode_length must be positive")
    if clip_actions is not None and (
        not math.isfinite(float(clip_actions)) or float(clip_actions) <= 0
    ):
        raise ValueError("clip_actions must be a positive finite value or None")

    observations, _extras = environment.reset(seed=seed)
    if not isinstance(observations, Mapping) or "policy" not in observations:
        raise ValueError("environment.reset() must return observations containing 'policy'")

    for steps in range(1, max_episode_length + 1):
        policy_observations = TensorDict(observations, batch_size=[num_envs])
        actions = policy(policy_observations)
        if not isinstance(actions, torch.Tensor):
            raise TypeError("RSL-RL inference policy must return a torch.Tensor of actions")
        if clip_actions is not None:
            actions = torch.clamp(actions, min=-float(clip_actions), max=float(clip_actions))

        step_result = environment.step(actions)
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise ValueError("DirectRLEnv.step() must return observations, reward, terminated, truncated, extras")
        observations, _rewards, terminated_values, truncated_values, _extras = step_result
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
            raise RuntimeError(
                "DirectRLEnv did not report terminated or truncated at max_episode_length"
            )

    raise RuntimeError("navigation episode exited without a terminal result")


def _single_bool(value: Any, name: str, torch_module) -> bool:
    if isinstance(value, torch_module.Tensor):
        if value.numel() != 1:
            raise ValueError(f"evaluation requires one {name} flag, got shape {tuple(value.shape)}")
        return bool(value.item())
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a bool or a single-element torch.Tensor")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
