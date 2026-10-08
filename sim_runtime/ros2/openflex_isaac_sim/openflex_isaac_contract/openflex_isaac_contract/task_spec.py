"""Task specification data structures and YAML loaders.

Framework-independent task description for cross-layer use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SuccessCriterion:
    """任务成功判定条件"""

    criterion_type: str
    params: dict[str, Any]
    tolerance: float = 0.05
    required: bool = True


@dataclass
class TaskSpec:
    """任务定义规范（版本化）"""

    spec_version: str
    task_id: str
    description: str
    scene_id: str
    episode_length_s: float
    success_criteria: list[SuccessCriterion]
    reward_config: dict[str, Any] | None = None
    termination_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    randomization_config: dict[str, Any] = field(default_factory=dict)
    supported_backends: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    action_spec: dict[str, Any] | None = None
    required_sensors: tuple[str, ...] = ()
    allowed_teleop_modes: tuple[str, ...] = ()
    maturity: str = "draft"

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "spec_version": self.spec_version,
            "task_id": self.task_id,
            "description": self.description,
            "scene_id": self.scene_id,
            "episode_length_s": self.episode_length_s,
            "success_criteria": [
                {
                    "criterion_type": crit.criterion_type,
                    "params": crit.params,
                    "tolerance": crit.tolerance,
                    "required": crit.required,
                }
                for crit in self.success_criteria
            ],
            "reward_config": self.reward_config,
            "termination_config": self.termination_config,
            "metadata": self.metadata,
            "initial_state": self.initial_state,
            "randomization_config": self.randomization_config,
            "supported_backends": list(self.supported_backends),
            "capabilities": list(self.capabilities),
            "action_spec": self.action_spec,
            "required_sensors": list(self.required_sensors),
            "allowed_teleop_modes": list(self.allowed_teleop_modes),
            "maturity": self.maturity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        """从字典反序列化"""
        # 验证task_id非空
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id不能为空")

        # 验证episode_length_s为正数
        episode_length_s = data.get("episode_length_s")
        if (
            isinstance(episode_length_s, bool)
            or not isinstance(episode_length_s, (int, float))
            or not math.isfinite(episode_length_s)
            or episode_length_s <= 0
        ):
            raise ValueError(f"episode_length_s必须为正数，实际值: {episode_length_s}")
        episode_length_s = float(episode_length_s)

        # 验证success_criteria非空
        criteria_data = data.get("success_criteria")
        if not isinstance(criteria_data, list) or not criteria_data:
            raise ValueError("success_criteria不能为空列表")

        success_criteria = []
        for crit in criteria_data:
            if not isinstance(crit, dict):
                raise ValueError("success_criteria中的每一项必须是字典")
            criterion_type = crit.get("criterion_type")
            params = crit.get("params", {})
            if not isinstance(criterion_type, str) or not criterion_type.strip():
                raise ValueError("criterion_type不能为空")
            if not isinstance(params, dict):
                raise ValueError(f"{criterion_type}.params必须是字典")
            tolerance = crit.get("tolerance", 0.05)
            if (
                isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance)
                or tolerance <= 0
            ):
                raise ValueError(f"tolerance必须为正数，实际值: {tolerance}")
            required = crit.get("required", True)
            if not isinstance(required, bool):
                raise ValueError("success_criteria.required必须为布尔值")

            success_criteria.append(
                SuccessCriterion(
                    criterion_type=criterion_type,
                    params=dict(params),
                    tolerance=float(tolerance),
                    required=required,
                )
            )

        initial_state = _mapping_field(data, "initial_state")
        randomization_config = _mapping_field(data, "randomization_config")
        action_spec = _mapping_field(data, "action_spec", optional=True)
        supported_backends = _string_tuple_field(data, "supported_backends")
        capabilities = _string_tuple_field(data, "capabilities")
        required_sensors = _string_tuple_field(data, "required_sensors")
        allowed_teleop_modes = _string_tuple_field(data, "allowed_teleop_modes")
        reward_config = _mapping_field(data, "reward_config", optional=True)
        termination_config = _mapping_field(data, "termination_config", optional=True)
        metadata = _mapping_field(data, "metadata")

        return cls(
            spec_version=data.get("spec_version", ""),
            task_id=task_id,
            description=data.get("description", ""),
            scene_id=data.get("scene_id", ""),
            episode_length_s=episode_length_s,
            success_criteria=success_criteria,
            reward_config=reward_config,
            termination_config=termination_config,
            metadata=metadata,
            initial_state=initial_state,
            randomization_config=randomization_config,
            supported_backends=supported_backends,
            capabilities=capabilities,
            action_spec=action_spec,
            required_sensors=required_sensors,
            allowed_teleop_modes=allowed_teleop_modes,
            maturity=str(data.get("maturity", "draft")),
        )

    def save_yaml(self, path: Path | str) -> None:
        """保存为YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def load_task_spec(path: Path | str) -> TaskSpec:
    """从YAML文件加载TaskSpec"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TaskSpec.from_dict(data)


def _mapping_field(data: dict[str, Any], name: str, *, optional: bool = False) -> dict[str, Any] | None:
    value = data.get(name)
    if value is None and optional:
        return None
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name}必须是字典")
    return dict(value)


def _string_tuple_field(data: dict[str, Any], name: str) -> tuple[str, ...]:
    value = data.get(name, [])
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name}必须是非空字符串列表")
    if len(set(value)) != len(value):
        raise ValueError(f"{name}不能包含重复值")
    return tuple(value)
