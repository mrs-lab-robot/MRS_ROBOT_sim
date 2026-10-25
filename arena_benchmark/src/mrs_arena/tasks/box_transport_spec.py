"""搬箱任务的参数解析（纯 Python，不 import Isaac，便于单测）。

参数唯一来源是 `sim_runtime/config/tasks/dual_arm_box_transport.yaml`。
这个模块刻意不依赖 isaaclab/isaacsim，这样「改 YAML → 参数跟着变」可以在
没有 Kit 的环境里被测试守住。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from mrs_arena.tasks.spec_placement import (
    object_placement_from_spec,
    randomization_from_spec,
    target_position_from_spec,
)

TASK_OBJECT_ID = "transport_box"
FALLBACK_INITIAL_POSITION = (0.65, 0.0, 0.80)
FALLBACK_TARGET_POSITION = (0.90, 0.0, 0.80)


def default_box_transport_task_path() -> Path:
    """仓库里那份 dual_arm_box_transport.yaml（可用 MRS_ROBOT_SIM_ROOT 覆盖）。"""

    configured_root = os.environ.get("MRS_ROBOT_SIM_ROOT")
    sim_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path(__file__).resolve().parents[4]
    )
    return sim_root / "sim_runtime/config/tasks/dual_arm_box_transport.yaml"


@dataclass(frozen=True)
class BoxTransportParameters:
    """搬箱任务的全部可调参数。"""

    initial_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    min_lift_height_m: float
    min_carry_distance_m: float
    position_tolerance_m: float
    max_linear_speed_mps: float
    stable_seconds: float
    episode_length_s: float
    box_randomization_xy_m: float = 0.0
    box_randomization_yaw_rad: float = 0.0


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number, got {value!r}") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _positive(value: Any, field: str, *, allow_zero: bool = False) -> float:
    number = _finite(value, field)
    if number < 0.0 or (number == 0.0 and not allow_zero):
        raise ValueError(
            f"{field} must be {'non-negative' if allow_zero else 'positive'}, got {number}"
        )
    return number


def _initial_box_position(task: Any) -> tuple[float, float, float]:
    """取箱子初始位置。

    YAML 里这条位姿的键名是 `box_pose`，而判据里的 object_id 是 `transport_box`
    ——两者不一致（历史命名）。先按 object_id 精确找；找不到时，如果 initial_state
    里只有一个非 robot 的 `*_pose`，就用它，避免为了改名去动已有 YAML 和它的读取方。
    """

    try:
        return object_placement_from_spec(task, TASK_OBJECT_ID).position
    except ValueError:
        document = task.to_dict() if hasattr(task, "to_dict") else task
        initial_state = (document or {}).get("initial_state") or {}
        candidates = [
            key for key in initial_state if key.endswith("_pose") and key != "robot_pose"
        ]
        if len(candidates) != 1:
            raise
        position = (initial_state[candidates[0]] or {}).get("position")
        if not isinstance(position, (tuple, list)) or len(position) != 3:
            raise ValueError(f"{candidates[0]}.position must contain three numbers")
        return (
            _finite(position[0], f"{candidates[0]}.position.x"),
            _finite(position[1], f"{candidates[0]}.position.y"),
            _finite(position[2], f"{candidates[0]}.position.z"),
        )


def box_transport_parameters_from_task(task: Any) -> BoxTransportParameters:
    """从 TaskSpec 读出搬箱参数；缺字段才退回仓库默认值。"""

    criteria = [
        criterion
        for criterion in getattr(task, "success_criteria", ())
        if getattr(criterion, "criterion_type", "") == "bilateral_box_transport"
    ]
    if len(criteria) != 1:
        raise ValueError(
            "box transport task must define exactly one bilateral_box_transport criterion"
        )
    criterion = criteria[0]
    params = dict(getattr(criterion, "params", {}) or {})

    initial = _initial_box_position(task)
    target = target_position_from_spec(task, TASK_OBJECT_ID)
    randomization = randomization_from_spec(task)

    return BoxTransportParameters(
        initial_position=initial,
        target_position=target,
        min_lift_height_m=_positive(params.get("min_lift_height", 0.10), "min_lift_height"),
        min_carry_distance_m=_positive(
            params.get("min_carry_distance", 0.20), "min_carry_distance"
        ),
        position_tolerance_m=_positive(getattr(criterion, "tolerance", 0.05), "position tolerance"),
        max_linear_speed_mps=_positive(
            params.get("max_linear_speed_mps", 0.02), "max_linear_speed_mps", allow_zero=True
        ),
        stable_seconds=_positive(params.get("stable_seconds", 1.0), "stable_seconds"),
        episode_length_s=_positive(getattr(task, "episode_length_s", 60.0), "episode_length_s"),
        box_randomization_xy_m=_positive(randomization.xy_m, "box_xy_m", allow_zero=True),
        box_randomization_yaw_rad=_positive(randomization.yaw_rad, "box_yaw_rad", allow_zero=True),
    )


__all__ = [
    "BoxTransportParameters",
    "TASK_OBJECT_ID",
    "box_transport_parameters_from_task",
    "default_box_transport_task_path",
]
