"""从 task spec（YAML）里取"物体该放在哪"，避免任务类里写死坐标。

这是给新任务用的共享工具：任务类只负责装配，数值一律从这里拿。
背景：`dual_arm_box_transport.py` 把箱子起点/终点写成模块级常量，导致改 YAML 不生效，
训练与评测分布不一致（见 docs/task-authoring.md 的三条规矩）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ObjectPlacement:
    """一个物体在 episode 开始时的位姿（米 / xyzw 四元数）。"""

    object_id: str
    position: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PlacementRandomization:
    """物体初始位姿的随机化幅度；0 表示不随机。"""

    xy_m: float = 0.0
    yaw_rad: float = 0.0


def _spec_mapping(spec: Any) -> Mapping[str, Any]:
    """接受 TaskSpec 对象或已经序列化的 dict。"""

    if isinstance(spec, Mapping):
        return spec
    to_dict = getattr(spec, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError("spec must be a TaskSpec-like object or a mapping")


def _floats(value: Any, length: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{field} must be a sequence of {length} numbers")
    numbers = tuple(float(item) for item in value)
    if any(number != number for number in numbers):  # NaN
        raise ValueError(f"{field} must not contain NaN")
    return numbers


def object_placement_from_spec(spec: Any, object_id: str | None = None) -> ObjectPlacement:
    """读取 `initial_state.<something>_pose` 里该物体的初始位姿。

    YAML 里的键名目前是 `box_pose` 这类「按物体命名」的形式；这里按
    `initial_state` 下所有 `*_pose` 条目查找，`object_id` 用于挑选与校验。
    """

    document = _spec_mapping(spec)
    initial_state = document.get("initial_state") or {}
    if not isinstance(initial_state, Mapping):
        raise ValueError("initial_state must be a mapping")

    candidates = {
        key: value
        for key, value in initial_state.items()
        if key.endswith("_pose") and isinstance(value, Mapping)
    }
    if not candidates:
        raise ValueError("initial_state has no *_pose entry")

    if object_id is None:
        if len(candidates) > 1:
            raise ValueError(
                "initial_state has multiple poses; pass object_id explicitly: "
                + ", ".join(sorted(candidates))
            )
        key, pose = next(iter(candidates.items()))
        resolved_id = key[: -len("_pose")]
    else:
        match = [key for key in candidates if resolved_object_id(key) == object_id]
        if not match:
            raise ValueError(
                f"no initial_state pose for object {object_id!r}; have {sorted(candidates)}"
            )
        key = match[0]
        pose = candidates[key]
        resolved_id = object_id

    position = _floats(pose.get("position"), 3, f"{key}.position")
    rotation = pose.get("rotation_xyzw")
    rotation_tuple = (
        (0.0, 0.0, 0.0, 1.0) if rotation is None else _floats(rotation, 4, f"{key}.rotation_xyzw")
    )
    return ObjectPlacement(object_id=resolved_id, position=position, rotation_xyzw=rotation_tuple)


def resolved_object_id(pose_key: str) -> str:
    """`box_pose` -> `box`；`transport_box_pose` -> `transport_box`。"""

    return pose_key[: -len("_pose")] if pose_key.endswith("_pose") else pose_key


def target_position_from_spec(spec: Any, object_id: str | None = None) -> tuple[float, float, float]:
    """从 success_criteria 里取目标位置（第一个带 target_position 的判据）。"""

    document = _spec_mapping(spec)
    criteria = document.get("success_criteria") or []
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)):
        raise ValueError("success_criteria must be a sequence")
    fallback: tuple[float, float, float] | None = None
    for criterion in criteria:
        if not isinstance(criterion, Mapping):
            continue
        params = criterion.get("params") or {}
        if not isinstance(params, Mapping):
            continue
        if "target_position" not in params:
            continue
        position = _floats(params["target_position"], 3, "success_criteria.target_position")
        criterion_object = params.get("object_id")
        if object_id is None or criterion_object in (None, object_id):
            return position
        # YAML 里的 object_id 和 initial_state 的键名可能不一致（例如 box_pose 对
        # transport_box），先记下来，实在没有精确匹配就用它。
        fallback = fallback or position
    if fallback is not None:
        return fallback
    raise ValueError("no success criterion provides target_position")


def randomization_from_spec(spec: Any) -> PlacementRandomization:
    """读取 `randomization_config` 里的物体随机化幅度。"""

    document = _spec_mapping(spec)
    config = document.get("randomization_config") or {}
    if not isinstance(config, Mapping):
        raise ValueError("randomization_config must be a mapping")
    return PlacementRandomization(
        xy_m=float(config.get("box_xy_m", 0.0) or 0.0),
        yaw_rad=float(config.get("box_yaw_rad", 0.0) or 0.0),
    )


__all__ = [
    "ObjectPlacement",
    "PlacementRandomization",
    "object_placement_from_spec",
    "randomization_from_spec",
    "resolved_object_id",
    "target_position_from_spec",
]
