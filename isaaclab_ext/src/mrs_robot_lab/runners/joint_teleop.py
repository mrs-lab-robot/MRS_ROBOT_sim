"""Simulator-independent mapping from keyboard jogs to OpenFlex joint actions."""

from __future__ import annotations

import math
from collections.abc import Mapping

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, ACTION_SLICES

_ACTION_TERMS = {
    "base_twist": "base_twist_action",
    "left_arm": "left_arm_action",
    "right_arm": "right_arm_action",
    "lift": "lift_action",
    "head": "head_action",
    "left_gripper": "left_gripper_action",
    "right_gripper": "right_gripper_action",
}


def action_vector_for_key(group: str, index: int, direction: float) -> list[float]:
    """Return a one-joint jog in the stable action-term order used by OpenFlex.

    Arm indices are zero-based. Singleton terms accept only index zero. Direction
    is normalized to [-1, 1] and the corresponding Arena action term applies its
    configured physical scale. ``base_twist`` values are already in SI units and
    are clipped to the canonical contract envelope. Relative actions accumulate
    every step, so callers should use small magnitudes for held-key jogging.
    """

    try:
        start, width = ACTION_SLICES[_ACTION_TERMS[group]]
    except KeyError as error:
        raise ValueError(f"unsupported OpenFlex action group: {group}") from error
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width:
        raise ValueError(f"index for {group} must be in [0, {width - 1}]")
    if (
        isinstance(direction, bool)
        or not isinstance(direction, (int, float))
        or not math.isfinite(direction)
        or (group != "base_twist" and not -1.0 <= direction <= 1.0)
    ):
        raise ValueError("joint direction must be normalized to [-1, 1]; base_twist must be finite SI units")
    action = [0.0] * ACTION_DIMENSION
    action[start + index] = float(direction)
    return action


def toggle_binary_gripper(command: float) -> float:
    """Toggle Isaac Lab's binary-joint convention: positive opens, negative closes."""

    if command not in (-1.0, 1.0):
        raise ValueError("gripper command must be -1 (close) or 1 (open)")
    return -command


def clamp_joint_target(current: float, delta: float, limits: tuple[float, float]) -> float:
    """Apply a direct position jog without leaving the canonical joint envelope."""

    values = (current, delta, *limits)
    if len(limits) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("joint target, jog, and limits must be finite numbers")
    minimum, maximum = limits
    if minimum >= maximum:
        raise ValueError("joint limits must satisfy minimum < maximum")
    return min(max(float(current) + float(delta), float(minimum)), float(maximum))


def validate_joint_targets(
    targets: Mapping[str, float], limits_by_joint: Mapping[str, tuple[float, float]]
) -> dict[str, float]:
    """Validate named direct joint targets against the canonical position limits."""

    validated: dict[str, float] = {}
    for name, target in targets.items():
        if name not in limits_by_joint:
            raise ValueError(f"joint {name!r} has no direct position-control contract")
        if isinstance(target, bool) or not isinstance(target, (int, float)) or not math.isfinite(target):
            raise ValueError(f"joint target for {name!r} must be a finite number")
        limits = limits_by_joint[name]
        if len(limits) != 2:
            raise ValueError(f"joint {name!r} must define minimum and maximum limits")
        minimum, maximum = limits
        if (
            any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in limits)
            or minimum >= maximum
        ):
            raise ValueError(f"joint {name!r} has invalid position limits")
        if not minimum <= target <= maximum:
            raise ValueError(f"joint target for {name!r} is outside [{minimum}, {maximum}]")
        validated[name] = float(target)
    return validated


def physical_jog_delta(
    direction: float,
    action_scale: float,
    *,
    control_dt: float,
    reference_dt: float = 1.0 / 90.0,
) -> float:
    """Scale a normalized per-step jog while preserving its reference physical speed."""

    values = (direction, action_scale, control_dt, reference_dt)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("jog direction, scale, and step durations must be finite numbers")
    if not -1.0 <= direction <= 1.0 or action_scale <= 0.0 or control_dt <= 0.0 or reference_dt <= 0.0:
        raise ValueError("jog direction must be normalized and scales/durations must be positive")
    return float(direction * action_scale * control_dt / reference_dt)
