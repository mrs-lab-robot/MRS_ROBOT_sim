"""Translate existing ROS VR setpoints into bounded Isaac Lab task actions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def normalize_base_twist(
    values: Sequence[float], limits: Sequence[tuple[float, float]]
) -> tuple[float, float, float]:
    """Map a physical ``(vx, vy, wz)`` command to the task's ``[-1, 1]`` action.

    Values are first clipped to the robot contract. The affine transform also
    handles non-symmetric limits, so a physical zero is not assumed to map to
    normalized zero when the contract says otherwise.
    """

    command = _finite_values(values, 3, "base twist must contain three finite values")
    checked_limits = _checked_limits(limits, 3)
    normalized = []
    for value, (lower, upper) in zip(command, checked_limits, strict=True):
        clipped = min(max(value, lower), upper)
        center = (lower + upper) / 2.0
        half_range = (upper - lower) / 2.0
        normalized.append(min(max((clipped - center) / half_range, -1.0), 1.0))
    return tuple(normalized)


def normalize_bilateral_arm_targets(
    left_values: Sequence[float] | None,
    right_values: Sequence[float] | None,
    joint_limits: Sequence[tuple[float, float]],
    *,
    hold_action: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Map ROS left/right ``7 arm + 1 gripper`` targets to Lab task order.

    Isaac Lab's dual-arm task order is left arm (7), right arm (7), left
    gripper, right gripper. Arm joints use a continuous bounded mapping. The
    existing Lab action contract defines negative gripper commands as the
    lower/open endpoint and non-negative commands as the upper/closed
    endpoint, matching the ROS VR node's ``0.0`` to ``0.044`` trigger output.
    """

    limits = _checked_limits(joint_limits, 16)
    held = None
    if hold_action is not None:
        held = _finite_values(hold_action, 16, "hold_action must contain 16 finite values")
        if any(value < -1.0 or value > 1.0 for value in held):
            raise ValueError("hold_action values must be in [-1, 1]")
    if left_values is None and right_values is None and held is None:
        raise ValueError("hold_action is required when both arm commands are missing")

    left = (
        _normalize_one_arm(left_values, limits[:7] + (limits[14],))
        if left_values is not None
        else None
    )
    right = (
        _normalize_one_arm(right_values, limits[7:14] + (limits[15],))
        if right_values is not None
        else None
    )
    if left is None:
        assert held is not None
        left = (*held[:7], held[14])
    if right is None:
        assert held is not None
        right = (*held[7:14], held[15])
    return (*left[:7], *right[:7], left[7], right[7])


def _normalize_one_arm(
    values: Sequence[float], limits: Sequence[tuple[float, float]]
) -> tuple[float, ...]:
    command = _finite_values(values, 8, "each arm command must contain eight values")
    normalized = []
    for index, (value, (lower, upper)) in enumerate(zip(command, limits, strict=True)):
        clipped = min(max(value, lower), upper)
        if index == 7:
            normalized.append(1.0 if clipped > (lower + upper) / 2.0 else -1.0)
        else:
            center = (lower + upper) / 2.0
            half_range = (upper - lower) / 2.0
            normalized.append(min(max((clipped - center) / half_range, -1.0), 1.0))
    return tuple(normalized)


def _finite_values(values: Sequence[float], length: int, message: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(message)
    try:
        if len(values) != length:
            raise ValueError(message)
        converted = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError("values must be finite numbers")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("values must be finite numbers") from error
            if not math.isfinite(number):
                raise ValueError("values must be finite numbers")
            converted.append(number)
    except TypeError as error:
        raise ValueError(message) from error
    return tuple(converted)


def _checked_limits(
    limits: Sequence[tuple[float, float]], length: int
) -> tuple[tuple[float, float], ...]:
    if isinstance(limits, (str, bytes)):
        raise ValueError("joint/twist limits must be finite minimum/maximum pairs")
    try:
        if len(limits) != length:
            raise ValueError(f"expected {length} limit pairs")
        result = []
        for pair in limits:
            if isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ValueError("joint/twist limits must be finite minimum/maximum pairs")
            lower, upper = _finite_values(pair, 2, "joint/twist limits must be finite")
            if lower >= upper:
                raise ValueError("each joint/twist limit must have minimum < maximum")
            result.append((lower, upper))
    except TypeError as error:
        raise ValueError("joint/twist limits must be finite minimum/maximum pairs") from error
    return tuple(result)


__all__ = ["normalize_base_twist", "normalize_bilateral_arm_targets"]
