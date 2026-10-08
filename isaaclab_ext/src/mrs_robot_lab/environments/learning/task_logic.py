"""Simulator-independent success state machines for the baseline tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence


class TransportPhase(str, Enum):
    APPROACH = "approach"
    GRASPED = "grasped"
    LIFTED = "lifted"
    CARRIED = "carried"
    PLACED = "placed"
    RELEASED = "released"
    SUCCESS = "success"


@dataclass
class NavigationGoalEvaluator:
    """Require a base pose to remain within position and heading bounds."""

    position_tolerance_m: float
    heading_tolerance_rad: float
    hold_seconds: float
    held_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.position_tolerance_m = _positive(self.position_tolerance_m, "position_tolerance_m")
        self.heading_tolerance_rad = _positive(self.heading_tolerance_rad, "heading_tolerance_rad")
        self.hold_seconds = _positive(self.hold_seconds, "hold_seconds")

    def update(
        self,
        base_position_xy: Sequence[float],
        base_yaw_rad: float,
        target_position_xy: Sequence[float],
        target_yaw_rad: float,
        dt: float,
    ) -> bool:
        """Advance the dwell timer and return whether the goal has been held."""

        base = _vector(base_position_xy, 2, "base_position_xy")
        target = _vector(target_position_xy, 2, "target_position_xy")
        base_yaw = _finite(base_yaw_rad, "base_yaw_rad")
        target_yaw = _finite(target_yaw_rad, "target_yaw_rad")
        step = _positive(dt, "dt")
        position_error = math.hypot(base[0] - target[0], base[1] - target[1])
        heading_error = abs(math.atan2(math.sin(base_yaw - target_yaw), math.cos(base_yaw - target_yaw)))
        if position_error <= self.position_tolerance_m and heading_error <= self.heading_tolerance_rad:
            self.held_seconds += step
        else:
            self.held_seconds = 0.0
        return self.held_seconds + 1e-12 >= self.hold_seconds


@dataclass
class BoxTransportEvaluator:
    """Verify bilateral grasp, lift, carry, place, release and settling in order.

    Gripper-contact inputs must come from simulator contact sensing; commanded
    finger positions alone are not accepted as evidence that the box is grasped.
    """

    initial_position: Sequence[float]
    target_position: Sequence[float]
    position_tolerance_m: float
    min_lift_height_m: float
    min_carry_distance_m: float
    stable_seconds: float
    max_linear_speed_mps: float
    phase: TransportPhase = TransportPhase.APPROACH
    stable_elapsed_seconds: float = 0.0
    _has_bilateral_grasp: bool = False
    _has_lifted: bool = False
    _has_carried: bool = False
    _has_placed: bool = False

    def __post_init__(self) -> None:
        self.initial_position = _vector(self.initial_position, 3, "initial_position")
        self.target_position = _vector(self.target_position, 3, "target_position")
        self.position_tolerance_m = _positive(self.position_tolerance_m, "position_tolerance_m")
        self.min_lift_height_m = _positive(self.min_lift_height_m, "min_lift_height_m")
        self.min_carry_distance_m = _positive(self.min_carry_distance_m, "min_carry_distance_m")
        self.stable_seconds = _positive(self.stable_seconds, "stable_seconds")
        self.max_linear_speed_mps = _positive(self.max_linear_speed_mps, "max_linear_speed_mps")

    def update(
        self,
        *,
        box_position: Sequence[float],
        box_linear_velocity: Sequence[float],
        left_gripper_contact: bool,
        right_gripper_contact: bool,
        dt: float,
    ) -> TransportPhase:
        if not isinstance(left_gripper_contact, bool) or not isinstance(right_gripper_contact, bool):
            raise ValueError("gripper contact evidence must be boolean")
        position = _vector(box_position, 3, "box_position")
        velocity = _vector(box_linear_velocity, 3, "box_linear_velocity")
        step = _positive(dt, "dt")
        if self.phase is TransportPhase.SUCCESS:
            return self.phase

        if left_gripper_contact and right_gripper_contact:
            self._has_bilateral_grasp = True
        if self._has_bilateral_grasp and position[2] >= self.initial_position[2] + self.min_lift_height_m:
            self._has_lifted = True
        carry_distance = math.hypot(position[0] - self.initial_position[0], position[1] - self.initial_position[1])
        if self._has_lifted and carry_distance >= self.min_carry_distance_m:
            self._has_carried = True
        if self._has_carried and _distance(position, self.target_position) <= self.position_tolerance_m:
            self._has_placed = True

        if self._has_placed:
            if left_gripper_contact or right_gripper_contact:
                self.phase = TransportPhase.PLACED
            else:
                self.phase = TransportPhase.RELEASED
                speed = math.sqrt(sum(component * component for component in velocity))
                at_target = _distance(position, self.target_position) <= self.position_tolerance_m
                if at_target and speed <= self.max_linear_speed_mps:
                    self.stable_elapsed_seconds += step
                else:
                    self.stable_elapsed_seconds = 0.0
                if self.stable_elapsed_seconds + 1e-12 >= self.stable_seconds:
                    self.phase = TransportPhase.SUCCESS
            return self.phase

        if self._has_carried:
            self.phase = TransportPhase.CARRIED
        elif self._has_lifted:
            self.phase = TransportPhase.LIFTED
        elif self._has_bilateral_grasp:
            self.phase = TransportPhase.GRASPED
        return self.phase


def _vector(values: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} finite numbers")
    return tuple(_finite(value, name) for value in values)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
