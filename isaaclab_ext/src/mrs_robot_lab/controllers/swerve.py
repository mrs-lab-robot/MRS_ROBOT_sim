"""Swerve kinematics loaded from the canonical MRS_ROBOT_sim configuration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


_MODULE_PREFIXES = ("fl", "fr", "bl", "br")


@dataclass(frozen=True)
class SwerveConfig:
    wheel_radius: float
    max_wheel_speed: float
    wheel_positions: tuple[tuple[float, float], ...]
    twist_limits: tuple[tuple[float, float], ...]
    steering_limits: tuple[tuple[float, float], ...] = ((-math.pi / 2, math.pi / 2),) * 4
    steering_joint_names: tuple[str, ...] = (
        "fl_steering_joint",
        "fr_steering_joint",
        "bl_steering_joint",
        "br_steering_joint",
    )
    wheel_joint_names: tuple[str, ...] = ("fl_wheel_joint", "fr_wheel_joint", "bl_wheel_joint", "br_wheel_joint")
    wheel_accel_limit: float = 0.2


@dataclass(frozen=True)
class SwerveTargets:
    steering_angles: tuple[float, float, float, float]
    wheel_angular_velocities: tuple[float, float, float, float]
    applied_twist: tuple[float, float, float]


def load_swerve_config(controller_path: str | Path, contract_path: str | Path) -> SwerveConfig:
    """Read wheel geometry/limits from the controller config and robot contract."""

    controller = _read_yaml(controller_path)
    contract = _read_yaml(contract_path)
    try:
        params = controller["swerve_drive_controller"]["ros__parameters"]
        action = next(item for item in contract["actions"] if item["name"] == "base_twist")
        wheel_radius = _positive_number(params["wheel_radius"], "wheel_radius")
        max_wheel_speed = _positive_number(params["max_wheel_speed"], "max_wheel_speed")
        wheel_accel_limit = _positive_number(params["wheel_accel_limit"], "wheel_accel_limit")
        steering_names = tuple(params["steering_joint_names"])
        wheel_names = tuple(params["wheel_joint_names"])
        wheel_positions = tuple(
            (_number(params[f"{prefix}_pos_x"], f"{prefix}_pos_x"),
             _number(params[f"{prefix}_pos_y"], f"{prefix}_pos_y"))
            for prefix in _MODULE_PREFIXES
        )
        steering_limits = tuple(
            (_number(params[f"{prefix}_steering_min"], f"{prefix}_steering_min"),
             _number(params[f"{prefix}_steering_max"], f"{prefix}_steering_max"))
            for prefix in _MODULE_PREFIXES
        )
        twist_limits = tuple(
            (_number(limit["minimum"], "minimum"), _number(limit["maximum"], "maximum"))
            for limit in action["limits"]
        )
    except (KeyError, StopIteration, TypeError) as error:
        raise ValueError(f"invalid OpenFlex swerve or base_twist configuration: {error}") from error
    if len(steering_names) != 4 or len(wheel_names) != 4 or len(twist_limits) != 3:
        raise ValueError("OpenFlex swerve requires four modules and a three-field base_twist limit")
    if any(low >= high for low, high in (*twist_limits, *steering_limits)):
        raise ValueError("base_twist and steering limits must have minimum < maximum")
    expected_steering = tuple(f"{prefix}_steering_joint" for prefix in _MODULE_PREFIXES)
    expected_wheels = tuple(f"{prefix}_wheel_joint" for prefix in _MODULE_PREFIXES)
    if steering_names != expected_steering or wheel_names != expected_wheels:
        raise ValueError("swerve joint names must use matching fl/fr/bl/br controller order")
    return SwerveConfig(
        wheel_radius=wheel_radius,
        max_wheel_speed=max_wheel_speed,
        wheel_positions=wheel_positions,
        twist_limits=twist_limits,
        steering_limits=steering_limits,
        steering_joint_names=steering_names,
        wheel_joint_names=wheel_names,
        wheel_accel_limit=wheel_accel_limit,
    )


def compute_swerve_targets(twist: tuple[float, float, float], cfg: SwerveConfig) -> SwerveTargets:
    """Convert body-frame ``(vx, vy, yaw_rate)`` into four steering and wheel targets.

    Each input component is clipped to the MRS contract envelope. Combined wheel
    speeds are then uniformly scaled to the controller's maximum linear speed.
    Steering targets are optimized into the USD joints' [-pi/2, pi/2] range by
    reversing the corresponding wheel speed when needed.
    """

    if len(twist) != 3 or len(cfg.wheel_positions) != 4:
        raise ValueError("swerve input must be a 3-field twist and four wheel positions")
    values = tuple(_number(value, "twist component") for value in twist)
    applied = tuple(min(max(value, low), high) for value, (low, high) in zip(values, cfg.twist_limits, strict=True))
    vx, vy, yaw_rate = applied
    vectors = tuple((vx - yaw_rate * y, vy + yaw_rate * x) for x, y in cfg.wheel_positions)
    speeds = [math.hypot(module_vx, module_vy) for module_vx, module_vy in vectors]
    peak_speed = max(speeds, default=0.0)
    if peak_speed > cfg.max_wheel_speed:
        scale = cfg.max_wheel_speed / peak_speed
        speeds = [speed * scale for speed in speeds]

    steering: list[float] = []
    wheel_speeds: list[float] = []
    for (module_vx, module_vy), speed, (minimum, maximum) in zip(
        vectors, speeds, cfg.steering_limits, strict=True
    ):
        if speed <= 1e-9:
            steering.append(0.0)
            wheel_speeds.append(0.0)
            continue
        angle = math.atan2(module_vy, module_vx)
        alternatives = ((angle, speed), (angle - math.pi, -speed), (angle + math.pi, -speed))
        valid = tuple(candidate for candidate in alternatives if minimum <= candidate[0] <= maximum)
        if not valid:
            raise ValueError(f"module velocity has no steering solution inside [{minimum}, {maximum}]")
        angle, speed = min(valid, key=lambda candidate: abs(candidate[0]))
        steering.append(angle)
        wheel_speeds.append(speed / cfg.wheel_radius)
    return SwerveTargets(tuple(steering), tuple(wheel_speeds), applied)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read swerve configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"swerve configuration root must be a mapping: {path}")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_number(value: Any, name: str) -> float:
    number = _number(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number
