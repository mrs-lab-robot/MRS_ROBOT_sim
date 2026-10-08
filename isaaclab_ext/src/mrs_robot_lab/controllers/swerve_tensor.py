"""Batched torch implementation of the contract-defined swerve kinematics."""

from __future__ import annotations

import math

import torch

from mrs_robot_lab.controllers.swerve import SwerveConfig


def normalized_to_body_twist(actions: torch.Tensor, config: SwerveConfig) -> torch.Tensor:
    """Map policy actions in [-1, 1] to the physical OpenFlex Twist envelope."""

    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError(f"expected normalized actions with shape (N, 3), got {tuple(actions.shape)}")
    if not bool(torch.isfinite(actions).all().item()):
        raise ValueError("normalized base actions must be finite")
    limits = torch.tensor(config.twist_limits, dtype=actions.dtype, device=actions.device)
    center = limits.mean(dim=1)
    half_range = (limits[:, 1] - limits[:, 0]) / 2.0
    normalized = actions.clamp(-1.0, 1.0)
    return center + normalized * half_range


def compute_swerve_targets_tensor(
    twist: torch.Tensor,
    config: SwerveConfig,
    previous_wheel_linear_speeds: torch.Tensor,
    *,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return steering angles, acceleration-limited wheel speeds (m/s), and applied twist.

    This is batched equivalent of :func:`compute_swerve_targets`; keeping the
    policy path on-device avoids synchronizing every vectorized environment to
    Python during PPO rollouts.
    """

    if twist.ndim != 2 or twist.shape[1] != 3:
        raise ValueError(f"expected body twists with shape (N, 3), got {tuple(twist.shape)}")
    if previous_wheel_linear_speeds.shape != (twist.shape[0], 4):
        raise ValueError(
            "previous_wheel_linear_speeds must have shape "
            f"({twist.shape[0]}, 4), got {tuple(previous_wheel_linear_speeds.shape)}"
        )
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not bool(torch.isfinite(twist).all().item()) or not bool(
        torch.isfinite(previous_wheel_linear_speeds).all().item()
    ):
        raise ValueError("twist and previous wheel speeds must be finite")

    limits = torch.tensor(config.twist_limits, dtype=twist.dtype, device=twist.device)
    applied = torch.maximum(torch.minimum(twist, limits[:, 1]), limits[:, 0])
    positions = torch.tensor(config.wheel_positions, dtype=twist.dtype, device=twist.device)
    vx = applied[:, 0:1] - applied[:, 2:3] * positions[:, 1].unsqueeze(0)
    vy = applied[:, 1:2] + applied[:, 2:3] * positions[:, 0].unsqueeze(0)
    angles = torch.atan2(vy, vx)
    linear_speeds = torch.hypot(vx, vy)
    peak = linear_speeds.amax(dim=1, keepdim=True)
    linear_speeds = linear_speeds * torch.clamp(config.max_wheel_speed / peak.clamp_min(1e-9), max=1.0)

    steering_limits = torch.tensor(config.steering_limits, dtype=twist.dtype, device=twist.device)
    direct_valid = (angles >= steering_limits[:, 0]) & (angles <= steering_limits[:, 1])
    minus_pi = angles - torch.pi
    plus_pi = angles + torch.pi
    minus_valid = (minus_pi >= steering_limits[:, 0]) & (minus_pi <= steering_limits[:, 1])
    plus_valid = (plus_pi >= steering_limits[:, 0]) & (plus_pi <= steering_limits[:, 1])
    if not bool((direct_valid | minus_valid | plus_valid).all().item()):
        raise ValueError("swerve vector has no steering solution within the configured joint limits")
    reverse = ~direct_valid
    angles = torch.where(direct_valid, angles, torch.where(minus_valid, minus_pi, plus_pi))
    linear_speeds = torch.where(reverse, -linear_speeds, linear_speeds)

    idle = applied.abs().amax(dim=1, keepdim=True) <= 1e-9
    angles = torch.where(idle, torch.zeros_like(angles), angles)
    linear_speeds = torch.where(idle, torch.zeros_like(linear_speeds), linear_speeds)

    max_delta = config.wheel_accel_limit * dt
    linear_speeds = torch.maximum(
        torch.minimum(linear_speeds, previous_wheel_linear_speeds + max_delta),
        previous_wheel_linear_speeds - max_delta,
    )
    return angles, linear_speeds, applied
