"""Simulator-independent success logic for the shared point-navigation task."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class NavigationParameters:
    target_xy: tuple[float, float]
    target_yaw_rad: float
    position_tolerance_m: float
    heading_tolerance_rad: float
    hold_seconds: float
    episode_length_s: float


def navigation_parameters_from_task(task) -> NavigationParameters:
    """Read and validate the Arena-relevant fields from the shared TaskSpec."""
    if getattr(task, "task_id", None) != "navigation_to_goal":
        raise ValueError("Arena navigation adapter requires task_id 'navigation_to_goal'")
    backends = tuple(getattr(task, "supported_backends", ()))
    if backends and "arena" not in backends:
        raise ValueError("navigation_to_goal does not declare the Arena backend")
    criteria = [
        criterion
        for criterion in getattr(task, "success_criteria", ())
        if criterion.criterion_type in {"base_pose_at_target", "base_at_target", "reach_target"}
    ]
    if len(criteria) != 1:
        raise ValueError("navigation task must define exactly one base-pose success criterion")
    criterion = criteria[0]
    if not criterion.required:
        raise ValueError("navigation base-pose success criterion must be required")

    params = criterion.params
    target = params.get("target_position")
    if not isinstance(target, (tuple, list)) or len(target) < 2:
        raise ValueError("navigation criterion target_position must contain at least x and y")
    target_xy = (_finite_number(target[0], "target_position.x"), _finite_number(target[1], "target_position.y"))
    target_yaw = _finite_number(params.get("target_yaw_rad", 0.0), "target_yaw_rad")
    heading_degrees = _finite_number(params.get("heading_tolerance_deg", 10.0), "heading_tolerance_deg")
    hold_seconds = _finite_number(params.get("hold_seconds", 1.0), "hold_seconds")
    if not 0.0 < heading_degrees <= 180.0:
        raise ValueError("heading_tolerance_deg must be in (0, 180]")
    if hold_seconds <= 0.0:
        raise ValueError("hold_seconds must be positive")
    tolerance = _finite_number(criterion.tolerance, "position tolerance")
    episode_length = _finite_number(task.episode_length_s, "episode_length_s")
    if tolerance <= 0.0 or episode_length <= 0.0:
        raise ValueError("navigation position tolerance and episode length must be positive")

    return NavigationParameters(
        target_xy=target_xy,
        target_yaw_rad=target_yaw,
        position_tolerance_m=tolerance,
        heading_tolerance_rad=math.radians(heading_degrees),
        hold_seconds=hold_seconds,
        episode_length_s=episode_length,
    )


def navigation_goal_mask(
    position_xy: torch.Tensor,
    yaw_rad: torch.Tensor,
    parameters: NavigationParameters,
) -> torch.Tensor:
    """Return one success-region flag per environment, with wrapped yaw error."""
    if position_xy.ndim != 2 or position_xy.shape[1] != 2:
        raise ValueError("position_xy must have shape (num_envs, 2)")
    if yaw_rad.shape != (position_xy.shape[0],):
        raise ValueError("yaw_rad must have shape (num_envs,)")
    if not position_xy.is_floating_point() or not yaw_rad.is_floating_point():
        raise ValueError("navigation positions and yaw must use floating-point tensors")
    if not bool(torch.isfinite(position_xy).all().item()) or not bool(torch.isfinite(yaw_rad).all().item()):
        raise ValueError("navigation positions and yaw must be finite")
    target = torch.tensor(parameters.target_xy, dtype=position_xy.dtype, device=position_xy.device)
    distance = torch.linalg.vector_norm(position_xy - target.unsqueeze(0), dim=-1)
    heading_error = torch.atan2(
        torch.sin(yaw_rad - parameters.target_yaw_rad),
        torch.cos(yaw_rad - parameters.target_yaw_rad),
    ).abs()
    return (distance <= parameters.position_tolerance_m) & (
        heading_error <= parameters.heading_tolerance_rad
    )


def navigation_observation_tensor(
    position_xy: torch.Tensor,
    yaw_rad: torch.Tensor,
    velocity_world_xy: torch.Tensor,
    angular_velocity_z: torch.Tensor,
    *,
    target_xy: tuple[float, float],
    target_yaw_rad: float,
) -> torch.Tensor:
    """Build the checkpoint's ordered goal-relative 8D observation batch."""
    count = position_xy.shape[0] if position_xy.ndim == 2 else -1
    if position_xy.shape != (count, 2) or velocity_world_xy.shape != (count, 2):
        raise ValueError("position_xy and velocity_world_xy must have shape (num_envs, 2)")
    if yaw_rad.shape != (count,) or angular_velocity_z.shape != (count,):
        raise ValueError("yaw_rad and angular_velocity_z must have shape (num_envs,)")
    tensors = (position_xy, yaw_rad, velocity_world_xy, angular_velocity_z)
    if any(not value.is_floating_point() for value in tensors):
        raise ValueError("navigation observation inputs must be floating-point tensors")
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        raise ValueError("navigation observation inputs must be finite")
    if len(target_xy) != 2 or not all(math.isfinite(float(value)) for value in target_xy):
        raise ValueError("target_xy must contain two finite values")
    if not math.isfinite(float(target_yaw_rad)):
        raise ValueError("target_yaw_rad must be finite")

    target = torch.tensor(target_xy, dtype=position_xy.dtype, device=position_xy.device)
    delta_world = target.unsqueeze(0) - position_xy
    cos_yaw = torch.cos(yaw_rad)
    sin_yaw = torch.sin(yaw_rad)
    goal_body_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
    goal_body_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
    heading_error = torch.atan2(
        torch.sin(float(target_yaw_rad) - yaw_rad),
        torch.cos(float(target_yaw_rad) - yaw_rad),
    )
    velocity_body_x = cos_yaw * velocity_world_xy[:, 0] + sin_yaw * velocity_world_xy[:, 1]
    velocity_body_y = -sin_yaw * velocity_world_xy[:, 0] + cos_yaw * velocity_world_xy[:, 1]
    return torch.stack(
        (
            goal_body_x,
            goal_body_y,
            torch.sin(heading_error),
            torch.cos(heading_error),
            velocity_body_x,
            velocity_body_y,
            angular_velocity_z,
            torch.linalg.vector_norm(delta_world, dim=-1),
        ),
        dim=-1,
    )


def quaternion_xyzw_to_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert Arena's ``(x, y, z, w)`` root quaternion tensor to planar yaw."""
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape (num_envs, 4) in xyzw order")
    if not quaternion.is_floating_point() or not bool(torch.isfinite(quaternion).all().item()):
        raise ValueError("quaternion must be a finite floating-point tensor")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((norm <= 1e-12).any().item()):
        raise ValueError("quaternion norm must be positive")
    x, y, z, w = (quaternion / norm).unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def advance_success_hold(
    held_seconds: torch.Tensor,
    inside_goal: torch.Tensor,
    *,
    dt: float,
    required_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate consecutive time in the success region and reset outside it."""
    if held_seconds.ndim != 1 or inside_goal.shape != held_seconds.shape:
        raise ValueError("held_seconds and inside_goal must be matching one-dimensional tensors")
    if not held_seconds.is_floating_point() or inside_goal.dtype != torch.bool:
        raise ValueError("held_seconds must be floating-point and inside_goal must be boolean")
    if not bool(torch.isfinite(held_seconds).all().item()) or bool((held_seconds < 0.0).any().item()):
        raise ValueError("held_seconds must be finite and non-negative")
    if not math.isfinite(float(dt)) or not math.isfinite(float(required_seconds)):
        raise ValueError("hold timing values must be finite")
    if dt <= 0.0 or required_seconds <= 0.0:
        raise ValueError("dt and required_seconds must be positive")
    if held_seconds.device != inside_goal.device:
        raise ValueError("held_seconds and inside_goal must be on the same device")

    updated = torch.where(inside_goal, held_seconds + float(dt), torch.zeros_like(held_seconds))
    success = updated + 1e-6 >= float(required_seconds)
    return updated, success


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)
