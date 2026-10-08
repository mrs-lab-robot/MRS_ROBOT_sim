"""Batched torch task progress for vectorized Isaac Lab environments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from mrs_robot_lab.environments.learning.task_logic import TransportPhase


class BatchedBoxTransportEvaluator:
    """GPU-friendly ordered bilateral transport success evaluator."""

    def __init__(
        self,
        *,
        initial_position: Sequence[float],
        target_position: Sequence[float],
        num_envs: int,
        position_tolerance_m: float,
        min_lift_height_m: float,
        min_carry_distance_m: float,
        stable_seconds: float,
        max_linear_speed_mps: float,
        device: str | torch.device,
    ) -> None:
        if isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.device = torch.device(device)
        self.initial_position = _point(initial_position, "initial_position", self.device).expand(num_envs, 3).clone()
        self.target_position = _point(target_position, "target_position", self.device).expand(num_envs, 3).clone()
        self.position_tolerance_m = _positive(position_tolerance_m, "position_tolerance_m")
        self.min_lift_height_m = _positive(min_lift_height_m, "min_lift_height_m")
        self.min_carry_distance_m = _positive(min_carry_distance_m, "min_carry_distance_m")
        self.stable_seconds = _positive(stable_seconds, "stable_seconds")
        self.max_linear_speed_mps = _positive(max_linear_speed_mps, "max_linear_speed_mps")
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.has_grasped = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.has_lifted = torch.zeros_like(self.has_grasped)
        self.has_carried = torch.zeros_like(self.has_grasped)
        self.has_placed = torch.zeros_like(self.has_grasped)
        self.stable_elapsed = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

    def update(
        self,
        box_position: torch.Tensor,
        box_linear_velocity: torch.Tensor,
        left_gripper_contact: torch.Tensor,
        right_gripper_contact: torch.Tensor,
        *,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step = _positive(dt, "dt")
        position = box_position.to(device=self.device, dtype=torch.float32)
        velocity = box_linear_velocity.to(device=self.device, dtype=torch.float32)
        left_contact = left_gripper_contact.to(device=self.device, dtype=torch.bool)
        right_contact = right_gripper_contact.to(device=self.device, dtype=torch.bool)
        count = self.phase.shape[0]
        if position.shape != (count, 3) or velocity.shape != (count, 3):
            raise ValueError(f"box position and velocity must have shape ({count}, 3)")
        if left_contact.shape != (count,) or right_contact.shape != (count,):
            raise ValueError(f"gripper contact tensors must have shape ({count},)")
        if not bool(torch.isfinite(position).all().item()) or not bool(torch.isfinite(velocity).all().item()):
            raise ValueError("box position and velocity must be finite")

        self.has_grasped |= left_contact & right_contact
        high_enough = position[:, 2] >= self.initial_position[:, 2] + self.min_lift_height_m
        self.has_lifted |= self.has_grasped & high_enough
        carry_distance = torch.linalg.vector_norm(
            position[:, :2] - self.initial_position[:, :2], dim=-1
        )
        self.has_carried |= self.has_lifted & (carry_distance >= self.min_carry_distance_m)
        at_target = torch.linalg.vector_norm(position - self.target_position, dim=-1) <= self.position_tolerance_m
        self.has_placed |= self.has_carried & at_target

        released = self.has_placed & ~left_contact & ~right_contact
        stable_speed = torch.linalg.vector_norm(velocity, dim=-1) <= self.max_linear_speed_mps
        self.stable_elapsed = torch.where(
            released & at_target & stable_speed,
            self.stable_elapsed + step,
            torch.zeros_like(self.stable_elapsed),
        )
        success = self.stable_elapsed + 1e-6 >= self.stable_seconds
        placed = self.has_placed
        self.phase = torch.where(self.has_grasped, 1, 0)
        self.phase = torch.where(self.has_lifted, 2, self.phase)
        self.phase = torch.where(self.has_carried, 3, self.phase)
        self.phase = torch.where(placed, 4, self.phase)
        self.phase = torch.where(released, 5, self.phase)
        self.phase = torch.where(success, 6, self.phase)
        return self.phase.clone(), success

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        initial_position: torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.phase.shape[0], device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if initial_position is not None:
            updated_initial_position = initial_position.to(device=self.device, dtype=torch.float32)
            if updated_initial_position.shape != (env_ids.numel(), 3):
                raise ValueError(f"randomized initial_position must have shape ({env_ids.numel()}, 3)")
            if not bool(torch.isfinite(updated_initial_position).all().item()):
                raise ValueError("randomized initial_position must be finite")
            self.initial_position[env_ids] = updated_initial_position
        self.phase[env_ids] = 0
        self.has_grasped[env_ids] = False
        self.has_lifted[env_ids] = False
        self.has_carried[env_ids] = False
        self.has_placed[env_ids] = False
        self.stable_elapsed[env_ids] = 0.0


def _point(values: Sequence[float], name: str, device: torch.device) -> torch.Tensor:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three finite numbers")
    point = torch.tensor(values, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(point).all().item()):
        raise ValueError(f"{name} must be finite")
    return point


def contact_mask_from_force_matrix(force_matrix: torch.Tensor, *, force_threshold: float) -> torch.Tensor:
    """Reduce filtered finger-to-box contact forces to one boolean per environment."""

    if not isinstance(force_matrix, torch.Tensor):
        if hasattr(force_matrix, "torch"):
            force_matrix = force_matrix.torch
        else:
            raise TypeError(
                f"force_matrix must be a torch.Tensor or have a .torch property, got {type(force_matrix).__name__}"
            )

    if not isinstance(force_matrix, torch.Tensor):
        raise TypeError(f"force_matrix.torch must be a torch.Tensor, got {type(force_matrix).__name__}")

    threshold = _positive(force_threshold, "force_threshold")
    if force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
        raise ValueError("filtered contact force matrix must have shape (N, sensors, partners, 3)")
    if not bool(torch.isfinite(force_matrix).all().item()):
        raise ValueError("filtered contact forces must be finite")
    return torch.linalg.vector_norm(force_matrix, dim=-1).amax(dim=-1).amax(dim=-1) > threshold


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)
