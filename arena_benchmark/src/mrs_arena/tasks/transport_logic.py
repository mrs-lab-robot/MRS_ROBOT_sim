"""Simulator-independent batched progress logic for the bilateral box task."""

from __future__ import annotations

import math

import torch


def create_transport_progress_state(num_envs: int, *, device: str | torch.device) -> dict[str, torch.Tensor]:
    if isinstance(num_envs, bool) or int(num_envs) <= 0:
        raise ValueError("num_envs must be a positive integer")
    return {
        "bilateral_grasp_seen": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "lift_seen": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "carry_seen": torch.zeros(num_envs, dtype=torch.bool, device=device),
        "release_stable_seconds": torch.zeros(num_envs, dtype=torch.float32, device=device),
    }


def update_transport_progress(
    state: dict[str, torch.Tensor],
    *,
    box_position: torch.Tensor,
    box_linear_speed: torch.Tensor,
    initial_position: torch.Tensor,
    target_position: tuple[float, float, float],
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    dt: float,
    min_lift_height_m: float,
    min_carry_distance_m: float,
    position_tolerance_m: float,
    max_linear_speed_mps: float,
    stable_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update phase history and return ``(success, phase)`` for each environment.

    ``phase`` is 0 before a bilateral grasp, 1 after grasp, 2 after a valid lift,
    and 3 after the carried-distance criterion. Release success then requires
    the box to be at its target and settled below the configured speed.
    """
    _validate_inputs(
        state,
        box_position,
        box_linear_speed,
        initial_position,
        left_contact,
        right_contact,
        dt,
        min_lift_height_m,
        min_carry_distance_m,
        position_tolerance_m,
        max_linear_speed_mps,
        stable_seconds,
    )
    target = torch.as_tensor(target_position, dtype=box_position.dtype, device=box_position.device)
    if target.shape != (3,) or not bool(torch.isfinite(target).all().item()):
        raise ValueError("target_position must contain three finite values")

    both_contacts = left_contact.bool() & right_contact.bool()
    any_contact = left_contact.bool() | right_contact.bool()
    state["bilateral_grasp_seen"] |= both_contacts

    height_gain = box_position[:, 2] - initial_position[:, 2]
    state["lift_seen"] |= both_contacts & (height_gain >= min_lift_height_m)

    carry_distance = torch.linalg.vector_norm(
        box_position[:, :2] - initial_position[:, :2], dim=-1
    )
    state["carry_seen"] |= (
        both_contacts
        & state["lift_seen"]
        & (carry_distance >= min_carry_distance_m)
    )

    target_error = torch.linalg.vector_norm(box_position - target.unsqueeze(0), dim=-1)
    at_target = target_error <= position_tolerance_m
    settled = box_linear_speed <= max_linear_speed_mps
    released = ~any_contact
    final_state_valid = (
        state["bilateral_grasp_seen"]
        & state["lift_seen"]
        & state["carry_seen"]
        & at_target
        & settled
        & released
    )
    state["release_stable_seconds"] = torch.where(
        final_state_valid,
        state["release_stable_seconds"] + float(dt),
        torch.zeros_like(state["release_stable_seconds"]),
    )
    success = state["release_stable_seconds"] + 1e-6 >= stable_seconds
    phase = torch.where(
        state["carry_seen"],
        torch.full_like(state["release_stable_seconds"], 3, dtype=torch.long),
        torch.where(
            state["lift_seen"],
            torch.full_like(state["release_stable_seconds"], 2, dtype=torch.long),
            torch.where(
                state["bilateral_grasp_seen"],
                torch.ones_like(state["release_stable_seconds"], dtype=torch.long),
                torch.zeros_like(state["release_stable_seconds"], dtype=torch.long),
            ),
        ),
    )
    return success, phase


def _validate_inputs(
    state: dict[str, torch.Tensor],
    box_position: torch.Tensor,
    box_linear_speed: torch.Tensor,
    initial_position: torch.Tensor,
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    dt: float,
    min_lift_height_m: float,
    min_carry_distance_m: float,
    position_tolerance_m: float,
    max_linear_speed_mps: float,
    stable_seconds: float,
) -> None:
    required_state = {
        "bilateral_grasp_seen",
        "lift_seen",
        "carry_seen",
        "release_stable_seconds",
    }
    if not isinstance(state, dict) or set(state) != required_state:
        raise ValueError("transport state must come from create_transport_progress_state")
    if box_position.ndim != 2 or box_position.shape[1] != 3:
        raise ValueError("box_position must have shape (num_envs, 3)")
    count = box_position.shape[0]
    if initial_position.shape != box_position.shape:
        raise ValueError("initial_position must have the same shape as box_position")
    if box_linear_speed.shape != (count,) or left_contact.shape != (count,) or right_contact.shape != (count,):
        raise ValueError("speed and contact inputs must have shape (num_envs,)")
    if any(value.shape != (count,) for value in state.values()):
        raise ValueError("transport state environment count does not match input tensors")
    numeric = (
        dt,
        min_lift_height_m,
        min_carry_distance_m,
        position_tolerance_m,
        max_linear_speed_mps,
        stable_seconds,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("transport thresholds must be finite")
    if dt <= 0 or min_lift_height_m <= 0 or min_carry_distance_m <= 0 or position_tolerance_m <= 0:
        raise ValueError("time step, lift, carry, and position thresholds must be positive")
    if max_linear_speed_mps < 0 or stable_seconds <= 0:
        raise ValueError("speed threshold must be non-negative and stable_seconds positive")
    if not bool(torch.isfinite(box_position).all().item()) or not bool(torch.isfinite(initial_position).all().item()):
        raise ValueError("box positions must be finite")
    if not bool(torch.isfinite(box_linear_speed).all().item()) or bool((box_linear_speed < 0).any().item()):
        raise ValueError("box_linear_speed must be finite and non-negative")
