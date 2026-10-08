"""Convert normalized policy outputs to named physical joint targets."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def normalized_to_joint_targets(
    actions: torch.Tensor,
    joint_limits: Sequence[tuple[float, float]],
    *,
    gripper_indices: Sequence[int],
) -> torch.Tensor:
    """Scale arm joints to their limits and map gripper signs to open/close.

    The action columns must already follow the joint-name order paired with the
    supplied limits. Arm columns use a continuous normalized position mapping;
    gripper columns use an explicit sign-to-open/close command so zero means
    open rather than an unintended half-closed target.
    """

    if actions.ndim != 2 or actions.shape[1] != len(joint_limits):
        raise ValueError(
            f"action dimension must equal the number of joint limits ({len(joint_limits)}), "
            f"got {tuple(actions.shape)}"
        )
    if not bool(torch.isfinite(actions).all().item()):
        raise ValueError("joint policy actions must be finite")
    for index in gripper_indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(joint_limits):
            raise ValueError(f"invalid gripper action index: {index}")
    if len(set(gripper_indices)) != len(gripper_indices):
        raise ValueError("gripper action indices must be unique")

    limits = torch.tensor(joint_limits, dtype=actions.dtype, device=actions.device)
    if limits.shape != (actions.shape[1], 2) or not bool(torch.isfinite(limits).all().item()):
        raise ValueError("joint limits must be finite (minimum, maximum) pairs")
    if not bool((limits[:, 0] < limits[:, 1]).all().item()):
        raise ValueError("each joint limit must have minimum < maximum")

    normalized = actions.clamp(-1.0, 1.0)
    targets = limits.mean(dim=1) + normalized * (limits[:, 1] - limits[:, 0]) / 2.0
    for index in gripper_indices:
        low = limits[index, 0]
        high = limits[index, 1]
        targets[:, index] = torch.where(actions[:, index] < 0.0, low, high)
    return targets
