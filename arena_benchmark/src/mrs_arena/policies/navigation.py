"""Bridge the normalized navigation checkpoint output to Arena's SI action term."""

from __future__ import annotations

import torch

from mrs_robot_lab.assets.robot_interface import SWERVE_CONFIG
from mrs_robot_lab.controllers.swerve_tensor import normalized_to_body_twist


def policy_actions_to_body_twist(actions: torch.Tensor) -> torch.Tensor:
    """Map PPO ``[-1, 1]`` actions into contract-limited ``(m/s, m/s, rad/s)``."""
    return normalized_to_body_twist(actions, SWERVE_CONFIG)
