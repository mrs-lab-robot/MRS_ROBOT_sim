"""Isaac Lab to Torch boundary for the shared navigation observation contract."""

from __future__ import annotations

import torch

from mrs_arena.tasks.navigation_logic import (
    navigation_observation_tensor,
    quaternion_xyzw_to_yaw,
)


def _torch_view(value, *, field_name: str) -> torch.Tensor:
    """Return Torch tensors directly and unwrap Isaac Lab 3.x ProxyArray views."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    raise TypeError(f"{field_name} must be a Torch tensor or expose an Isaac Lab .torch view")


def navigation_state_tensors(robot, env_origins):
    """Read navigation state through explicit Torch views at the Isaac Lab boundary."""
    data = robot.data
    position_world = _torch_view(data.root_pos_w, field_name="root_pos_w")
    quaternion_world = _torch_view(data.root_quat_w, field_name="root_quat_w")
    velocity_world = _torch_view(data.root_lin_vel_w, field_name="root_lin_vel_w")
    angular_velocity_world = _torch_view(data.root_ang_vel_w, field_name="root_ang_vel_w")
    origins = _torch_view(env_origins, field_name="env_origins")
    position_xy = position_world[:, :2] - origins[:, :2]
    yaw = quaternion_xyzw_to_yaw(quaternion_world)
    return position_xy, yaw, velocity_world[:, :2], angular_velocity_world[:, 2]


def navigation_policy_observation(
    env,
    *,
    target_xy: tuple[float, float],
    target_yaw_rad: float,
    asset_cfg,
) -> torch.Tensor:
    """Produce the same goal-relative odometry vector used to train navigation PPO."""
    robot = env.scene[asset_cfg.name]
    position_xy, yaw, velocity_world_xy, angular_velocity_z = navigation_state_tensors(
        robot,
        env.scene.env_origins,
    )
    return navigation_observation_tensor(
        position_xy,
        yaw,
        velocity_world_xy,
        angular_velocity_z,
        target_xy=target_xy,
        target_yaw_rad=target_yaw_rad,
    )
