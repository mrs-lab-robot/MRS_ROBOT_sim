"""Isaac Lab action terms for native OpenFlex mobile-base control."""

from __future__ import annotations

import torch
from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass

from mrs_robot_lab.assets.asset_resolver import AssetResolver
from mrs_robot_lab.contracts import load_embodiment_contract
from mrs_robot_lab.controllers.swerve import load_swerve_config
from mrs_robot_lab.controllers.swerve_tensor import compute_swerve_targets_tensor


class SwerveBaseAction(ActionTerm):
    """Body-frame ``(vx, vy, yaw_rate)`` control for the four OpenFlex swerve modules."""

    def __init__(self, cfg: ActionTermCfg, env):
        super().__init__(cfg, env)
        resolver = AssetResolver()
        contract_path = resolver.resolve("openflex_embodiment_contract")
        self._swerve = load_swerve_config(resolver.resolve("openflex_swerve_controller"), contract_path)
        contract = load_embodiment_contract(contract_path)
        self._action_fields = contract.action_fields.get("base_twist", ())
        if len(self._action_fields) != 3:
            raise ValueError(f"expected three base_twist fields, got {self._action_fields}")

        steering_ids, steering_names = self._asset.find_joints(
            list(self._swerve.steering_joint_names), preserve_order=True
        )
        wheel_ids, wheel_names = self._asset.find_joints(list(self._swerve.wheel_joint_names), preserve_order=True)
        if tuple(steering_names) != self._swerve.steering_joint_names:
            raise ValueError(f"OpenFlex steering joints differ from controller config: {steering_names}")
        if tuple(wheel_names) != self._swerve.wheel_joint_names:
            raise ValueError(f"OpenFlex wheel joints differ from controller config: {wheel_names}")
        self._steering_ids = steering_ids
        self._wheel_ids = wheel_ids

        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._steering_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self._wheel_speed_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self._previous_wheel_speeds = torch.zeros_like(self._wheel_speed_targets)
        self._wheel_radius = self._swerve.wheel_radius
        self._max_wheel_speed = self._swerve.max_wheel_speed

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "BaseTwist"
        self._IO_descriptor.extras = {
            "fields": self._action_fields,
            "units": ("m/s", "m/s", "rad/s"),
            "steering_joint_names": self._swerve.steering_joint_names,
            "wheel_joint_names": self._swerve.wheel_joint_names,
            "wheel_radius_m": self._wheel_radius,
            "max_wheel_speed_mps": self._max_wheel_speed,
        }
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"expected base action shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}")
        if not bool(torch.isfinite(actions).all().item()):
            raise ValueError("base Twist action must contain only finite values")
        self._raw_actions.copy_(actions)
        steering, speeds, processed = compute_swerve_targets_tensor(
            actions,
            self._swerve,
            self._previous_wheel_speeds,
            dt=self._env.step_dt,
        )
        self._processed_actions.copy_(processed)
        idle = self._processed_actions.abs().amax(dim=1, keepdim=True) <= 1e-9
        current_steering = self._asset.data.joint_pos[:, self._steering_ids]
        steering = torch.where(idle, current_steering, steering)
        self._steering_targets.copy_(steering)
        self._wheel_speed_targets.copy_(speeds)
        self._previous_wheel_speeds.copy_(speeds)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._steering_targets,
            joint_ids=self._steering_ids,
        )
        self._asset.set_joint_velocity_target_index(
            target=self._wheel_speed_targets / self._wheel_radius,
            joint_ids=self._wheel_ids,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._wheel_speed_targets[env_ids] = 0.0
        self._previous_wheel_speeds[env_ids] = 0.0


@configclass
class SwerveBaseActionCfg(ActionTermCfg):
    """Configuration for the contract-backed OpenFlex swerve base action."""

    class_type: type[ActionTerm] = SwerveBaseAction
