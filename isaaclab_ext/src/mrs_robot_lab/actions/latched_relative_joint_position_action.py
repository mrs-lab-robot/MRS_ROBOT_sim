"""Relative joint actions whose absolute targets are held between control steps."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass


class LatchedRelativeJointPositionAction(ActionTerm):
    """Apply scaled relative deltas once per control step and latch absolute targets."""

    def __init__(self, cfg: RelativeJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        if len(self._joint_ids) == 0:
            raise ValueError("relative joint position action must select at least one joint")

        self._scale = self.cfg.scale
        action_shape = (self.num_envs, len(self._joint_ids))
        self._raw_actions = torch.zeros(action_shape, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._position_targets = torch.zeros(action_shape, device=self.device)
        # Rebase after reset on the first control step, after callers finish state writes.
        self._needs_relock = torch.ones((self.num_envs,), dtype=torch.bool, device=self.device)
        self.reset()

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

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
        self._IO_descriptor.action_type = "LatchedRelativeJointPosition"
        self._IO_descriptor.extras = {"joint_names": tuple(self._joint_names), "scale": self._scale}
        return self._IO_descriptor

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"expected relative joint action shape {tuple(self._raw_actions.shape)}, got {tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all().item()):
            raise ValueError("relative joint actions must contain only finite values")

        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self._scale
        current_positions = self._asset.data.joint_pos[:, self._joint_ids]
        relock = self._needs_relock
        self._position_targets[relock] = current_positions[relock]
        self._needs_relock[relock] = False
        updated_targets = current_positions + self._processed_actions
        # A zero delta means hold the existing target, not follow current joint motion.
        self._position_targets[:] = torch.where(
            self._raw_actions != 0.0,
            updated_targets,
            self._position_targets,
        )

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(
            target=self._position_targets,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        current_positions = self._asset.data.joint_pos[:, self._joint_ids]
        self._position_targets[env_ids] = current_positions[env_ids]
        self._needs_relock[env_ids] = True


@configclass
class LatchedRelativeJointPositionActionCfg(RelativeJointPositionActionCfg):
    """Configuration for a control-step-latched relative joint position action."""

    class_type: type[ActionTerm] = LatchedRelativeJointPositionAction
