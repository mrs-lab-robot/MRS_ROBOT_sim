"""Joint-state and action observations for the canonical OpenFlex interface."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from mrs_robot_lab.assets.robot_interface import JOINT_STATE_NAMES


@configclass
class OpenFlexObservationsCfg:
    """Contract-ordered proprioception and the previous action."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_position = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_STATE_NAMES), preserve_order=True)},
        )
        joint_velocity = ObservationTermCfg(
            func=mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_STATE_NAMES), preserve_order=True)},
        )
        previous_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
