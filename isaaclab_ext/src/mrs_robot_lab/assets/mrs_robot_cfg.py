"""Isaac Lab articulation configuration for the existing canonical robot USD.

Import this module after Isaac Sim has started. Contract parsing and action/
observation definitions live in their own simulator-adapter modules.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

from mrs_robot_lab.actions.openflex_actions import OpenFlexActionsCfg
from mrs_robot_lab.assets.robot_interface import (
    JOINT_STATE_NAMES,
    ROBOT_CONTRACT,
    ROBOT_RESOLVER,
    SWERVE_CONFIG,
)
from mrs_robot_lab.observations.proprioception import OpenFlexObservationsCfg


def _initial_joint_positions() -> dict[str, float]:
    joint_names = (
        *ROBOT_CONTRACT.joint_action_names,
        *SWERVE_CONFIG.steering_joint_names,
        *SWERVE_CONFIG.wheel_joint_names,
    )
    positions = dict.fromkeys(joint_names, 0.0)
    left_finger = ROBOT_CONTRACT.action_joints["left_gripper_position"][0]
    right_finger = ROBOT_CONTRACT.action_joints["right_gripper_position"][0]
    positions[left_finger] = ROBOT_CONTRACT.action_limits["left_gripper_position"][0][1]
    positions[right_finger] = ROBOT_CONTRACT.action_limits["right_gripper_position"][0][1]
    return positions


MRS_ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_RESOLVER.resolve("openflex_robot_usd")),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=_initial_joint_positions(),
    ),
    soft_joint_pos_limit_factor=1.0,
    # Preserve the stiffness/damping and effort values authored on the USD joints.
    actuators={
        "usd_authored_drives": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
            effort_limit=None,
            velocity_limit=None,
        )
    },
)


__all__ = [
    "JOINT_STATE_NAMES",
    "MRS_ROBOT_CFG",
    "OpenFlexActionsCfg",
    "OpenFlexObservationsCfg",
]
