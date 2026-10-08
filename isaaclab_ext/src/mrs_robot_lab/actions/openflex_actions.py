"""Contract-derived Isaac Lab action groups for the OpenFlex articulation."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.utils.configclass import configclass

from mrs_robot_lab.actions.base_actions import SwerveBaseActionCfg
from mrs_robot_lab.assets.robot_interface import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_LIMITS,
    LIFT_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_LIMITS,
    ACTION_SCALES,
)


@configclass
class OpenFlexActionsCfg:
    """Contract-ordered actions; the base action remains simulator-native."""

    base_twist_action: ActionTermCfg = SwerveBaseActionCfg(asset_name="robot")
    left_arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(LEFT_ARM_JOINTS), preserve_order=True,
        scale=ACTION_SCALES["left_arm_position"], use_zero_offset=True,
    )
    right_arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(RIGHT_ARM_JOINTS), preserve_order=True,
        scale=ACTION_SCALES["right_arm_position"], use_zero_offset=True,
    )
    lift_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(LIFT_JOINTS), preserve_order=True,
        scale=ACTION_SCALES["lift_position"], use_zero_offset=True,
    )
    head_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(HEAD_JOINTS), preserve_order=True,
        scale=ACTION_SCALES["head_position"], use_zero_offset=True,
    )
    left_gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEFT_GRIPPER_JOINTS),
        open_command_expr={LEFT_GRIPPER_JOINTS[0]: LEFT_GRIPPER_LIMITS[0][1]},
        close_command_expr={LEFT_GRIPPER_JOINTS[0]: LEFT_GRIPPER_LIMITS[0][0]},
    )
    right_gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(RIGHT_GRIPPER_JOINTS),
        open_command_expr={RIGHT_GRIPPER_JOINTS[0]: RIGHT_GRIPPER_LIMITS[0][1]},
        close_command_expr={RIGHT_GRIPPER_JOINTS[0]: RIGHT_GRIPPER_LIMITS[0][0]},
    )
