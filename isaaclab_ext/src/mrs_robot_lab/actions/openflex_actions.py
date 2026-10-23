"""Contract-derived Isaac Lab action groups for the OpenFlex articulation."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, JointPositionActionCfg
from mrs_robot_lab.actions.latched_relative_joint_position_action import (
    LatchedRelativeJointPositionActionCfg,
)
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
    left_arm_action: ActionTermCfg = LatchedRelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(LEFT_ARM_JOINTS), preserve_order=True,
        scale=ACTION_SCALES["left_arm_position"], use_zero_offset=True,
    )
    right_arm_action: ActionTermCfg = LatchedRelativeJointPositionActionCfg(
        asset_name="robot", joint_names=list(RIGHT_ARM_JOINTS), preserve_order=True,
        scale=ACTION_SCALES["right_arm_position"], use_zero_offset=True,
    )
    # 升降用「绝对位置」动作：relay 算出来的就是绝对目标（米），
    # 相对动作的“零=保持/增量”语义在这里反复出错（立柱掉地、指令反向）。
    lift_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=list(LIFT_JOINTS), preserve_order=True,
        scale=1.0, use_default_offset=False,
    )
    head_action: ActionTermCfg = LatchedRelativeJointPositionActionCfg(
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
