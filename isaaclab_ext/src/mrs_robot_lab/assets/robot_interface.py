"""Simulator-independent, contract-derived MRS Robot interface constants."""

from __future__ import annotations

import os
from pathlib import Path

from mrs_robot_lab.assets.asset_resolver import AssetResolver
from mrs_robot_lab.contracts import ContractError, EmbodimentContract, load_embodiment_contract
from mrs_robot_lab.controllers.swerve import SwerveConfig, load_swerve_config


def _checkout_root() -> Path:
    """Resolve an explicit root or the enclosing source checkout without machine paths."""

    configured_root = os.environ.get("MRS_ROBOT_SIM_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


ROBOT_RESOLVER = AssetResolver(_checkout_root())
ROBOT_CONTRACT_PATH = ROBOT_RESOLVER.resolve("openflex_embodiment_contract")
ROBOT_CONTRACT: EmbodimentContract = load_embodiment_contract(ROBOT_CONTRACT_PATH)
if ROBOT_CONTRACT.robot_id != "openflex":
    raise ContractError(f"expected the OpenFleX contract, got robot id {ROBOT_CONTRACT.robot_id!r}")

SWERVE_CONFIG: SwerveConfig = load_swerve_config(
    ROBOT_RESOLVER.resolve("openflex_swerve_controller"), ROBOT_CONTRACT_PATH
)


def _joint_state_names(contract: EmbodimentContract) -> tuple[str, ...]:
    try:
        fields = contract.observation_fields["joint_state"]
    except KeyError as error:
        raise ContractError("the robot contract does not define joint_state observation fields") from error
    positions = tuple(field.removeprefix("position.") for field in fields if field.startswith("position."))
    velocities = tuple(field.removeprefix("velocity.") for field in fields if field.startswith("velocity."))
    if not positions or positions != velocities:
        raise ContractError("joint_state must list matching position and velocity joint names")
    return positions


JOINT_STATE_NAMES = _joint_state_names(ROBOT_CONTRACT)
LEFT_ARM_JOINTS = ROBOT_CONTRACT.action_joints["left_arm_position"]
RIGHT_ARM_JOINTS = ROBOT_CONTRACT.action_joints["right_arm_position"]
LIFT_JOINTS = ROBOT_CONTRACT.action_joints["lift_position"]
HEAD_JOINTS = ROBOT_CONTRACT.action_joints["head_position"]
LEFT_GRIPPER_JOINTS = ROBOT_CONTRACT.action_joints["left_gripper_position"]
RIGHT_GRIPPER_JOINTS = ROBOT_CONTRACT.action_joints["right_gripper_position"]
LEFT_GRIPPER_LIMITS = ROBOT_CONTRACT.action_limits["left_gripper_position"]
RIGHT_GRIPPER_LIMITS = ROBOT_CONTRACT.action_limits["right_gripper_position"]

ACTION_JOINTS_BY_TERM = {
    "left_arm_action": LEFT_ARM_JOINTS,
    "right_arm_action": RIGHT_ARM_JOINTS,
    "lift_action": LIFT_JOINTS,
    "head_action": HEAD_JOINTS,
    "left_gripper_action": LEFT_GRIPPER_JOINTS,
    "right_gripper_action": RIGHT_GRIPPER_JOINTS,
}
ACTION_TERM_LAYOUT = (
    ("base_twist_action", "base_twist"),
    ("left_arm_action", "left_arm_position"),
    ("right_arm_action", "right_arm_position"),
    ("lift_action", "lift_position"),
    ("head_action", "head_position"),
    ("left_gripper_action", "left_gripper_position"),
    ("right_gripper_action", "right_gripper_position"),
)
ACTION_SLICES: dict[str, tuple[int, int]] = {}
_action_offset = 0
for _term_name, _contract_action_name in ACTION_TERM_LAYOUT:
    _width = ROBOT_CONTRACT.action_dimensions[_contract_action_name]
    ACTION_SLICES[_term_name] = (_action_offset, _width)
    _action_offset += _width
ACTION_DIMENSION = _action_offset
ACTION_SCALES = {
    "left_arm_position": 0.25,
    "right_arm_position": 0.25,
    "lift_position": 0.05,
    "head_position": 0.15,
}

JOINT_POSITION_LIMITS: dict[str, tuple[float, float]] = {}
for _action_name, _joints in ROBOT_CONTRACT.action_joints.items():
    if not _action_name.endswith("_position"):
        continue
    for _joint_name, _limits in zip(_joints, ROBOT_CONTRACT.action_limits[_action_name], strict=True):
        JOINT_POSITION_LIMITS[_joint_name] = _limits
for _joint_name, _limits in zip(SWERVE_CONFIG.steering_joint_names, SWERVE_CONFIG.steering_limits, strict=True):
    JOINT_POSITION_LIMITS[_joint_name] = _limits

__all__ = [
    "ACTION_DIMENSION",
    "ACTION_JOINTS_BY_TERM",
    "ACTION_SCALES",
    "ACTION_SLICES",
    "ACTION_TERM_LAYOUT",
    "HEAD_JOINTS",
    "JOINT_POSITION_LIMITS",
    "JOINT_STATE_NAMES",
    "LEFT_ARM_JOINTS",
    "LEFT_GRIPPER_JOINTS",
    "LEFT_GRIPPER_LIMITS",
    "LIFT_JOINTS",
    "RIGHT_ARM_JOINTS",
    "RIGHT_GRIPPER_JOINTS",
    "RIGHT_GRIPPER_LIMITS",
    "ROBOT_CONTRACT",
    "ROBOT_CONTRACT_PATH",
    "ROBOT_RESOLVER",
    "SWERVE_CONFIG",
]
