"""Map ROS physical setpoints to the contract-derived Isaac Lab action interface."""

from __future__ import annotations

import math

from mrs_robot_lab.assets.robot_interface import (
    ACTION_DIMENSION,
    ACTION_JOINTS_BY_TERM,
    ACTION_SCALES,
    ACTION_SLICES,
    HEAD_JOINTS,
    JOINT_STATE_NAMES,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_LIMITS,
    LIFT_JOINTS,
    JOINT_POSITION_LIMITS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_LIMITS,
)
from mrs_teleoperation.protocol import CommandFrame


ARM_NAMES = {"left": LEFT_ARM_JOINTS, "right": RIGHT_ARM_JOINTS}
_MAX_SPEEDS = {
    "arm": 0.35,  # rad/s; bounded independently of the existing IK step limiter.
    "lift": 0.05,  # m/s.
    "head": 0.5,  # rad/s.
}


def _finite_clamp(value: float, lower: float, upper: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("teleoperation setpoints must be finite")
    return max(lower, min(upper, number))


def action_from_command(
    command: CommandFrame | None,
    current_joint_positions: dict[str, float],
    *,
    step_dt: float,
    motion_enabled: bool | None = None,
) -> list[float]:
    """Produce one action while preserving the Lab action group's declared order.

    The ROS stack publishes absolute joint targets; Isaac Lab consumes normalized
    relative joint actions. This adapter rate-limits target deltas and scales them
    according to the single Lab action-scale configuration.
    """

    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be a positive finite duration")
    enabled = bool(command and command.deadman and not command.estop) if motion_enabled is None else motion_enabled
    action = [0.0] * ACTION_DIMENSION
    base_start, base_width = ACTION_SLICES["base_twist_action"]
    if enabled and command is not None and command.base_twist is not None:
        action[base_start:base_start + base_width] = [
            _finite_clamp(command.base_twist[0], -0.8, 0.8),
            _finite_clamp(command.base_twist[1], -0.8, 0.8),
            _finite_clamp(command.base_twist[2], -2.320037210795224, 2.320037210795224),
        ]

    if enabled and command is not None:
        for side, target_field, action_term in (
            ("left", command.left_arm_position, "left_arm_action"),
            ("right", command.right_arm_position, "right_arm_action"),
        ):
            if target_field is None:
                continue
            action_start, _ = ACTION_SLICES[action_term]
            scale = ACTION_SCALES[f"{side}_arm_position"]
            for offset, (joint_name, target) in enumerate(zip(ARM_NAMES[side], target_field, strict=True)):
                current = float(current_joint_positions[joint_name])
                delta_limit = _MAX_SPEEDS["arm"] * step_dt
                bounded_target = _finite_clamp(float(target), *JOINT_POSITION_LIMITS[joint_name])
                delta = _finite_clamp(bounded_target - current, -delta_limit, delta_limit)
                action[action_start + offset] = delta / scale

        if command.lift_position is not None:
            action_start, _ = ACTION_SLICES["lift_action"]
            scale = ACTION_SCALES["lift_position"]
            current = float(current_joint_positions[LIFT_JOINTS[0]])
            target = _finite_clamp(command.lift_position[0], *JOINT_POSITION_LIMITS[LIFT_JOINTS[0]])
            delta_limit = _MAX_SPEEDS["lift"] * step_dt
            action[action_start] = _finite_clamp(target - current, -delta_limit, delta_limit) / scale
        elif command.lift_velocity is not None:
            action_start, _ = ACTION_SLICES["lift_action"]
            scale = ACTION_SCALES["lift_position"]
            lift_delta = _finite_clamp(command.lift_velocity[0], -_MAX_SPEEDS["lift"], _MAX_SPEEDS["lift"]) * step_dt
            action[action_start] = lift_delta / scale

        if command.head_position is not None:
            action_start, _ = ACTION_SLICES["head_action"]
            scale = ACTION_SCALES["head_position"]
            for offset, (joint_name, target) in enumerate(zip(HEAD_JOINTS, command.head_position, strict=True)):
                current = float(current_joint_positions[joint_name])
                delta_limit = _MAX_SPEEDS["head"] * step_dt
                bounded_target = _finite_clamp(float(target), *JOINT_POSITION_LIMITS[joint_name])
                delta = _finite_clamp(bounded_target - current, -delta_limit, delta_limit)
                action[action_start + offset] = delta / scale

    for action_term, joints, limits, desired in (
        (
            "left_gripper_action",
            LEFT_GRIPPER_JOINTS,
            LEFT_GRIPPER_LIMITS,
            command.left_gripper_position if command else None,
        ),
        (
            "right_gripper_action",
            RIGHT_GRIPPER_JOINTS,
            RIGHT_GRIPPER_LIMITS,
            command.right_gripper_position if command else None,
        ),
    ):
        action_index, _ = ACTION_SLICES[action_term]
        low, high = limits[0]
        position = desired[0] if enabled and desired is not None else float(current_joint_positions[joints[0]])
        # Existing BinaryJointPositionAction uses non-negative=open, negative=close.
        action[action_index] = 1.0 if _finite_clamp(position, low, high) > (low + high) / 2 else -1.0

    return action


def joint_targets_from_action(
    action: list[float] | tuple[float, ...], current_joint_positions: dict[str, float]
) -> dict[str, float]:
    """Convert normalized Lab/Arena action deltas to bounded physical joint targets."""

    if len(action) != ACTION_DIMENSION:
        raise ValueError(f"expected {ACTION_DIMENSION} actions, got {len(action)}")
    if not all(math.isfinite(float(value)) for value in action):
        raise ValueError("actions must contain only finite values")

    targets: dict[str, float] = {}
    for term, contract_term in (
        ("left_arm_action", "left_arm_position"),
        ("right_arm_action", "right_arm_position"),
        ("lift_action", "lift_position"),
        ("head_action", "head_position"),
    ):
        start, width = ACTION_SLICES[term]
        scale = ACTION_SCALES[contract_term]
        for offset, joint_name in enumerate(ACTION_JOINTS_BY_TERM[term]):
            target = float(current_joint_positions[joint_name]) + float(action[start + offset]) * scale
            targets[joint_name] = _finite_clamp(target, *JOINT_POSITION_LIMITS[joint_name])

    for term in ("left_gripper_action", "right_gripper_action"):
        start, _width = ACTION_SLICES[term]
        joint_name = ACTION_JOINTS_BY_TERM[term][0]
        low, high = JOINT_POSITION_LIMITS[joint_name]
        targets[joint_name] = high if float(action[start]) > 0.0 else low
    return targets


__all__ = [
    "ACTION_DIMENSION",
    "ARM_NAMES",
    "JOINT_STATE_NAMES",
    "action_from_command",
    "joint_targets_from_action",
]
