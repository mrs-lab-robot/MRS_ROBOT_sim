"""Small, bounded synthetic command source for GUI capture smoke tests."""

from __future__ import annotations

import math

from mrs_robot_lab.assets.robot_interface import (
    HEAD_JOINTS,
    JOINT_POSITION_LIMITS,
    JOINT_STATE_NAMES,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LIFT_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
)
from mrs_teleoperation.protocol import CommandFrame


class SyntheticVrSource:
    """Generate slow, low-amplitude setpoints around the current simulated pose."""

    _AMPLITUDE_RAD = 0.05
    _FREQUENCY_HZ = 0.25

    def __init__(self) -> None:
        self._home: dict[str, float] | None = None

    def start(self, joint_positions: dict[str, float]) -> None:
        missing = set(JOINT_STATE_NAMES) - set(joint_positions)
        if missing:
            raise ValueError(f"synthetic VR home pose is missing joints: {sorted(missing)}")
        home = {name: float(joint_positions[name]) for name in JOINT_STATE_NAMES}
        if not all(math.isfinite(value) for value in home.values()):
            raise ValueError("synthetic VR home pose must contain finite joint positions")
        self._home = home

    def next_frame(self, *, seq: int, source_time_ns: int) -> CommandFrame:
        if self._home is None:
            raise RuntimeError("synthetic VR source has not been started with a robot pose")
        if seq < 0 or source_time_ns < 0:
            raise ValueError("synthetic VR sequence and timestamp must be non-negative")

        phase = 2.0 * math.pi * self._FREQUENCY_HZ * (source_time_ns / 1_000_000_000.0)

        def targets(names: tuple[str, ...], joint_offset: tuple[float, ...]) -> tuple[float, ...]:
            values = []
            for name, offset in zip(names, joint_offset, strict=True):
                lower, upper = JOINT_POSITION_LIMITS[name]
                values.append(max(lower, min(upper, self._home[name] + offset)))
            return tuple(values)

        left_offsets = (self._AMPLITUDE_RAD * math.sin(phase),) + (0.0,) * (len(LEFT_ARM_JOINTS) - 1)
        right_offsets = (0.0, self._AMPLITUDE_RAD * math.sin(phase + math.pi / 2)) + (
            (0.0,) * (len(RIGHT_ARM_JOINTS) - 2)
        )
        head_offsets = (0.0, 0.025 * math.sin(phase))
        return CommandFrame(
            seq=int(seq),
            source_time_ns=int(source_time_ns),
            session_id=1,
            deadman=True,
            estop=False,
            base_twist=(0.0, 0.0, 0.0),
            left_arm_position=targets(LEFT_ARM_JOINTS, left_offsets),
            right_arm_position=targets(RIGHT_ARM_JOINTS, right_offsets),
            lift_position=targets(LIFT_JOINTS, (0.0,)),
            head_position=targets(HEAD_JOINTS, head_offsets),
            left_gripper_position=targets(LEFT_GRIPPER_JOINTS, (0.0,)),
            right_gripper_position=targets(RIGHT_GRIPPER_JOINTS, (0.0,)),
        )


__all__ = ["SyntheticVrSource"]
