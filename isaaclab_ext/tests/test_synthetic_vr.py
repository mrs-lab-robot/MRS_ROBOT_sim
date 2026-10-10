from __future__ import annotations

import unittest

from mrs_robot_lab.adapters.synthetic_vr import SyntheticVrSource
from mrs_robot_lab.assets.robot_interface import (
    HEAD_JOINTS,
    JOINT_POSITION_LIMITS,
    JOINT_STATE_NAMES,
    LEFT_ARM_JOINTS,
    LIFT_JOINTS,
    RIGHT_ARM_JOINTS,
)


class SyntheticVrSourceTest(unittest.TestCase):
    def _started_source(self) -> SyntheticVrSource:
        home = {name: 0.0 for name in JOINT_STATE_NAMES}
        source = SyntheticVrSource()
        source.start(home)
        return source

    def test_generates_visibly_large_command_motion_with_deadman(self) -> None:
        source = self._started_source()

        initial = source.next_frame(seq=1, source_time_ns=0)
        peak = source.next_frame(seq=2, source_time_ns=1_000_000_000)

        self.assertTrue(peak.deadman)
        self.assertFalse(peak.estop)
        self.assertGreater(abs(peak.base_twist[2]), 0.2)
        self.assertGreater(abs(peak.head_position[0]), 0.2)
        self.assertGreater(abs(initial.head_position[1]), 0.1)
        self.assertGreater(abs(peak.left_arm_position[1]), 0.1)
        self.assertGreater(abs(peak.right_arm_position[1]), 0.1)
        self.assertLess(peak.left_arm_position[0] * peak.right_arm_position[0], 0.0)

    def test_all_targets_remain_inside_robot_joint_limits(self) -> None:
        source = self._started_source()
        frame = source.next_frame(seq=3, source_time_ns=1_000_000_000)

        named_targets = {
            name: value
            for names, values in (
                (LEFT_ARM_JOINTS, frame.left_arm_position),
                (RIGHT_ARM_JOINTS, frame.right_arm_position),
                (HEAD_JOINTS, frame.head_position),
                (LIFT_JOINTS, frame.lift_position),
            )
            for name, value in zip(names, values, strict=True)
        }
        for name, value in named_targets.items():
            lower, upper = JOINT_POSITION_LIMITS[name]
            self.assertGreaterEqual(value, lower - 1e-9)
            self.assertLessEqual(value, upper + 1e-9)

    def test_requires_a_robot_home_pose_before_generating_motion(self) -> None:
        source = SyntheticVrSource()
        with self.assertRaisesRegex(RuntimeError, "started"):
            source.next_frame(seq=1, source_time_ns=0)


if __name__ == "__main__":
    unittest.main()
