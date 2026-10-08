from __future__ import annotations

import unittest

from mrs_robot_lab.adapters.synthetic_vr import SyntheticVrSource
from mrs_robot_lab.assets.robot_interface import JOINT_STATE_NAMES


class SyntheticVrSourceTest(unittest.TestCase):
    def test_produces_bounded_deadman_command_frames_from_robot_home_pose(self) -> None:
        home = {name: 0.0 for name in JOINT_STATE_NAMES}
        source = SyntheticVrSource()
        source.start(home)

        first = source.next_frame(seq=1, source_time_ns=10_000_000)
        later = source.next_frame(seq=30, source_time_ns=300_000_000)

        self.assertTrue(first.deadman)
        self.assertFalse(first.estop)
        self.assertEqual(first.base_twist, (0.0, 0.0, 0.0))
        self.assertEqual(len(first.left_arm_position), 7)
        self.assertEqual(len(first.right_arm_position), 7)
        self.assertNotEqual(first.left_arm_position, later.left_arm_position)
        self.assertLessEqual(max(abs(value) for value in first.left_arm_position), 0.1)
        self.assertLessEqual(max(abs(value) for value in first.right_arm_position), 0.1)

    def test_requires_a_robot_home_pose_before_generating_motion(self) -> None:
        source = SyntheticVrSource()
        with self.assertRaisesRegex(RuntimeError, "started"):
            source.next_frame(seq=1, source_time_ns=0)


if __name__ == "__main__":
    unittest.main()
