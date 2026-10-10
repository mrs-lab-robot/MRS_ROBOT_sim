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

    def test_produces_bounded_deadman_command_frames_from_robot_home_pose(self) -> None:
        source = self._started_source()

        first = source.next_frame(seq=1, source_time_ns=10_000_000)
        later = source.next_frame(seq=30, source_time_ns=300_000_000)

        self.assertTrue(first.deadman)
        self.assertFalse(first.estop)
        self.assertEqual(first.base_twist[:2], (0.0, 0.0))
        self.assertLess(abs(first.base_twist[2]), 0.02)
        self.assertEqual(len(first.left_arm_position), 7)
        self.assertEqual(len(first.right_arm_position), 7)
        self.assertNotEqual(first.left_arm_position, later.left_arm_position)
        # 起始瞬间（~0.016 rad 相位）仍在极小值附近
        self.assertLessEqual(max(abs(value) for value in first.left_arm_position), 0.02)
        self.assertLessEqual(max(abs(value) for value in first.right_arm_position), 0.02)

    def test_motion_is_visible_and_stays_within_joint_limits(self) -> None:
        source = self._started_source()
        # 1 秒 = 四分之一周期：正弦到达峰值
        frame = source.next_frame(seq=10, source_time_ns=1_000_000_000)

        left_shoulder = frame.left_arm_position[1]
        right_shoulder = frame.right_arm_position[1]
        head_yaw = frame.head_position[0]
        # 可见动作：底盘摇摆 ±0.40 rad/s、头部 ±0.35、肩 ±0.30（限位截断后仍明显）
        self.assertGreater(abs(frame.base_twist[2]), 0.2)
        self.assertGreater(abs(head_yaw), 0.2)
        # 点头（cos 相位）在 t=0 达到峰值
        early = source.next_frame(seq=2, source_time_ns=0)
        self.assertGreater(abs(early.head_position[1]), 0.1)
        self.assertGreater(abs(left_shoulder), 0.1)
        self.assertGreater(abs(right_shoulder), 0.1)
        self.assertLess(left_shoulder * right_shoulder, 0.0)  # 反相
        self.assertGreater(abs(frame.left_arm_position[0]), 0.3)
        self.assertLess(frame.right_arm_position[0], 0.0)  # 与左臂反向

        # 全帧都在限位内
        checked = {
            name: value
            for names, values in (
                (LEFT_ARM_JOINTS, frame.left_arm_position),
                (RIGHT_ARM_JOINTS, frame.right_arm_position),
                (HEAD_JOINTS, frame.head_position),
                (LIFT_JOINTS, frame.lift_position),
            )
            for name, value in zip(names, values, strict=True)
        }
        for name, value in checked.items():
            lower, upper = JOINT_POSITION_LIMITS[name]
            self.assertGreaterEqual(value, lower - 1e-9)
            self.assertLessEqual(value, upper + 1e-9)

    def test_requires_a_robot_home_pose_before_generating_motion(self) -> None:
        source = SyntheticVrSource()
        with self.assertRaisesRegex(RuntimeError, "started"):
            source.next_frame(seq=1, source_time_ns=0)


if __name__ == "__main__":
    unittest.main()
