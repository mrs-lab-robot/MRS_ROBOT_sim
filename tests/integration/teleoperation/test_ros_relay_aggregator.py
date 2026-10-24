from __future__ import annotations

import unittest

from mrs_robot_arena_bridge.aggregator import CommandAggregator


class CommandAggregatorTest(unittest.TestCase):
    def test_collects_ros_controller_outputs_into_named_command_frame(self) -> None:
        relay = CommandAggregator(deadman_timeout=0.25)
        relay.update_vr_pose(now=2.0)
        relay.update_controller_pose("left", [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        relay.update_hand_value("left", "grip", 0.7)
        relay.update_base_twist(0.2, -0.1, 0.3, now=2.0)
        relay.update_arm("left", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01])
        relay.update_head([0.1, -0.2])
        relay.update_lift_velocity(0.04, now=2.0)
        relay.update_lift_position(0.18)

        frame = relay.command_frame(now=2.1)

        self.assertTrue(frame.deadman)
        self.assertEqual(frame.base_twist, (0.2, -0.1, 0.3))
        self.assertEqual(frame.left_arm_position, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self.assertEqual(frame.left_gripper_position, (0.01,))
        self.assertEqual(frame.head_position, (0.1, -0.2))
        self.assertEqual(frame.lift_velocity, (0.04,))
        self.assertEqual(frame.lift_position, (0.18,))
        self.assertEqual(frame.left_controller_pose, (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0))
        self.assertEqual(frame.left_grip, (0.7,))

    def test_stale_vr_or_estop_assertion_disables_deadman(self) -> None:
        relay = CommandAggregator(deadman_timeout=0.25)
        relay.update_vr_pose(now=2.0)
        self.assertFalse(relay.command_frame(now=2.251).deadman)
        relay.update_vr_pose(now=3.0)
        relay.update_estop(True)
        self.assertTrue(relay.command_frame(now=3.01).estop)
        self.assertFalse(relay.command_frame(now=3.01).deadman)

    def test_ros_mode_keeps_joint_targets_without_vr_and_expires_velocity_commands(self) -> None:
        relay = CommandAggregator(deadman_timeout=0.25, control_mode="ros")
        relay.update_base_twist(0.2, 0.0, 0.0, now=4.0)
        relay.update_lift_velocity(0.04, now=4.0)
        relay.update_arm("left", [0.1] * 7)
        relay.update_lift_position(0.2)

        fresh = relay.command_frame(now=4.1)
        stale = relay.command_frame(now=4.3)

        self.assertTrue(fresh.deadman)
        self.assertEqual(fresh.base_twist, (0.2, 0.0, 0.0))
        self.assertEqual(fresh.lift_velocity, (0.04,))
        self.assertEqual(stale.base_twist, (0.0, 0.0, 0.0))
        self.assertEqual(stale.lift_velocity, (0.0,))
        self.assertTrue(stale.deadman)
        self.assertEqual(stale.left_arm_position, (0.1,) * 7)
        self.assertEqual(stale.lift_position, (0.2,))

    def test_rejects_unknown_control_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "control_mode"):
            CommandAggregator(control_mode="unsafe")


if __name__ == "__main__":
    unittest.main()
