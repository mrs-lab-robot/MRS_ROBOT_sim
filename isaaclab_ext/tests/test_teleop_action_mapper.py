from __future__ import annotations

import unittest

from mrs_robot_lab.adapters.teleop_adapter import action_from_command, joint_targets_from_action
from mrs_teleoperation.protocol import CommandFrame


class TeleopActionMapperTest(unittest.TestCase):
    def test_maps_absolute_ros_joint_targets_to_bounded_relative_arena_actions(self) -> None:
        current = {
            **{f"openarmx_left_joint{i}": 0.0 for i in range(1, 8)},
            **{f"openarmx_right_joint{i}": 0.0 for i in range(1, 8)},
            "lift_joint": 0.0,
            "openarmx_head_yaw_joint": 0.0,
            "openarmx_head_pitch_joint": 0.0,
            "openarmx_left_finger_joint1": 0.044,
            "openarmx_right_finger_joint1": 0.0,
        }
        command = CommandFrame(
            seq=1,
            deadman=True,
            left_arm_position=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            right_arm_position=(0.0,) * 7,
            lift_velocity=(0.1,),
            lift_position=(0.001,),
            head_position=(0.3, -0.3),
            left_gripper_position=(0.0,),
            right_gripper_position=(0.044,),
        )

        action = action_from_command(command, current, step_dt=1.0 / 15.0)

        self.assertEqual(len(action), 22)
        self.assertAlmostEqual(action[3], 0.0933333333, places=6)  # 0.35 rad/s * dt / 0.25 scale
        self.assertAlmostEqual(action[17], 0.02, places=6)  # absolute lift target takes priority over jog velocity
        self.assertAlmostEqual(action[18], 0.2222222222, places=6)  # 0.5 rad/s * dt / 0.15 scale
        self.assertEqual(action[20], -1.0)  # close left finger
        self.assertEqual(action[21], 1.0)  # open right finger

    def test_disconnected_or_deadman_released_input_holds_joints_and_stops_base(self) -> None:
        current = {name: 0.0 for name in (
            *(f"openarmx_left_joint{i}" for i in range(1, 8)),
            *(f"openarmx_right_joint{i}" for i in range(1, 8)),
            "lift_joint", "openarmx_head_yaw_joint", "openarmx_head_pitch_joint",
            "openarmx_left_finger_joint1", "openarmx_right_finger_joint1",
        )}
        action = action_from_command(
            CommandFrame(seq=1, deadman=False, base_twist=(0.5, 0.0, 0.3), left_arm_position=(1.0,) * 7),
            current,
            step_dt=1.0 / 15.0,
        )

        self.assertEqual(action[:20], [0.0] * 20)
        self.assertEqual(action[20:], [-1.0, -1.0])

    def test_no_packet_holds_the_current_gripper_state_instead_of_opening_it(self) -> None:
        current = {name: 0.0 for name in (
            *(f"openarmx_left_joint{i}" for i in range(1, 8)),
            *(f"openarmx_right_joint{i}" for i in range(1, 8)),
            "lift_joint", "openarmx_head_yaw_joint", "openarmx_head_pitch_joint",
            "openarmx_left_finger_joint1", "openarmx_right_finger_joint1",
        )}

        action = action_from_command(None, current, step_dt=1.0 / 15.0)

        self.assertEqual(action[:20], [0.0] * 20)
        self.assertEqual(action[20:], [-1.0, -1.0])

    def test_converts_bounded_action_deltas_to_lab_joint_targets(self) -> None:
        current = {name: 0.0 for name in (
            *(f"openarmx_left_joint{i}" for i in range(1, 8)),
            *(f"openarmx_right_joint{i}" for i in range(1, 8)),
            "lift_joint", "openarmx_head_yaw_joint", "openarmx_head_pitch_joint",
            "openarmx_left_finger_joint1", "openarmx_right_finger_joint1",
        )}
        command = CommandFrame(
            seq=1,
            deadman=True,
            left_arm_position=(0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            lift_position=(0.001,),
            left_gripper_position=(0.044,),
        )
        action = action_from_command(command, current, step_dt=1.0 / 15.0)

        targets = joint_targets_from_action(action, current)

        self.assertAlmostEqual(targets["openarmx_left_joint1"], 0.35 / 15.0)
        self.assertAlmostEqual(targets["lift_joint"], 0.001)
        self.assertAlmostEqual(targets["openarmx_left_finger_joint1"], 0.044)

    def test_joint_target_conversion_rejects_bad_action_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 22"):
            joint_targets_from_action([0.0], {})
        with self.assertRaisesRegex(ValueError, "finite"):
            joint_targets_from_action([0.0] * 21 + [float("nan")], {})


if __name__ == "__main__":
    unittest.main()
