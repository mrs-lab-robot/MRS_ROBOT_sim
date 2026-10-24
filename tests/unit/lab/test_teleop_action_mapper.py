from __future__ import annotations

import unittest

from mrs_robot_lab.adapters.teleop_adapter import (
    _MAX_SPEEDS,
    action_from_command,
    joint_targets_from_action,
)
from mrs_robot_lab.assets.robot_interface import ACTION_SCALES
from mrs_teleoperation.protocol import CommandFrame


class TeleopActionMapperTest(unittest.TestCase):
    def test_lift_velocity_alone_is_relative_to_current_lift_position(self) -> None:
        """只有 lift_velocity 时不能崩，且必须是「当前升降位置 + 增量」。

        历史 bug：该分支直接复用臂循环里的局部变量 current，只有 lift_velocity 时抛
        UnboundLocalError；有臂指令时又把最后一个臂关节的位置当升降基准。
        """

        current = {
            **{f"openarmx_left_joint{i}": 0.0 for i in range(1, 8)},
            **{f"openarmx_right_joint{i}": 0.0 for i in range(1, 8)},
            "lift_joint": -0.2,
            "openarmx_head_yaw_joint": 0.0,
            "openarmx_head_pitch_joint": 0.0,
            "openarmx_left_finger_joint1": 0.044,
            "openarmx_right_finger_joint1": 0.0,
        }
        step_dt = 1.0 / 30.0

        only_lift = action_from_command(
            CommandFrame(seq=1, deadman=True, lift_velocity=(0.1,)), current, step_dt=step_dt
        )
        self.assertAlmostEqual(only_lift[17], -0.2 + 0.1 * step_dt, places=6)

        with_arm = action_from_command(
            CommandFrame(
                seq=2,
                deadman=True,
                left_arm_position=(0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
                lift_velocity=(0.1,),
            ),
            current,
            step_dt=step_dt,
        )
        self.assertAlmostEqual(with_arm[17], -0.2 + 0.1 * step_dt, places=6)

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
            # 0.9 rad：joint1 机械上限 0.9056 内，且远超单步限速，才能验证「被限速截断」
            left_arm_position=(0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            right_arm_position=(0.0,) * 7,
            lift_velocity=(0.1,),
            lift_position=(0.001,),
            head_position=(0.3, -0.3),
            left_gripper_position=(0.0,),
            right_gripper_position=(0.044,),
        )

        action = action_from_command(command, current, step_dt=1.0 / 15.0)

        self.assertEqual(len(action), 22)
        # 期望值从限速常量推出来，避免调速度后测试假红（曾写死 0.35 rad/s）。
        self.assertAlmostEqual(
            action[3], _MAX_SPEEDS["arm"] * (1.0 / 15.0) / ACTION_SCALES["left_arm_position"], places=6
        )
        # 升降是绝对位置语义：直接下发目标米值，不乘 scale（旧断言 0.02 是乘了 0.05 的结果）
        self.assertAlmostEqual(action[17], 0.001, places=6)
        self.assertAlmostEqual(
            action[18], _MAX_SPEEDS["head"] * (1.0 / 15.0) / ACTION_SCALES["head_position"], places=6
        )
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
            # 0.9 rad：joint1 机械上限 0.9056 内，且远超单步限速，才能验证「被限速截断」
            left_arm_position=(0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            lift_position=(0.001,),
            left_gripper_position=(0.044,),
        )
        action = action_from_command(command, current, step_dt=1.0 / 15.0)

        targets = joint_targets_from_action(action, current)

        self.assertAlmostEqual(
            targets["openarmx_left_joint1"], _MAX_SPEEDS["arm"] / 15.0
        )
        self.assertAlmostEqual(targets["lift_joint"], 0.001)
        self.assertAlmostEqual(targets["openarmx_left_finger_joint1"], 0.044)

    def test_joint_target_conversion_rejects_bad_action_vectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 22"):
            joint_targets_from_action([0.0], {})
        with self.assertRaisesRegex(ValueError, "finite"):
            joint_targets_from_action([0.0] * 21 + [float("nan")], {})


if __name__ == "__main__":
    unittest.main()
