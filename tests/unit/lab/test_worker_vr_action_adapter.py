from __future__ import annotations

import unittest

from mrs_robot_lab.teleoperation.vr_action_adapter import (
    normalize_base_twist,
    normalize_bilateral_arm_targets,
)


class WorkerVrActionAdapterTest(unittest.TestCase):
    def test_base_twist_is_normalized_and_clipped_against_asymmetric_contract_limits(self):
        actions = normalize_base_twist(
            (0.4, 1.2, -1.0),
            ((-0.8, 0.8), (-0.6, 1.0), (-2.0, 2.0)),
        )

        self.assertEqual(actions, (0.5, 1.0, -0.5))

    def test_bilateral_absolute_joint_targets_keep_task_order_and_binary_gripper_semantics(self):
        limits = ((-1.0, 1.0),) * 14 + ((0.0, 0.044), (0.0, 0.044))
        action = normalize_bilateral_arm_targets(
            (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044),
            limits,
        )

        self.assertEqual(action, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  -1.0, 1.0))

    def test_missing_arm_command_holds_that_arm_instead_of_resetting_it(self):
        limits = ((-1.0, 1.0),) * 14 + ((0.0, 0.044), (0.0, 0.044))
        held = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                -1.0, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7,
                -1.0, 1.0)

        action = normalize_bilateral_arm_targets(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044),
            None,
            limits,
            hold_action=held,
        )

        self.assertEqual(action, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  *held[7:14], 1.0, held[15]))

    def test_missing_both_arm_commands_requires_an_explicit_hold_target(self):
        with self.assertRaisesRegex(ValueError, "hold_action"):
            normalize_bilateral_arm_targets(None, None, ((-1.0, 1.0),) * 16)

    def test_rejects_malformed_or_non_finite_vr_commands(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_base_twist((0.1, float("nan"), 0.0), ((-1.0, 1.0),) * 3)
        with self.assertRaisesRegex(ValueError, "three finite"):
            normalize_base_twist((0.1, 0.0), ((-1.0, 1.0),) * 3)
        with self.assertRaisesRegex(ValueError, "eight values"):
            normalize_bilateral_arm_targets((0.0,) * 7, (0.0,) * 8, ((-1.0, 1.0),) * 16)
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_bilateral_arm_targets(
                (float("inf"),) + (0.0,) * 7,
                (0.0,) * 8,
                ((-1.0, 1.0),) * 16,
            )


if __name__ == "__main__":
    unittest.main()
