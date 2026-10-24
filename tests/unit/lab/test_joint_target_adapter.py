from __future__ import annotations

import unittest

import torch

try:
    from mrs_robot_lab.actions.joint_target_adapter import normalized_to_joint_targets
    ADAPTER_IMPORT_ERROR = None
except ImportError as error:
    normalized_to_joint_targets = None
    ADAPTER_IMPORT_ERROR = error


class JointTargetAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        if ADAPTER_IMPORT_ERROR is not None:
            self.fail(f"named joint target adapter is missing: {ADAPTER_IMPORT_ERROR}")

    def test_arm_targets_map_normalized_actions_to_physical_joint_limits(self) -> None:
        actions = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
        limits = ((-2.0, 2.0), (-1.0, 3.0), (0.5, 1.5))

        targets = normalized_to_joint_targets(actions, limits, gripper_indices=())

        self.assertTrue(torch.allclose(targets, torch.tensor([[-2.0, 1.0, 1.5]])))

    def test_grippers_use_explicit_open_close_commands_not_midpoint(self) -> None:
        actions = torch.tensor([[0.2, -0.2]], dtype=torch.float32)
        limits = ((0.0, 0.044), (0.0, 0.044))

        targets = normalized_to_joint_targets(actions, limits, gripper_indices=(0, 1))

        self.assertTrue(torch.allclose(targets, torch.tensor([[0.044, 0.0]])))

    def test_rejects_action_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "action dimension"):
            normalized_to_joint_targets(torch.zeros((2, 3)), ((-1.0, 1.0),), gripper_indices=())


if __name__ == "__main__":
    unittest.main()
