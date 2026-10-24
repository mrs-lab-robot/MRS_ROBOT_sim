from __future__ import annotations

import unittest

import torch

from mrs_robot_lab.assets.robot_interface import SWERVE_CONFIG
from mrs_robot_lab.controllers.swerve import compute_swerve_targets

try:
    from mrs_robot_lab.controllers.swerve_tensor import (
        compute_swerve_targets_tensor,
        normalized_to_body_twist,
    )
    SWERVE_TENSOR_IMPORT_ERROR = None
except ImportError as error:
    compute_swerve_targets_tensor = None
    normalized_to_body_twist = None
    SWERVE_TENSOR_IMPORT_ERROR = error


class SwerveTensorAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        if SWERVE_TENSOR_IMPORT_ERROR is not None:
            self.fail(f"batched swerve action adapter is missing: {SWERVE_TENSOR_IMPORT_ERROR}")

    def test_normalized_policy_actions_map_to_contract_si_limits(self) -> None:
        actions = torch.tensor([[1.0, -1.0, 0.5], [0.0, 0.0, 0.0]], dtype=torch.float32)

        twist = normalized_to_body_twist(actions, SWERVE_CONFIG)

        self.assertTrue(torch.allclose(twist[0], torch.tensor([0.8, -0.8, 1.1600186])))
        self.assertTrue(torch.allclose(twist[1], torch.zeros(3)))

    def test_tensor_targets_match_scalar_contract_kinematics(self) -> None:
        twist = torch.tensor([[0.3, -0.1, 0.2], [0.0, 0.0, 0.0]], dtype=torch.float32)
        previous = torch.zeros((2, 4), dtype=torch.float32)

        steering, wheel_linear_speeds, processed = compute_swerve_targets_tensor(
            twist, SWERVE_CONFIG, previous, dt=10.0
        )

        scalar = compute_swerve_targets((0.3, -0.1, 0.2), SWERVE_CONFIG)
        self.assertTrue(torch.allclose(processed[0], torch.tensor(scalar.applied_twist), atol=1e-6))
        self.assertTrue(torch.allclose(steering[0], torch.tensor(scalar.steering_angles), atol=1e-6))
        expected_linear = torch.tensor(scalar.wheel_angular_velocities) * SWERVE_CONFIG.wheel_radius
        self.assertTrue(torch.allclose(wheel_linear_speeds[0], expected_linear, atol=1e-6))
        self.assertTrue(torch.allclose(wheel_linear_speeds[1], torch.zeros(4)))

    def test_wheel_acceleration_is_limited_in_physical_linear_speed(self) -> None:
        twist = torch.tensor([[0.8, 0.0, 0.0]], dtype=torch.float32)
        previous = torch.zeros((1, 4), dtype=torch.float32)

        _, wheel_speeds, _ = compute_swerve_targets_tensor(
            twist, SWERVE_CONFIG, previous, dt=0.1
        )

        self.assertLessEqual(
            float(wheel_speeds.abs().max()),
            SWERVE_CONFIG.wheel_accel_limit * 0.1 + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
