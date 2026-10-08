from __future__ import annotations

import unittest

try:
    from mrs_robot_lab.environments.learning.task_logic import (
        BoxTransportEvaluator,
        NavigationGoalEvaluator,
        TransportPhase,
    )
    TASK_LOGIC_IMPORT_ERROR = None
except ImportError as error:
    BoxTransportEvaluator = None
    NavigationGoalEvaluator = None
    TransportPhase = None
    TASK_LOGIC_IMPORT_ERROR = error


class NavigationGoalEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        if TASK_LOGIC_IMPORT_ERROR is not None:
            self.fail(f"task evaluator implementation is missing: {TASK_LOGIC_IMPORT_ERROR}")

    def test_success_requires_position_heading_and_continuous_hold(self) -> None:
        evaluator = NavigationGoalEvaluator(
            position_tolerance_m=0.15,
            heading_tolerance_rad=0.17453292519943295,
            hold_seconds=1.0,
        )

        self.assertFalse(evaluator.update((0.1, 0.0), 0.0, (0.0, 0.0), 0.0, 0.5))
        self.assertFalse(evaluator.update((0.0, 0.0), 0.3, (0.0, 0.0), 0.0, 0.6))
        self.assertFalse(evaluator.update((0.0, 0.0), 0.0, (0.0, 0.0), 0.0, 0.5))
        self.assertTrue(evaluator.update((0.0, 0.0), 0.0, (0.0, 0.0), 0.0, 0.5))

    def test_heading_error_wraps_across_pi_boundary(self) -> None:
        evaluator = NavigationGoalEvaluator(0.1, 0.05, 0.1)
        self.assertTrue(evaluator.update((0.0, 0.0), -3.13, (0.0, 0.0), 3.13, 0.1))


class BoxTransportEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        if TASK_LOGIC_IMPORT_ERROR is not None:
            self.fail(f"task evaluator implementation is missing: {TASK_LOGIC_IMPORT_ERROR}")
        self.evaluator = BoxTransportEvaluator(
            initial_position=(0.65, 0.0, 0.80),
            target_position=(0.90, 0.0, 0.80),
            position_tolerance_m=0.05,
            min_lift_height_m=0.10,
            min_carry_distance_m=0.20,
            stable_seconds=1.0,
            max_linear_speed_mps=0.02,
        )

    def step(self, position, *, speed=0.0, left=False, right=False, dt=0.5):
        return self.evaluator.update(
            box_position=position,
            box_linear_velocity=(speed, 0.0, 0.0),
            left_gripper_contact=left,
            right_gripper_contact=right,
            dt=dt,
        )

    def test_success_requires_bilateral_grasp_lift_carry_place_release_and_stability(self) -> None:
        self.assertEqual(self.step((0.65, 0.0, 0.80)), TransportPhase.APPROACH)
        self.assertEqual(self.step((0.65, 0.0, 0.80), left=True), TransportPhase.APPROACH)
        self.assertEqual(self.step((0.65, 0.0, 0.80), left=True, right=True), TransportPhase.GRASPED)
        self.assertEqual(self.step((0.65, 0.0, 0.91), left=True, right=True), TransportPhase.LIFTED)
        self.assertEqual(self.step((0.86, 0.0, 0.91), left=True, right=True), TransportPhase.CARRIED)
        self.assertEqual(self.step((0.90, 0.0, 0.80), left=True, right=True), TransportPhase.PLACED)
        self.assertEqual(self.step((0.90, 0.0, 0.80)), TransportPhase.RELEASED)
        self.assertEqual(self.step((0.90, 0.0, 0.80)), TransportPhase.SUCCESS)
        self.assertEqual(self.step((0.90, 0.0, 0.80)), TransportPhase.SUCCESS)

    def test_early_release_does_not_skip_lift_and_carry_requirements(self) -> None:
        self.step((0.65, 0.0, 0.80), left=True, right=True)
        self.assertEqual(self.step((0.65, 0.0, 0.80)), TransportPhase.GRASPED)
        self.assertNotEqual(self.evaluator.phase, TransportPhase.SUCCESS)

    def test_stability_timer_resets_when_box_moves_too_fast(self) -> None:
        for position, left, right in (
            ((0.65, 0.0, 0.80), True, True),
            ((0.65, 0.0, 0.91), True, True),
            ((0.86, 0.0, 0.91), True, True),
            ((0.90, 0.0, 0.80), True, True),
            ((0.90, 0.0, 0.80), False, False),
        ):
            self.step(position, left=left, right=right)
        self.assertEqual(self.step((0.90, 0.0, 0.80), speed=0.1), TransportPhase.RELEASED)
        self.assertEqual(self.step((0.90, 0.0, 0.80)), TransportPhase.RELEASED)
        self.assertEqual(self.step((0.90, 0.0, 0.80)), TransportPhase.SUCCESS)


if __name__ == "__main__":
    unittest.main()
