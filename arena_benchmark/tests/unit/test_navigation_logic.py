from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
import unittest

import torch

from openflex_isaac_contract.task_spec import load_task_spec


SIM_ROOT = Path(__file__).resolve().parents[3]


class NavigationLogicTest(unittest.TestCase):
    def setUp(self) -> None:
        module_name = "mrs_arena.tasks.navigation_logic"
        if importlib.util.find_spec(module_name) is None:
            self.fail(f"{module_name} must implement the shared Arena navigation behavior")
        self.logic = importlib.import_module(module_name)

    def test_shared_task_yaml_produces_the_arena_goal_contract(self) -> None:
        task = load_task_spec(SIM_ROOT / "sim_runtime/config/tasks/navigation_to_goal.yaml")

        parameters = self.logic.navigation_parameters_from_task(task)

        self.assertEqual(parameters.target_xy, (2.0, 0.0))
        self.assertEqual(parameters.target_yaw_rad, 0.0)
        self.assertEqual(parameters.position_tolerance_m, 0.15)
        self.assertAlmostEqual(parameters.heading_tolerance_rad, math.radians(10.0))
        self.assertEqual(parameters.hold_seconds, 1.0)
        self.assertEqual(parameters.episode_length_s, 30.0)

    def test_goal_mask_uses_xy_distance_and_wrapped_heading_error(self) -> None:
        task = load_task_spec(SIM_ROOT / "sim_runtime/config/tasks/navigation_to_goal.yaml")
        parameters = self.logic.navigation_parameters_from_task(task)

        result = self.logic.navigation_goal_mask(
            torch.tensor([[2.0, 0.0], [2.1, 0.0], [2.0, 0.0]]),
            torch.tensor([2.0 * math.pi, 0.0, math.radians(15.0)]),
            parameters,
        )

        self.assertEqual(result.tolist(), [True, True, False])

    def test_xyzw_yaw_conversion_matches_the_arena_pose_contract(self) -> None:
        converter = getattr(self.logic, "quaternion_xyzw_to_yaw", None)
        self.assertIsNotNone(converter, "navigation task must interpret Arena poses as xyzw")

        yaw = converter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)],
                ]
            )
        )

        self.assertTrue(torch.allclose(yaw, torch.tensor([0.0, math.pi / 2.0]), atol=1e-6))

    def test_navigation_observation_keeps_the_trained_feature_order(self) -> None:
        observation_fn = getattr(self.logic, "navigation_observation_tensor", None)
        self.assertIsNotNone(observation_fn, "Arena must produce the 8-feature PPO navigation observation")

        observation = observation_fn(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([math.pi / 2.0]),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([0.2]),
            target_xy=(2.0, 0.0),
            target_yaw_rad=0.0,
        )

        expected = torch.tensor([[0.0, -2.0, -1.0, 0.0, 1.0, 0.0, 0.2, 2.0]])
        self.assertEqual(tuple(observation.shape), (1, 8))
        self.assertTrue(torch.allclose(observation, expected, atol=1e-6, rtol=0.0))

    def test_success_hold_accumulates_inside_goal_and_resets_outside(self) -> None:
        held = torch.tensor([0.0, 0.75])

        held, success = self.logic.advance_success_hold(
            held,
            torch.tensor([True, False]),
            dt=0.25,
            required_seconds=1.0,
        )

        self.assertEqual(held.tolist(), [0.25, 0.0])
        self.assertEqual(success.tolist(), [False, False])

        held, success = self.logic.advance_success_hold(
            held,
            torch.tensor([True, True]),
            dt=0.75,
            required_seconds=1.0,
        )

        self.assertEqual(held.tolist(), [1.0, 0.75])
        self.assertEqual(success.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
