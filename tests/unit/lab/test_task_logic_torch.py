from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

try:
    from mrs_robot_lab.environments.learning.task_logic_torch import (
        BatchedBoxTransportEvaluator,
        contact_mask_from_force_matrix,
    )
    TORCH_LOGIC_IMPORT_ERROR = None
except ImportError as error:
    BatchedBoxTransportEvaluator = None
    contact_mask_from_force_matrix = None
    TORCH_LOGIC_IMPORT_ERROR = error


class BatchedBoxTransportEvaluatorTest(unittest.TestCase):
    def setUp(self) -> None:
        if TORCH_LOGIC_IMPORT_ERROR is not None:
            self.fail(f"batched box task evaluator is missing: {TORCH_LOGIC_IMPORT_ERROR}")
        self.evaluator = BatchedBoxTransportEvaluator(
            initial_position=(0.65, 0.0, 0.80),
            target_position=(0.90, 0.0, 0.80),
            num_envs=2,
            position_tolerance_m=0.05,
            min_lift_height_m=0.10,
            min_carry_distance_m=0.20,
            stable_seconds=1.0,
            max_linear_speed_mps=0.02,
            device="cpu",
        )

    def update(self, positions, *, left=(False, False), right=(False, False), speeds=(0.0, 0.0)):
        return self.evaluator.update(
            torch.tensor(positions),
            torch.tensor([[speed, 0.0, 0.0] for speed in speeds]),
            torch.tensor(left),
            torch.tensor(right),
            dt=0.5,
        )

    def test_two_environments_follow_independent_ordered_task_phases(self) -> None:
        phase, success = self.update([[0.65, 0.0, 0.80], [0.65, 0.0, 0.80]])
        self.assertEqual(phase.tolist(), [0, 0])
        phase, _ = self.update(
            [[0.65, 0.0, 0.80], [0.65, 0.0, 0.80]], left=(True, False), right=(True, False)
        )
        self.assertEqual(phase.tolist(), [1, 0])
        phase, _ = self.update(
            [[0.65, 0.0, 0.91], [0.65, 0.0, 0.91]], left=(True, True), right=(True, True)
        )
        self.assertEqual(phase.tolist(), [2, 2])
        phase, _ = self.update(
            [[0.86, 0.0, 0.91], [0.86, 0.0, 0.91]], left=(True, True), right=(True, True)
        )
        self.assertEqual(phase.tolist(), [3, 3])
        phase, _ = self.update(
            [[0.90, 0.0, 0.80], [0.90, 0.0, 0.80]], left=(True, True), right=(True, True)
        )
        self.assertEqual(phase.tolist(), [4, 4])
        phase, _ = self.update([[0.90, 0.0, 0.80], [0.90, 0.0, 0.80]])
        self.assertEqual(phase.tolist(), [5, 5])
        phase, success = self.update([[0.90, 0.0, 0.80], [0.90, 0.0, 0.80]])
        self.assertEqual(phase.tolist(), [6, 6])
        self.assertEqual(success.tolist(), [True, True])

    def test_reset_clears_only_requested_environment_progress(self) -> None:
        self.update(
            [[0.65, 0.0, 0.80], [0.65, 0.0, 0.80]], left=(True, True), right=(True, True)
        )
        self.evaluator.reset(torch.tensor([0]))

        phase, _ = self.update(
            [[0.65, 0.0, 0.80], [0.65, 0.0, 0.91]], left=(True, True), right=(True, True)
        )

        self.assertEqual(phase.tolist(), [1, 2])

    def test_contact_mask_uses_force_from_any_finger_against_the_box(self) -> None:
        forces = torch.tensor(
            [
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]]],
                [[[0.0, 0.0, 0.05], [0.0, 0.0, 0.0]]],
            ]
        )

        mask = contact_mask_from_force_matrix(forces, force_threshold=0.1)

        self.assertEqual(mask.tolist(), [True, False])

    def test_reset_can_update_randomized_initial_pose_for_selected_environments(self) -> None:
        self.update(
            [[0.65, 0.0, 0.80], [0.65, 0.0, 0.80]], left=(True, True), right=(True, True)
        )
        self.update(
            [[0.65, 0.0, 0.91], [0.65, 0.0, 0.91]], left=(True, True), right=(True, True)
        )
        try:
            self.evaluator.reset(
                torch.tensor([0]),
                initial_position=torch.tensor([[0.70, 0.0, 0.80]]),
            )
        except TypeError as error:
            self.fail(f"task evaluator cannot reset randomized initial states: {error}")

        phase, _ = self.update(
            [[0.86, 0.0, 0.91], [0.86, 0.0, 0.91]], left=(True, True), right=(True, True)
        )

        self.assertEqual(phase.tolist(), [2, 3])

    def test_contact_mask_accepts_isaac_lab_proxyarray_force_matrix(self) -> None:
        force_tensor = torch.tensor(
            [
                [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]]],
                [[[0.0, 0.0, 0.08], [0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.15], [0.0, 0.0, 0.12]]],
            ]
        )
        proxy_like_input = SimpleNamespace(torch=force_tensor)

        mask = contact_mask_from_force_matrix(proxy_like_input, force_threshold=0.1)

        self.assertEqual(mask.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
