"""CPU-only tests for bilateral transport progress and release criteria."""

from __future__ import annotations

import torch

from mrs_arena.tasks.transport_logic import (
    create_transport_progress_state,
    update_transport_progress,
)


class TestBoxTransportLogic:
    def setup_method(self):
        self.state = create_transport_progress_state(1, device="cpu")
        self.initial = torch.tensor([[0.65, 0.0, 0.80]])
        self.target = (0.90, 0.0, 0.80)

    def update(self, position, *, left, right, speed=0.0, dt=0.1):
        return update_transport_progress(
            self.state,
            box_position=torch.tensor([position], dtype=torch.float32),
            box_linear_speed=torch.tensor([speed], dtype=torch.float32),
            initial_position=self.initial,
            target_position=self.target,
            left_contact=torch.tensor([left]),
            right_contact=torch.tensor([right]),
            dt=dt,
            min_lift_height_m=0.10,
            min_carry_distance_m=0.20,
            position_tolerance_m=0.05,
            max_linear_speed_mps=0.02,
            stable_seconds=1.0,
        )

    def test_requires_bilateral_grasp_lift_carry_release_and_stability(self):
        assert not self.update((0.65, 0.0, 0.80), left=True, right=False)[0]
        assert not self.update((0.65, 0.0, 1.00), left=True, right=True)[0]
        assert not self.update((0.90, 0.0, 0.80), left=True, right=True)[0]
        assert not self.update((0.90, 0.0, 0.80), left=False, right=False, dt=0.5)[0]
        assert self.update((0.90, 0.0, 0.80), left=False, right=False, dt=0.5)[0]

    def test_never_grasped_box_cannot_succeed_at_target(self):
        for _ in range(20):
            success, _ = self.update((0.90, 0.0, 0.80), left=False, right=False)
        assert not success

    def test_box_moving_too_fast_does_not_accumulate_release_stability(self):
        self.update((0.65, 0.0, 0.80), left=True, right=True)
        self.update((0.65, 0.0, 1.00), left=True, right=True)
        self.update((0.90, 0.0, 0.80), left=True, right=True)
        assert not self.update((0.90, 0.0, 0.80), left=False, right=False, speed=0.1)[0]
        assert not self.update((0.90, 0.0, 0.80), left=False, right=False, dt=0.9)[0]

    def test_reports_progress_phase_and_reset_clears_episode_history(self):
        _, phase = self.update((0.65, 0.0, 0.80), left=True, right=True)
        assert phase.item() == 1
        _, phase = self.update((0.65, 0.0, 1.00), left=True, right=True)
        assert phase.item() == 2
        _, phase = self.update((0.90, 0.0, 0.80), left=True, right=True)
        assert phase.item() == 3

        for value in self.state.values():
            value.zero_()
        success, phase = self.update((0.90, 0.0, 0.80), left=False, right=False, dt=1.1)
        assert not success.item()
        assert phase.item() == 0
