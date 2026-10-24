"""CPU-only tests for the arena adapter over the bilateral transport criterion.

The criterion itself lives in the learning layer; these tests pin the arena-facing
contract: ``(success, phase)`` return order, the lab 0..6 phase ladder, and the
"both grippers must let go" release rule. Cross-backend equality is guarded by
``tests/integration/architecture/test_transport_criteria_consistency.py``.
"""

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

    def reset(self):
        # arena 的既有 reset 姿势：把 state 里的张量就地清零（它们别名到 lab 的缓冲区）。
        for value in self.state.values():
            value.zero_()

    def test_requires_bilateral_grasp_lift_carry_release_and_stability(self):
        assert not self.update((0.65, 0.0, 0.80), left=True, right=False)[0]
        assert not self.update((0.65, 0.0, 1.00), left=True, right=True)[0]
        assert not self.update((0.90, 0.0, 0.80), left=True, right=True)[0]
        assert not self.update((0.90, 0.0, 0.80), left=False, right=False, dt=0.5)[0]
        assert self.update((0.90, 0.0, 0.80), left=False, right=False, dt=0.5)[0]

    def test_never_grasped_box_cannot_succeed_at_target(self):
        results = [self.update((0.90, 0.0, 0.80), left=False, right=False) for _ in range(20)]
        assert not any(success.item() for success, _ in results)

    def test_box_moving_too_fast_does_not_accumulate_release_stability(self):
        self.update((0.65, 0.0, 0.80), left=True, right=True)
        self.update((0.65, 0.0, 1.00), left=True, right=True)
        self.update((0.90, 0.0, 0.80), left=True, right=True)
        assert not self.update((0.90, 0.0, 0.80), left=False, right=False, speed=0.1)[0]
        assert not self.update((0.90, 0.0, 0.80), left=False, right=False, dt=0.9)[0]

    def test_release_requires_both_grippers_to_let_go(self):
        self.update((0.65, 0.0, 0.80), left=True, right=True)
        self.update((0.65, 0.0, 1.00), left=True, right=True)
        self.update((0.90, 0.0, 0.80), left=True, right=True)

        results = [self.update((0.90, 0.0, 0.80), left=True, right=False, dt=0.5) for _ in range(5)]
        success, phase = results[-1]

        # 任意一只手还按着 → 没释放，稳定计时不累加（lab 的判据：双手都松）。
        assert not success.item()
        assert phase.item() == 4
        assert not any(step_success.item() for step_success, _ in results)

    def test_reports_the_lab_phase_ladder_and_reset_clears_episode_history(self):
        _, phase = self.update((0.65, 0.0, 0.80), left=False, right=False)
        assert phase.item() == 0
        _, phase = self.update((0.65, 0.0, 0.80), left=True, right=True)
        assert phase.item() == 1
        _, phase = self.update((0.65, 0.0, 1.00), left=True, right=True)
        assert phase.item() == 2
        _, phase = self.update((0.86, 0.0, 0.91), left=True, right=True)
        assert phase.item() == 3
        # 到目标且仍双手接触：已经 PLACED（lab 的 4），不是旧的 arena 3。
        _, phase = self.update((0.90, 0.0, 0.80), left=True, right=True)
        assert phase.item() == 4
        _, phase = self.update((0.90, 0.0, 0.80), left=False, right=False)
        assert phase.item() == 5
        success, phase = self.update((0.90, 0.0, 0.80), left=False, right=False, dt=1.0)
        assert phase.item() == 6
        assert success.item()

        self.reset()
        success, phase = self.update((0.90, 0.0, 0.80), left=False, right=False, dt=1.1)
        assert not success.item()
        assert phase.item() == 0

    def test_returns_success_then_phase_with_the_documented_dtypes(self):
        self.update((0.65, 0.0, 0.80), left=True, right=True)
        self.update((0.65, 0.0, 1.00), left=True, right=True)
        self.update((0.90, 0.0, 0.80), left=True, right=True)
        self.update((0.90, 0.0, 0.80), left=False, right=False)
        success, phase = self.update((0.90, 0.0, 0.80), left=False, right=False, dt=1.0)

        assert success.dtype is torch.bool
        assert phase.dtype is torch.long
        assert success.shape == phase.shape == (1,)
        assert success.tolist() == [True]
        assert phase.tolist() == [6]
