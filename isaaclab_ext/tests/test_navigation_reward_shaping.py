from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from mrs_robot_lab.environments.learning.navigation_task import NavigationTaskEnvironment


def _environment_for_reward(
    *,
    position_xy: tuple[float, float],
    yaw: float,
    previous_distance: float,
    previous_heading_error: float,
    reward_config: dict[str, float],
    success: bool = False,
) -> NavigationTaskEnvironment:
    environment = object.__new__(NavigationTaskEnvironment)
    environment._robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[position_xy[0], position_xy[1], 0.0]]),
            root_quat_w=torch.tensor(
                [[0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]]
            ),
        )
    )
    environment.scene = SimpleNamespace(env_origins=torch.zeros((1, 3)))
    environment._target_position = torch.tensor([2.0, 0.0])
    environment._target_yaw_tensor = torch.zeros(1)
    environment._success_mask = torch.tensor([success])
    environment._previous_distance = torch.tensor([previous_distance])
    environment._previous_heading_error = torch.tensor([previous_heading_error])
    environment.cfg = SimpleNamespace(decimation=1, sim=SimpleNamespace(dt=0.1))
    environment._is_closed = True
    environment.task_spec = SimpleNamespace(reward_config=reward_config)
    return environment


class NavigationRewardShapingTest(unittest.TestCase):
    def test_progress_toward_goal_produces_positive_distance_shaping(self) -> None:
        environment = _environment_for_reward(
            position_xy=(1.5, 0.0),
            yaw=0.0,
            previous_distance=1.0,
            previous_heading_error=0.0,
            reward_config={
                "step_penalty": 0.0,
                "distance_reward_scale": 0.0,
                "distance_progress_reward_scale": 1.0,
                "heading_error_reward_scale": 0.0,
                "heading_progress_reward_scale": 0.0,
                "success_reward": 0.0,
            },
        )

        reward = environment._get_rewards()

        self.assertAlmostEqual(float(reward[0]), 5.0, places=5)
        self.assertAlmostEqual(float(environment._previous_distance[0]), 0.5, places=5)

    def test_moving_away_from_goal_produces_negative_distance_shaping(self) -> None:
        environment = _environment_for_reward(
            position_xy=(1.0, 0.0),
            yaw=0.0,
            previous_distance=0.5,
            previous_heading_error=0.0,
            reward_config={
                "step_penalty": 0.0,
                "distance_reward_scale": 0.0,
                "distance_progress_reward_scale": 1.0,
                "heading_error_reward_scale": 0.0,
                "heading_progress_reward_scale": 0.0,
                "success_reward": 0.0,
            },
        )

        reward = environment._get_rewards()

        self.assertAlmostEqual(float(reward[0]), -5.0, places=5)

    def test_heading_error_is_penalized_and_heading_progress_is_rewarded(self) -> None:
        environment = _environment_for_reward(
            position_xy=(2.0, 0.0),
            yaw=0.0,
            previous_distance=0.0,
            previous_heading_error=math.pi / 2.0,
            reward_config={
                "step_penalty": 0.0,
                "distance_reward_scale": 0.0,
                "distance_progress_reward_scale": 0.0,
                "heading_error_reward_scale": 0.5,
                "heading_progress_reward_scale": 0.1,
                "success_reward": 0.0,
            },
        )

        reward = environment._get_rewards()

        self.assertAlmostEqual(float(reward[0]), math.pi / 2.0, places=5)
        self.assertAlmostEqual(float(environment._previous_heading_error[0]), 0.0, places=5)

    def test_heading_error_is_penalized_at_fixed_position(self) -> None:
        environment = _environment_for_reward(
            position_xy=(2.0, 0.0),
            yaw=math.pi / 2.0,
            previous_distance=0.0,
            previous_heading_error=math.pi / 2.0,
            reward_config={
                "step_penalty": 0.0,
                "distance_reward_scale": 0.0,
                "distance_progress_reward_scale": 0.0,
                "heading_error_reward_scale": 0.5,
                "heading_progress_reward_scale": 0.0,
                "success_reward": 0.0,
            },
        )

        reward = environment._get_rewards()

        self.assertAlmostEqual(float(reward[0]), -math.pi / 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
