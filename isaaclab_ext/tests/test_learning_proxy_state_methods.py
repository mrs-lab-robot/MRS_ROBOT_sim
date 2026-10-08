from __future__ import annotations

import math
import unittest
import warnings
from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.utils.warp.proxy_array import ProxyArray

from mrs_robot_lab.environments.learning.dual_arm_box_task import DualArmBoxTaskEnvironment
from mrs_robot_lab.environments.learning.navigation_task import NavigationTaskEnvironment


def _proxy_root_state(
    *,
    position=(0.0, 0.0, 0.0),
    quaternion=(0.0, 0.0, 0.0, 1.0),
    linear_velocity=(0.0, 0.0, 0.0),
    angular_velocity=(0.0, 0.0, 0.0),
):
    return SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=ProxyArray(wp.array([wp.vec3f(*position)], dtype=wp.vec3f, device="cpu")),
            root_quat_w=ProxyArray(wp.array([wp.quatf(*quaternion)], dtype=wp.quatf, device="cpu")),
            root_lin_vel_w=ProxyArray(wp.array([wp.vec3f(*linear_velocity)], dtype=wp.vec3f, device="cpu")),
            root_ang_vel_w=ProxyArray(wp.array([wp.vec3f(*angular_velocity)], dtype=wp.vec3f, device="cpu")),
        )
    )


class LearningProxyStateMethodsTest(unittest.TestCase):
    def _call_without_implicit_proxy_conversion(self, callback):
        prior_deprecation_state = ProxyArray._deprecation_warned
        ProxyArray._deprecation_warned = False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = callback()
        finally:
            ProxyArray._deprecation_warned = prior_deprecation_state
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught),
            "task methods must explicitly request Isaac Lab ProxyArray .torch views",
        )
        return result

    def test_navigation_reward_reads_proxy_position_and_orientation_explicitly(self) -> None:
        task = SimpleNamespace(
            _robot=_proxy_root_state(),
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
            _target_position=torch.tensor([2.0, 0.0]),
            _target_yaw_tensor=torch.tensor([0.0]),
            _previous_distance=torch.tensor([3.0]),
            _previous_heading_error=torch.tensor([0.0]),
            _success_mask=torch.tensor([False]),
            step_dt=0.1,
            task_spec=SimpleNamespace(
                reward_config={
                    "step_penalty": -0.01,
                    "distance_reward_scale": 0.1,
                    "distance_progress_reward_scale": 0.5,
                    "heading_error_reward_scale": 0.3,
                    "heading_progress_reward_scale": 0.25,
                }
            ),
        )

        reward = self._call_without_implicit_proxy_conversion(
            lambda: NavigationTaskEnvironment._get_rewards(task)
        )

        self.assertTrue(torch.allclose(reward, torch.tensor([4.79]), atol=1e-5, rtol=0.0))

    def test_navigation_success_termination_reads_proxy_pose_explicitly(self) -> None:
        task = SimpleNamespace(
            _robot=_proxy_root_state(position=(2.0, 0.0, 0.0)),
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
            _target_position=torch.tensor([2.0, 0.0]),
            _target_yaw_tensor=torch.tensor([0.0]),
            _position_tolerance=0.15,
            _heading_tolerance_tensor=torch.tensor([math.radians(10.0)]),
            _success_hold_seconds=torch.zeros(1),
            _success_mask=torch.zeros(1, dtype=torch.bool),
            step_dt=0.1,
            _hold_seconds=0.1,
            episode_length_buf=torch.tensor([0]),
            max_episode_length=100,
        )

        terminated, truncated = self._call_without_implicit_proxy_conversion(
            lambda: NavigationTaskEnvironment._get_dones(task)
        )

        self.assertEqual(terminated.tolist(), [True])
        self.assertEqual(truncated.tolist(), [False])

    def test_dual_arm_reward_reads_proxy_box_position_explicitly(self) -> None:
        task = SimpleNamespace(
            _box=_proxy_root_state(position=(1.0, 2.0, 0.5)),
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
            _target_position_tensor=torch.tensor([2.0, 0.0, 1.0]),
            _last_phase=torch.tensor([1]),
            _previous_phase=torch.tensor([0]),
            _success_mask=torch.tensor([True]),
            task_spec=SimpleNamespace(
                reward_config={
                    "step_penalty": -0.1,
                    "distance_reward_scale": 0.2,
                    "phase_progress_reward": 2.0,
                    "success_reward": 10.0,
                }
            ),
        )

        reward = self._call_without_implicit_proxy_conversion(
            lambda: DualArmBoxTaskEnvironment._get_rewards(task)
        )

        expected = torch.tensor([-0.1 - 0.2 * math.sqrt(5.25) + 2.0 + 10.0])
        self.assertTrue(torch.allclose(reward, expected, atol=1e-6, rtol=0.0))

    def test_dual_arm_termination_reads_proxy_box_pose_and_velocity_explicitly(self) -> None:
        def update(box_position, box_velocity, left_contact, right_contact, *, dt):
            return torch.tensor([2]), torch.tensor([True])

        task = SimpleNamespace(
            _box=_proxy_root_state(position=(1.0, 2.0, 0.5), linear_velocity=(0.1, 0.0, 0.0)),
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
            _side_contact_mask=lambda side: torch.tensor([side == "left"]),
            _task_logic=SimpleNamespace(update=update),
            step_dt=0.1,
            _previous_phase=torch.tensor([0]),
            _last_phase=torch.tensor([0]),
            _success_mask=torch.tensor([False]),
            extras={},
            episode_length_buf=torch.tensor([0]),
            max_episode_length=100,
        )

        terminated, truncated = self._call_without_implicit_proxy_conversion(
            lambda: DualArmBoxTaskEnvironment._get_dones(task)
        )

        self.assertEqual(terminated.tolist(), [True])
        self.assertEqual(truncated.tolist(), [False])
        self.assertEqual(task.extras["task_phase"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
