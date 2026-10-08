from __future__ import annotations

import unittest
import warnings
from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.utils.warp.proxy_array import ProxyArray

from mrs_robot_lab.environments.learning.dual_arm_box_task import DualArmBoxTaskEnvironment


class DualArmObservationProxyArrayTest(unittest.TestCase):
    def test_box_observation_uses_explicit_torch_views_for_root_state(self) -> None:
        box = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=ProxyArray(
                    wp.array([wp.vec3f(1.0, 2.0, 0.5)], dtype=wp.vec3f, device="cpu")
                ),
                root_quat_w=ProxyArray(
                    wp.array([wp.quatf(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device="cpu")
                ),
                root_lin_vel_w=ProxyArray(
                    wp.array([wp.vec3f(0.1, 0.2, 0.3)], dtype=wp.vec3f, device="cpu")
                ),
            )
        )
        task = SimpleNamespace(
            _robot=SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=torch.arange(16, dtype=torch.float32).reshape(1, 16),
                    joint_vel=-torch.arange(16, dtype=torch.float32).reshape(1, 16),
                )
            ),
            _action_joint_ids=slice(None),
            _box=box,
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
            _target_position_tensor=torch.tensor([2.0, 0.0, 1.0]),
            _left_contact=torch.tensor([True]),
            _right_contact=torch.tensor([False]),
        )

        prior_deprecation_state = ProxyArray._deprecation_warned
        ProxyArray._deprecation_warned = False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                observation = DualArmBoxTaskEnvironment._get_observations(task)["policy"]
        finally:
            ProxyArray._deprecation_warned = prior_deprecation_state

        expected = torch.cat(
            (
                torch.arange(16, dtype=torch.float32),
                -torch.arange(16, dtype=torch.float32),
                torch.tensor([1.0, 2.0, 0.5]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
                torch.tensor([0.1, 0.2, 0.3]),
                torch.tensor([1.0, -2.0, 0.5]),
                torch.tensor([1.0, 0.0]),
            )
        ).reshape(1, 47)
        self.assertTrue(torch.allclose(observation, expected, atol=1e-6, rtol=0.0))
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught),
            "box task observations must use explicit tensor views for all root-state fields",
        )


if __name__ == "__main__":
    unittest.main()
