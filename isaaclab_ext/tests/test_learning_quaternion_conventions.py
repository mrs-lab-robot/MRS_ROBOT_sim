from __future__ import annotations

import math
import unittest
import warnings
from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.utils.warp.proxy_array import ProxyArray

from mrs_robot_lab.environments.learning.dual_arm_box_task import (
    _as_xyzw,
    _yaw_tensor_to_xyzw,
)
from mrs_robot_lab.environments.learning.navigation_task import (
    NavigationTaskEnvironment,
    _quat_xyzw_to_yaw,
    _yaw_to_xyzw,
)


class LearningQuaternionConventionTest(unittest.TestCase):
    def test_navigation_root_pose_writer_uses_xyzw_identity(self) -> None:
        self.assertEqual(_yaw_to_xyzw(0.0), (0.0, 0.0, 0.0, 1.0))

    def test_navigation_root_quaternion_reader_decodes_xyzw_yaw(self) -> None:
        half_yaw = math.pi / 4.0
        quaternion_xyzw = torch.tensor(
            [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]
        )

        yaw = _quat_xyzw_to_yaw(quaternion_xyzw)

        self.assertAlmostEqual(float(yaw), math.pi / 2.0, places=6)

    def test_navigation_root_quaternion_reader_uses_the_explicit_proxy_torch_view(self) -> None:
        quaternion = ProxyArray(
            wp.array(
                [wp.quatf(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))],
                dtype=wp.quatf,
                device="cpu",
            )
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yaw = _quat_xyzw_to_yaw(quaternion)

        self.assertEqual(tuple(yaw.shape), (1,))
        self.assertAlmostEqual(float(yaw[0]), math.pi / 2.0, places=6)
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught),
            "navigation must not rely on Isaac Lab's deprecated implicit ProxyArray conversion",
        )

    def test_navigation_observation_reads_proxy_root_state_without_implicit_conversion(self) -> None:
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=ProxyArray(
                    wp.array([wp.vec3f(0.0, 0.0, 0.0)], dtype=wp.vec3f, device="cpu")
                ),
                root_quat_w=ProxyArray(
                    wp.array(
                        [wp.quatf(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))],
                        dtype=wp.quatf,
                        device="cpu",
                    )
                ),
                root_lin_vel_w=ProxyArray(
                    wp.array([wp.vec3f(0.0, 1.0, 0.0)], dtype=wp.vec3f, device="cpu")
                ),
                root_ang_vel_w=ProxyArray(
                    wp.array([wp.vec3f(0.0, 0.0, 0.2)], dtype=wp.vec3f, device="cpu")
                ),
            )
        )
        task = SimpleNamespace(
            _robot=robot,
            scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
            _target_position=torch.tensor([2.0, 0.0]),
            _target_yaw_tensor=torch.tensor([0.0]),
        )

        prior_deprecation_state = ProxyArray._deprecation_warned
        ProxyArray._deprecation_warned = False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                observation = NavigationTaskEnvironment._get_observations(task)["policy"]
        finally:
            ProxyArray._deprecation_warned = prior_deprecation_state

        expected = torch.tensor([[0.0, -2.0, -1.0, 0.0, 1.0, 0.0, 0.2, 2.0]])
        self.assertTrue(torch.allclose(observation, expected, atol=1e-6, rtol=0.0))
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught),
            "navigation must use explicit tensor views for every root-state field",
        )

    def test_box_usd_spawn_preserves_xyzw_orientation(self) -> None:
        half_yaw = math.pi / 4.0
        orientation_xyzw = (
            0.0,
            0.0,
            math.sin(half_yaw),
            math.cos(half_yaw),
        )

        self.assertEqual(_as_xyzw(orientation_xyzw), orientation_xyzw)

    def test_box_reset_pose_writer_uses_xyzw_yaw_quaternions(self) -> None:
        yaws = torch.tensor([0.0, math.pi / 2.0])
        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
            ]
        )

        torch.testing.assert_close(_yaw_tensor_to_xyzw(yaws), expected)


if __name__ == "__main__":
    unittest.main()
