from __future__ import annotations

import math
import unittest
import warnings
from types import SimpleNamespace

import torch
import warp as wp
from isaaclab.utils.warp.proxy_array import ProxyArray

from mrs_arena.adapters.navigation_observation import navigation_policy_observation


class _Scene:
    def __init__(self, robot) -> None:
        self.env_origins = torch.zeros((1, 3), dtype=torch.float32)
        self._entities = {"robot": robot}

    def __getitem__(self, name):
        return self._entities[name]


class NavigationObservationAdapterTest(unittest.TestCase):
    def test_policy_observation_converts_isaaclab_proxy_arrays_to_torch_views(self) -> None:
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
        env = SimpleNamespace(scene=_Scene(robot))
        asset_cfg = SimpleNamespace(name="robot")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            observation = navigation_policy_observation(
                env,
                target_xy=(2.0, 0.0),
                target_yaw_rad=0.0,
                asset_cfg=asset_cfg,
            )

        expected = torch.tensor([[0.0, -2.0, -1.0, 0.0, 1.0, 0.0, 0.2, 2.0]])
        self.assertTrue(torch.allclose(observation, expected, atol=1e-6, rtol=0.0))
        self.assertFalse(
            any(issubclass(item.category, DeprecationWarning) for item in caught),
            "navigation observations must use explicit Isaac Lab tensor views",
        )


if __name__ == "__main__":
    unittest.main()
