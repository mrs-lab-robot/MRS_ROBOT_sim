from __future__ import annotations

import importlib
import importlib.util
import unittest

import torch


class NavigationPolicyAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        module_name = "mrs_arena.policies.navigation"
        if importlib.util.find_spec(module_name) is None:
            self.fail(f"{module_name} must adapt normalized PPO actions to Arena SI actions")
        self.adapter = importlib.import_module(module_name)

    def test_policy_actions_become_contract_limited_body_twist(self) -> None:
        actions = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.0]])

        twist = self.adapter.policy_actions_to_body_twist(actions)

        expected = torch.tensor(
            [
                [-0.8, 0.0, 2.320037210795224],
                [0.8, -0.8, 0.0],
            ]
        )
        self.assertTrue(torch.allclose(twist, expected, atol=1e-6, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
