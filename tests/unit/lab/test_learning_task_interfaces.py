from __future__ import annotations

import unittest

try:
    from mrs_robot_lab.environments.learning.navigation_task import NavigationTaskEnvCfg
    from mrs_robot_lab.environments.learning.dual_arm_box_task import DualArmBoxTaskEnvCfg
    TASK_IMPORT_ERROR = None
except ImportError as error:
    NavigationTaskEnvCfg = None
    DualArmBoxTaskEnvCfg = None
    TASK_IMPORT_ERROR = error


class LearningTaskInterfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        if TASK_IMPORT_ERROR is not None:
            self.fail(f"baseline Isaac Lab task adapter cannot be imported: {TASK_IMPORT_ERROR}")

    def test_navigation_observation_and_action_shapes_match_documented_contract(self) -> None:
        config = NavigationTaskEnvCfg()

        self.assertEqual(config.num_observations, 8)
        self.assertEqual(config.num_actions, 3)
        self.assertEqual(config.observation_space, 8, "DirectRLEnvCfg observation_space must match documented dimension")
        self.assertEqual(config.action_space, 3, "DirectRLEnvCfg action_space must match documented dimension")

    def test_dual_arm_action_is_sixteen_named_arm_and_gripper_commands(self) -> None:
        config = DualArmBoxTaskEnvCfg()

        self.assertEqual(config.num_actions, 16)
        self.assertEqual(config.num_observations, 47)
        self.assertEqual(config.observation_space, 47, "DirectRLEnvCfg observation_space must match documented dimension")
        self.assertEqual(config.action_space, 16, "DirectRLEnvCfg action_space must match documented dimension")


if __name__ == "__main__":
    unittest.main()
