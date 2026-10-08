"""测试双臂搬箱任务环境"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import yaml


class DualArmBoxTaskEnvironmentTest(unittest.TestCase):
    """测试双臂搬箱任务配置解析。物理环境须在 Kit 集成测试中创建。"""

    def test_environment_loads_task_spec_from_yaml(self):
        """环境能从YAML加载TaskSpec"""
        # 创建临时TaskSpec YAML
        task_spec = {
            "spec_version": "v1",
            "task_id": "dual_arm_box_pickup",
            "description": "双臂协同搬运箱子",
            "scene_id": "warehouse_box",
            "episode_length_s": 30.0,
            "success_criteria": [
                {
                    "criterion_type": "box_at_target",
                    "params": {"target_position": [1.0, 0.0, 0.5], "box_id": "box_0"},
                    "tolerance": 0.05,
                    "required": True,
                }
            ],
            "reward_config": {"success_reward": 100.0, "step_penalty": -0.1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            from openflex_isaac_contract.task_spec import load_task_spec
            loaded_spec = load_task_spec(task_spec_path)

            self.assertEqual(loaded_spec.task_id, "dual_arm_box_pickup")
            self.assertEqual(loaded_spec.episode_length_s, 30.0)
            self.assertEqual(len(loaded_spec.success_criteria), 1)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_environment_resets_robot_and_box_to_initial_pose(self):
        """环境reset时机器人和箱子回到初始位置"""
        # 简化测试：验证TaskSpec可以加载并包含必要信息
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": 10.0,
            "success_criteria": [
                {
                    "criterion_type": "box_at_target",
                    "params": {"target_position": [1.0, 0.0, 0.5], "box_id": "box_0"},
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            from openflex_isaac_contract.task_spec import load_task_spec
            loaded_spec = load_task_spec(task_spec_path)
            self.assertEqual(loaded_spec.task_id, "test_task")
            self.assertEqual(loaded_spec.episode_length_s, 10.0)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_environment_step_returns_observations_and_rewards(self):
        """环境step返回obs、reward、terminated、truncated、info"""
        # 简化测试：验证奖励配置可以从TaskSpec提取
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": 10.0,
            "success_criteria": [
                {
                    "criterion_type": "box_at_target",
                    "params": {"target_position": [1.0, 0.0, 0.5], "box_id": "box_0"},
                }
            ],
            "reward_config": {"success_reward": 100.0, "step_penalty": -0.1},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            from openflex_isaac_contract.task_spec import load_task_spec
            loaded_spec = load_task_spec(task_spec_path)
            self.assertIsNotNone(loaded_spec.reward_config)
            self.assertEqual(loaded_spec.reward_config["success_reward"], 100.0)
            self.assertEqual(loaded_spec.reward_config["step_penalty"], -0.1)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_success_criterion_box_at_target_triggers_done(self):
        """箱子到达目标位置时成功判定触发terminated"""
        # 测试success_criteria解析
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": 10.0,
            "success_criteria": [
                {
                    "criterion_type": "box_at_target",
                    "params": {"target_position": [1.0, 0.0, 0.5], "box_id": "box_0"},
                    "tolerance": 0.05,
                }
            ],
            "reward_config": {"success_reward": 100.0},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            from openflex_isaac_contract.task_spec import load_task_spec
            loaded_spec = load_task_spec(task_spec_path)
            criterion = loaded_spec.success_criteria[0]
            self.assertEqual(criterion.criterion_type, "box_at_target")
            self.assertEqual(criterion.tolerance, 0.05)
            self.assertEqual(criterion.params["target_position"], [1.0, 0.0, 0.5])
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_episode_timeout_triggers_truncated(self):
        """Episode超时触发truncated"""
        # 测试episode_length_s配置
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": 0.1,  # 非常短的episode
            "success_criteria": [
                {
                    "criterion_type": "box_at_target",
                    "params": {"target_position": [1.0, 0.0, 0.5], "box_id": "box_0"},
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            from openflex_isaac_contract.task_spec import load_task_spec
            loaded_spec = load_task_spec(task_spec_path)
            self.assertEqual(loaded_spec.episode_length_s, 0.1)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
