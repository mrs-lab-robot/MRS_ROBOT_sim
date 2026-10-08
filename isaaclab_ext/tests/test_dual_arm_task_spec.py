"""测试双臂搬箱任务环境的TaskSpec加载和配置解析"""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
import yaml

from openflex_isaac_contract.task_spec import load_task_spec


class DualArmBoxTaskSpecTest(unittest.TestCase):
    """测试双臂搬箱任务的TaskSpec加载和解析（不依赖Isaac Lab）"""

    def test_task_spec_loads_dual_arm_box_configuration(self):
        """TaskSpec能正确加载双臂搬箱任务配置"""
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
            loaded_spec = load_task_spec(task_spec_path)
            self.assertEqual(loaded_spec.task_id, "dual_arm_box_pickup")
            self.assertEqual(loaded_spec.episode_length_s, 30.0)
            self.assertEqual(len(loaded_spec.success_criteria), 1)
            self.assertEqual(loaded_spec.success_criteria[0].criterion_type, "box_at_target")
            self.assertEqual(loaded_spec.success_criteria[0].tolerance, 0.05)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_task_spec_extracts_target_position_from_success_criteria(self):
        """能从success_criteria提取目标位置"""
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": 10.0,
            "success_criteria": [
                {
                    "criterion_type": "box_at_target",
                    "params": {"target_position": [1.5, -0.5, 0.8], "box_id": "box_0"},
                    "tolerance": 0.1,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            loaded_spec = load_task_spec(task_spec_path)
            criterion = loaded_spec.success_criteria[0]
            target_pos = criterion.params["target_position"]
            self.assertEqual(target_pos, [1.5, -0.5, 0.8])
            self.assertEqual(criterion.tolerance, 0.1)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_task_spec_parses_reward_configuration(self):
        """能解析奖励配置"""
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
            "reward_config": {
                "success_reward": 200.0,
                "step_penalty": -0.05,
                "distance_reward_scale": 0.1,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            loaded_spec = load_task_spec(task_spec_path)
            self.assertIsNotNone(loaded_spec.reward_config)
            self.assertEqual(loaded_spec.reward_config["success_reward"], 200.0)
            self.assertEqual(loaded_spec.reward_config["step_penalty"], -0.05)
            self.assertEqual(loaded_spec.reward_config["distance_reward_scale"], 0.1)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_task_spec_handles_missing_reward_config(self):
        """reward_config可选，默认为None"""
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
            loaded_spec = load_task_spec(task_spec_path)
            # reward_config应该为None或者环境应该能处理缺失情况
            self.assertIsNone(loaded_spec.reward_config)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_task_spec_validates_episode_length_positive(self):
        """episode_length_s必须为正数"""
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": -1.0,  # 非法：负数
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
            with self.assertRaises(ValueError):
                load_task_spec(task_spec_path)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)

    def test_task_spec_requires_non_empty_success_criteria(self):
        """success_criteria不能为空列表"""
        task_spec = {
            "spec_version": "v1",
            "task_id": "test_task",
            "description": "test",
            "scene_id": "test_scene",
            "episode_length_s": 10.0,
            "success_criteria": [],  # 非法：空列表
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(task_spec, f)
            task_spec_path = f.name

        try:
            with self.assertRaises(ValueError):
                load_task_spec(task_spec_path)
        finally:
            Path(task_spec_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
