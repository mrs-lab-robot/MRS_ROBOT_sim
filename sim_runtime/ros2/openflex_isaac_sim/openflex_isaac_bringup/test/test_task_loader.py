"""测试TaskSpec加载器：YAML → 运行时任务配置实例化

遵循 RED-GREEN 测试原则。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml


class TestTaskLoader(unittest.TestCase):
    """TaskLoader 核心功能测试"""

    def test_task_loader_initializes(self):
        """测试TaskLoader可以初始化"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        loader = TaskLoader()
        self.assertIsNotNone(loader)

    def test_load_task_from_yaml(self):
        """测试从YAML加载TaskSpec并返回任务配置"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        yaml_content = {
            "spec_version": "1.0",
            "task_id": "test_navigation",
            "description": "测试导航任务",
            "scene_id": "test_scene",
            "episode_length_s": 30.0,
            "success_criteria": [
                {
                    "criterion_type": "base_at_target",
                    "params": {"target_position": [2.0, 1.0, 0.0]},
                    "tolerance": 0.1,
                    "required": True,
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            loader = TaskLoader()
            result = loader.load_task(yaml_path)

            self.assertTrue(result["success"])
            self.assertEqual(result["task_id"], "test_navigation")
            self.assertEqual(result["scene_id"], "test_scene")
            self.assertEqual(result["episode_length_s"], 30.0)
            self.assertEqual(len(result["success_criteria"]), 1)
        finally:
            yaml_path.unlink(missing_ok=True)

    def test_load_task_validates_task_id(self):
        """测试加载任务时验证task_id非空"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        yaml_content = {
            "spec_version": "1.0",
            "task_id": "",
            "description": "无效任务",
            "scene_id": "test_scene",
            "episode_length_s": 30.0,
            "success_criteria": [
                {
                    "criterion_type": "test",
                    "params": {},
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            loader = TaskLoader()
            result = loader.load_task(yaml_path)

            self.assertFalse(result["success"])
            self.assertIn("error", result)
            self.assertIn("task_id", result["error"].lower())
        finally:
            yaml_path.unlink(missing_ok=True)

    def test_load_task_handles_file_not_found(self):
        """测试加载不存在的任务文件时返回错误"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        loader = TaskLoader()
        result = loader.load_task("/nonexistent/task.yaml")

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    def test_load_task_extracts_action_spec(self):
        """测试加载任务时提取action_spec配置"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        yaml_content = {
            "spec_version": "1.0",
            "task_id": "test_with_action",
            "description": "测试带动作规范的任务",
            "scene_id": "test_scene",
            "episode_length_s": 45.0,
            "success_criteria": [
                {
                    "criterion_type": "reach_goal",
                    "params": {"goal": [1.0, 2.0, 0.0]},
                }
            ],
            "action_spec": {
                "name": "bilateral_arm_joint_targets",
                "units": "rad",
                "normalized_range": [-1.0, 1.0],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            loader = TaskLoader()
            result = loader.load_task(yaml_path)

            self.assertTrue(result["success"])
            self.assertIn("action_spec", result)
            self.assertEqual(result["action_spec"]["name"], "bilateral_arm_joint_targets")
            self.assertEqual(result["action_spec"]["units"], "rad")
        finally:
            yaml_path.unlink(missing_ok=True)

    def test_load_task_extracts_initial_state(self):
        """测试加载任务时提取initial_state配置"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        yaml_content = {
            "spec_version": "1.0",
            "task_id": "test_with_initial_state",
            "description": "测试带初始状态的任务",
            "scene_id": "test_scene",
            "episode_length_s": 40.0,
            "success_criteria": [
                {
                    "criterion_type": "test_criterion",
                    "params": {},
                }
            ],
            "initial_state": {
                "robot_pose": {
                    "position": [0.0, 0.0, 0.0],
                    "yaw_rad": 0.0,
                },
                "box_pose": {
                    "position": [0.65, 0.0, 0.80],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            loader = TaskLoader()
            result = loader.load_task(yaml_path)

            self.assertTrue(result["success"])
            self.assertIn("initial_state", result)
            self.assertIn("robot_pose", result["initial_state"])
            self.assertIn("box_pose", result["initial_state"])
        finally:
            yaml_path.unlink(missing_ok=True)

    def test_load_task_extracts_randomization_config(self):
        """测试加载任务时提取randomization_config"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        yaml_content = {
            "spec_version": "1.0",
            "task_id": "test_with_randomization",
            "description": "测试带随机化的任务",
            "scene_id": "test_scene",
            "episode_length_s": 50.0,
            "success_criteria": [
                {
                    "criterion_type": "test_criterion",
                    "params": {},
                }
            ],
            "randomization_config": {
                "box_xy_m": 0.02,
                "box_yaw_rad": 0.1,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            loader = TaskLoader()
            result = loader.load_task(yaml_path)

            self.assertTrue(result["success"])
            self.assertIn("randomization_config", result)
            self.assertEqual(result["randomization_config"]["box_xy_m"], 0.02)
        finally:
            yaml_path.unlink(missing_ok=True)

    def test_get_active_task_returns_none_when_no_task_loaded(self):
        """测试未加载任务时get_active_task返回None"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        loader = TaskLoader()
        active = loader.get_active_task()

        self.assertIsNone(active)

    def test_get_active_task_returns_loaded_task_config(self):
        """测试加载任务后get_active_task返回任务配置"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        yaml_content = {
            "spec_version": "1.0",
            "task_id": "active_test_task",
            "description": "活动任务测试",
            "scene_id": "test_scene",
            "episode_length_s": 35.0,
            "success_criteria": [
                {
                    "criterion_type": "test",
                    "params": {},
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            yaml_path = Path(f.name)

        try:
            loader = TaskLoader()
            loader.load_task(yaml_path)
            active = loader.get_active_task()

            self.assertIsNotNone(active)
            self.assertEqual(active["task_id"], "active_test_task")
        finally:
            yaml_path.unlink(missing_ok=True)

    def test_switch_task_replaces_active_task(self):
        """测试切换任务时替换当前活动任务"""
        from openflex_isaac_bringup.task_loader import TaskLoader

        task1_content = {
            "spec_version": "1.0",
            "task_id": "task_one",
            "description": "第一个任务",
            "scene_id": "scene_1",
            "episode_length_s": 30.0,
            "success_criteria": [{"criterion_type": "test", "params": {}}],
        }

        task2_content = {
            "spec_version": "1.0",
            "task_id": "task_two",
            "description": "第二个任务",
            "scene_id": "scene_2",
            "episode_length_s": 40.0,
            "success_criteria": [{"criterion_type": "test", "params": {}}],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f1:
            yaml.dump(task1_content, f1)
            path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f2:
            yaml.dump(task2_content, f2)
            path2 = Path(f2.name)

        try:
            loader = TaskLoader()
            loader.load_task(path1)
            self.assertEqual(loader.get_active_task()["task_id"], "task_one")

            loader.load_task(path2)
            self.assertEqual(loader.get_active_task()["task_id"], "task_two")
        finally:
            path1.unlink(missing_ok=True)
            path2.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
