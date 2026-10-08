"""SceneSpec和TaskSpec验证测试 - 预期失败

测试数据验证逻辑：拒绝格式错误、空标识符、重复ID等。
"""

import tempfile
import unittest
from pathlib import Path


class SceneSpecValidationTest(unittest.TestCase):
    """SceneSpec数据验证测试"""

    def test_rejects_empty_scene_id(self):
        """拒绝空的scene_id"""
        from openflex_isaac_contract.scene_spec import load_scene_spec

        yaml_content = """
spec_version: "1.0"
scene_id: ""
description: 测试
base_stage_usd: test.usda
robot_spawn_pose:
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0, 1.0]
objects: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            # 预期失败：应抛出ValueError
            with self.assertRaises(ValueError) as cm:
                load_scene_spec(yaml_path)
            self.assertIn("scene_id", str(cm.exception))
        finally:
            Path(yaml_path).unlink()

    def test_rejects_invalid_robot_pose_dimensions(self):
        """拒绝错误维度的robot_spawn_pose"""
        from openflex_isaac_contract.scene_spec import load_scene_spec

        yaml_content = """
spec_version: "1.0"
scene_id: test_invalid_pose
description: 测试
base_stage_usd: test.usda
robot_spawn_pose:
  position: [0.0, 0.0]
  rotation: [0.0, 0.0, 0.0, 1.0]
objects: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_scene_spec(yaml_path)
            self.assertIn("position", str(cm.exception).lower())
        finally:
            Path(yaml_path).unlink()

    def test_rejects_duplicate_object_ids(self):
        """拒绝重复的object_id"""
        from openflex_isaac_contract.scene_spec import load_scene_spec

        yaml_content = """
spec_version: "1.0"
scene_id: test_duplicate
description: 测试
base_stage_usd: test.usda
robot_spawn_pose:
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0, 1.0]
objects:
  - object_id: box_1
    object_type: box
    position: [1.0, 0.0, 0.5]
    rotation: [0.0, 0.0, 0.0, 1.0]
  - object_id: box_1
    object_type: cylinder
    position: [2.0, 0.0, 0.5]
    rotation: [0.0, 0.0, 0.0, 1.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_scene_spec(yaml_path)
            self.assertIn("重复", str(cm.exception))
        finally:
            Path(yaml_path).unlink()

    def test_rejects_empty_object_id(self):
        """拒绝空的object_id"""
        from openflex_isaac_contract.scene_spec import load_scene_spec

        yaml_content = """
spec_version: "1.0"
scene_id: test_empty_obj_id
description: 测试
base_stage_usd: test.usda
robot_spawn_pose:
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0, 1.0]
objects:
  - object_id: ""
    object_type: box
    position: [1.0, 0.0, 0.5]
    rotation: [0.0, 0.0, 0.0, 1.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_scene_spec(yaml_path)
            self.assertIn("object_id", str(cm.exception))
        finally:
            Path(yaml_path).unlink()


class TaskSpecValidationTest(unittest.TestCase):
    """TaskSpec数据验证测试"""

    def test_rejects_empty_task_id(self):
        """拒绝空的task_id"""
        from openflex_isaac_contract.task_spec import load_task_spec

        yaml_content = """
spec_version: "1.0"
task_id: ""
description: 测试
scene_id: test_scene
episode_length_s: 30.0
success_criteria:
  - criterion_type: base_at_target
    params:
      target_position: [2.0, 1.0, 0.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_task_spec(yaml_path)
            self.assertIn("task_id", str(cm.exception))
        finally:
            Path(yaml_path).unlink()

    def test_rejects_non_positive_episode_length(self):
        """拒绝非正的episode_length_s"""
        from openflex_isaac_contract.task_spec import load_task_spec

        yaml_content = """
spec_version: "1.0"
task_id: test_invalid_length
description: 测试
scene_id: test_scene
episode_length_s: -10.0
success_criteria:
  - criterion_type: base_at_target
    params:
      target_position: [2.0, 1.0, 0.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_task_spec(yaml_path)
            self.assertIn("episode_length_s", str(cm.exception))
        finally:
            Path(yaml_path).unlink()

    def test_rejects_empty_success_criteria(self):
        """拒绝空的success_criteria列表"""
        from openflex_isaac_contract.task_spec import load_task_spec

        yaml_content = """
spec_version: "1.0"
task_id: test_no_criteria
description: 测试
scene_id: test_scene
episode_length_s: 30.0
success_criteria: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_task_spec(yaml_path)
            self.assertIn("success_criteria", str(cm.exception))
        finally:
            Path(yaml_path).unlink()

    def test_rejects_non_positive_tolerance(self):
        """拒绝非正的tolerance"""
        from openflex_isaac_contract.task_spec import load_task_spec

        yaml_content = """
spec_version: "1.0"
task_id: test_invalid_tolerance
description: 测试
scene_id: test_scene
episode_length_s: 30.0
success_criteria:
  - criterion_type: base_at_target
    params:
      target_position: [2.0, 1.0, 0.0]
    tolerance: -0.1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with self.assertRaises(ValueError) as cm:
                load_task_spec(yaml_path)
            self.assertIn("tolerance", str(cm.exception))
        finally:
            Path(yaml_path).unlink()


if __name__ == "__main__":
    unittest.main()
