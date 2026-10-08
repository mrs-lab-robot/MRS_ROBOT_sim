"""SceneSpec和TaskSpec YAML契约的最小TDD测试

本测试定义框架无关的YAML格式契约，不依赖ROS/Kit/IsaacLab/Arena。
预期：全部失败，因为数据结构和加载器尚未实现。

测试覆盖：
1. SceneSpec YAML round-trip
2. TaskSpec YAML round-trip
3. 最小字段验证
"""

import tempfile
import unittest
from pathlib import Path


class SceneSpecContractTest(unittest.TestCase):
    """SceneSpec YAML契约测试（框架无关）"""

    def test_scene_spec_yaml_loads_minimal_scene(self):
        """验证最小SceneSpec可从YAML加载"""
        # 预期失败：load_scene_spec不存在
        from openflex_isaac_contract.scene_spec import load_scene_spec

        yaml_content = """
spec_version: "1.0"
scene_id: test_empty_scene
description: 空场景测试
base_stage_usd: environments/empty.usda
robot_spawn_pose:
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0, 1.0]
objects: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            scene = load_scene_spec(yaml_path)

            # 验证必需字段
            self.assertEqual(scene.spec_version, "1.0")
            self.assertEqual(scene.scene_id, "test_empty_scene")
            self.assertEqual(scene.base_stage_usd, "environments/empty.usda")
            self.assertEqual(scene.robot_spawn_pose, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
            self.assertEqual(len(scene.objects), 0)
        finally:
            Path(yaml_path).unlink()

    def test_scene_spec_yaml_loads_scene_with_object(self):
        """验证带物体的SceneSpec可从YAML加载"""
        from openflex_isaac_contract.scene_spec import load_scene_spec

        yaml_content = """
spec_version: "1.0"
scene_id: test_single_box
description: 单箱场景
base_stage_usd: environments/factory.usda
robot_spawn_pose:
  position: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0, 1.0]
objects:
  - object_id: box_1
    object_type: box
    position: [1.0, 0.0, 0.5]
    rotation: [0.0, 0.0, 0.0, 1.0]
    scale: [0.2, 0.3, 0.15]
    physics:
      mass: 2.0
      static_friction: 0.6
    semantic_label: target_object
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            scene = load_scene_spec(yaml_path)

            self.assertEqual(len(scene.objects), 1)
            obj = scene.objects[0]
            self.assertEqual(obj.object_id, "box_1")
            self.assertEqual(obj.object_type, "box")
            self.assertEqual(obj.position, (1.0, 0.0, 0.5))
            self.assertEqual(obj.semantic_label, "target_object")
        finally:
            Path(yaml_path).unlink()

    def test_scene_spec_round_trip_preserves_data(self):
        """验证SceneSpec保存后加载数据不变"""
        from openflex_isaac_contract.scene_spec import load_scene_spec, SceneSpec

        # 创建SceneSpec
        scene = SceneSpec(
            spec_version="1.0",
            scene_id="test_roundtrip",
            description="Round-trip测试",
            base_stage_usd="test.usda",
            robot_spawn_pose=((1.0, 2.0, 0.0), (0.0, 0.0, 0.707, 0.707)),
            objects=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "scene.yaml"
            scene.save_yaml(yaml_path)

            # 重新加载
            loaded = load_scene_spec(yaml_path)

            self.assertEqual(loaded.scene_id, scene.scene_id)
            self.assertEqual(loaded.robot_spawn_pose, scene.robot_spawn_pose)


class TaskSpecContractTest(unittest.TestCase):
    """TaskSpec YAML契约测试（框架无关）"""

    def test_task_spec_yaml_loads_minimal_task(self):
        """验证最小TaskSpec可从YAML加载"""
        # 预期失败：load_task_spec不存在
        from openflex_isaac_contract.task_spec import load_task_spec

        yaml_content = """
spec_version: "1.0"
task_id: test_navigation
description: 导航任务测试
scene_id: test_empty_scene
episode_length_s: 30.0
success_criteria:
  - criterion_type: base_at_target
    params:
      target_position: [2.0, 1.0, 0.0]
    tolerance: 0.1
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            task = load_task_spec(yaml_path)

            # 验证必需字段
            self.assertEqual(task.spec_version, "1.0")
            self.assertEqual(task.task_id, "test_navigation")
            self.assertEqual(task.scene_id, "test_empty_scene")
            self.assertEqual(task.episode_length_s, 30.0)
            self.assertEqual(len(task.success_criteria), 1)
            self.assertEqual(task.success_criteria[0].criterion_type, "base_at_target")
        finally:
            Path(yaml_path).unlink()

    def test_task_spec_yaml_loads_multi_criteria_task(self):
        """验证多条件TaskSpec可从YAML加载"""
        from openflex_isaac_contract.task_spec import load_task_spec

        yaml_content = """
spec_version: "1.0"
task_id: test_dual_arm_transport
description: 双臂搬运任务
scene_id: test_single_box
episode_length_s: 60.0
success_criteria:
  - criterion_type: object_at_location
    params:
      object_id: box_1
      target_position: [3.0, 1.0, 0.8]
    tolerance: 0.05
  - criterion_type: base_at_target
    params:
      target_position: [2.0, 0.0, 0.0]
    tolerance: 0.2
    required: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            task = load_task_spec(yaml_path)

            self.assertEqual(len(task.success_criteria), 2)
            # 第一个条件必需
            self.assertTrue(task.success_criteria[0].required)
            # 第二个条件可选
            self.assertFalse(task.success_criteria[1].required)
        finally:
            Path(yaml_path).unlink()

    def test_task_spec_round_trip_preserves_data(self):
        """验证TaskSpec保存后加载数据不变"""
        from openflex_isaac_contract.task_spec import load_task_spec, TaskSpec, SuccessCriterion

        # 创建TaskSpec
        task = TaskSpec(
            spec_version="1.0",
            task_id="test_roundtrip",
            description="Round-trip测试",
            scene_id="test_scene",
            episode_length_s=45.0,
            success_criteria=[
                SuccessCriterion(
                    criterion_type="base_at_target",
                    params={"target_position": [1.0, 2.0, 0.0]},
                    tolerance=0.15,
                )
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "task.yaml"
            task.save_yaml(yaml_path)

            # 重新加载
            loaded = load_task_spec(yaml_path)

            self.assertEqual(loaded.task_id, task.task_id)
            self.assertEqual(loaded.episode_length_s, task.episode_length_s)
            self.assertEqual(len(loaded.success_criteria), 1)

    def test_task_spec_preserves_initial_state_randomization_and_capabilities(self):
        from openflex_isaac_contract.task_spec import load_task_spec

        initial_state = {
            "robot_pose": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            "box_pose": {"position": [0.5, 0.0, 0.75], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        }
        randomization = {
            "box_xy_m": 0.02,
            "box_yaw_rad": 0.1,
            "light_intensity_scale": [0.9, 1.1],
        }
        task_yaml = """
spec_version: "1.0"
task_id: dual_arm_box_transport
description: dual-arm box transport
scene_id: tabletop_box
episode_length_s: 60.0
success_criteria:
  - criterion_type: box_at_target
    params: {target_position: [0.8, 0.0, 0.75], min_lift_height: 0.1, min_carry_distance: 0.2, stable_seconds: 1.0}
    tolerance: 0.05
initial_state:
  robot_pose: {position: [0.0, 0.0, 0.0], yaw: 0.0}
  box_pose: {position: [0.5, 0.0, 0.75], rotation_xyzw: [0.0, 0.0, 0.0, 1.0]}
randomization_config:
  box_xy_m: 0.02
  box_yaw_rad: 0.1
  light_intensity_scale: [0.9, 1.1]
supported_backends: [isaac_sim, isaac_lab, arena]
capabilities: [capture, replay, mimic, rl, evaluation]
action_spec: {name: bilateral_joint_targets, units: rad}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as stream:
            stream.write(task_yaml)
            path = Path(stream.name)
        try:
            task = load_task_spec(path)
            self.assertEqual(getattr(task, "initial_state", None), initial_state)
            self.assertEqual(getattr(task, "randomization_config", None), randomization)
            self.assertEqual(getattr(task, "supported_backends", None), ("isaac_sim", "isaac_lab", "arena"))
            self.assertEqual(getattr(task, "capabilities", None), ("capture", "replay", "mimic", "rl", "evaluation"))
            self.assertEqual(getattr(task, "action_spec", None), {"name": "bilateral_joint_targets", "units": "rad"})
            saved = path.with_name("roundtrip_task.yaml")
            task.save_yaml(saved)
            round_trip = load_task_spec(saved)
            self.assertEqual(round_trip.initial_state, initial_state)
            self.assertEqual(round_trip.randomization_config, randomization)
            self.assertEqual(round_trip.capabilities, task.capabilities)
            saved.unlink()
        finally:
            path.unlink(missing_ok=True)

    def test_navigation_and_box_baseline_configs_are_versioned_and_asset_resolvable(self):
        from openflex_isaac_contract.scene_spec import load_scene_spec
        from openflex_isaac_contract.task_spec import load_task_spec

        sim_root = Path(__file__).resolve().parents[4]
        scenes = {
            "navigation": sim_root / "config/scenes/flat_navigation.yaml",
            "box": sim_root / "config/scenes/dual_arm_box_tabletop.yaml",
        }
        tasks = {
            "navigation": sim_root / "config/tasks/navigation_to_goal.yaml",
            "box": sim_root / "config/tasks/dual_arm_box_transport.yaml",
        }
        for config_path in (*scenes.values(), *tasks.values()):
            self.assertTrue(config_path.is_file(), f"missing baseline config: {config_path}")

        nav_scene = load_scene_spec(scenes["navigation"])
        box_scene = load_scene_spec(scenes["box"])
        nav_task = load_task_spec(tasks["navigation"])
        box_task = load_task_spec(tasks["box"])

        self.assertEqual(nav_scene.spec_version, "1.0")
        self.assertEqual(nav_scene.scene_id, "flat_navigation")
        self.assertTrue((scenes["navigation"].parent / nav_scene.base_stage_usd).resolve().is_file())
        self.assertEqual(nav_task.episode_length_s, 30.0)
        nav_criterion = nav_task.success_criteria[0]
        self.assertEqual(nav_criterion.tolerance, 0.15)
        self.assertEqual(nav_criterion.params["heading_tolerance_deg"], 10.0)
        self.assertEqual(nav_criterion.params["hold_seconds"], 1.0)
        self.assertEqual(nav_task.supported_backends, ("isaac_lab", "arena"))
        self.assertEqual(nav_task.capabilities, ("capture", "replay", "rl", "evaluation"))

        self.assertEqual(box_scene.scene_id, "dual_arm_box_tabletop")
        self.assertTrue((scenes["box"].parent / box_scene.base_stage_usd).resolve().is_file())
        self.assertEqual(len(box_scene.objects), 2)
        for placement in box_scene.objects:
            self.assertTrue((scenes["box"].parent / placement.usd_path).resolve().is_file())
        self.assertEqual(box_task.episode_length_s, 60.0)
        box_criterion = box_task.success_criteria[0]
        self.assertEqual(box_criterion.params["min_lift_height"], 0.10)
        self.assertEqual(box_criterion.params["min_carry_distance"], 0.20)
        self.assertEqual(box_criterion.tolerance, 0.05)
        self.assertEqual(box_criterion.params["stable_seconds"], 1.0)


if __name__ == "__main__":
    unittest.main()
