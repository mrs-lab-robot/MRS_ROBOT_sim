from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

try:
    from mrs_robot_lab.environments.learning.configuration import load_task_configuration
    CONFIGURATION_IMPORT_ERROR = None
except ImportError as error:
    load_task_configuration = None
    CONFIGURATION_IMPORT_ERROR = error


class TaskConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        if CONFIGURATION_IMPORT_ERROR is not None:
            self.fail(f"task/scene configuration loader is missing: {CONFIGURATION_IMPORT_ERROR}")

    def test_baselines_load_and_resolve_all_scene_assets_relative_to_yaml(self) -> None:
        sim_root = Path(__file__).resolve().parents[2]
        navigation = load_task_configuration(
            sim_root / "sim_runtime/config/tasks/navigation_to_goal.yaml", backend="isaac_lab"
        )
        box = load_task_configuration(
            sim_root / "sim_runtime/config/tasks/dual_arm_box_transport.yaml", backend="isaac_lab"
        )

        self.assertEqual(navigation.scene.scene_id, "flat_navigation")
        self.assertTrue(navigation.base_stage_path.is_file())
        self.assertEqual(box.scene.scene_id, "dual_arm_box_tabletop")
        self.assertEqual(set(box.object_asset_paths), {"tabletop", "transport_box"})
        self.assertTrue(all(path.is_file() for path in box.object_asset_paths.values()))

    def test_rejects_task_scene_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "stage.usd").write_text("stage", encoding="utf-8")
            (root / "scene.yaml").write_text(
                yaml.safe_dump(
                    {
                        "spec_version": "1.0",
                        "scene_id": "scene_a",
                        "description": "test",
                        "base_stage_usd": "stage.usd",
                        "robot_spawn_pose": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]},
                        "objects": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "tasks").mkdir()
            task_path = root / "tasks/task.yaml"
            task_path.write_text(
                yaml.safe_dump(
                    {
                        "spec_version": "1.0",
                        "task_id": "task_a",
                        "description": "test",
                        "scene_id": "scene_b",
                        "episode_length_s": 1,
                        "success_criteria": [{"criterion_type": "done", "params": {}}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "references scene"):
                load_task_configuration(task_path, scene_spec_path=root / "scene.yaml")

    def test_rejects_unavailable_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "scene.yaml").write_text(
                yaml.safe_dump(
                    {
                        "spec_version": "1.0",
                        "scene_id": "scene_a",
                        "description": "test",
                        "base_stage_usd": "stage.usd",
                        "robot_spawn_pose": {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]},
                        "objects": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "stage.usd").write_text("stage", encoding="utf-8")
            task_path = root / "task.yaml"
            task_path.write_text(
                yaml.safe_dump(
                    {
                        "spec_version": "1.0",
                        "task_id": "task_a",
                        "description": "test",
                        "scene_id": "scene_a",
                        "episode_length_s": 1,
                        "success_criteria": [{"criterion_type": "done", "params": {}}],
                        "supported_backends": ["arena"],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not support backend 'isaac_lab'"):
                load_task_configuration(task_path, scene_spec_path=root / "scene.yaml")


if __name__ == "__main__":
    unittest.main()
