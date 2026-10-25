"""守卫「参数只来自 YAML」这条规矩（见 docs/task-authoring.md）。

做法：拿真实的 task YAML 复制一份、改掉数值，再让任务类去读——
如果读出来的还是旧值，说明有人把坐标写死进了 Python。
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from mrs_arena.tasks.spec_placement import (
    object_placement_from_spec,
    randomization_from_spec,
    target_position_from_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BOX_TASK = REPO_ROOT / "sim_runtime/config/tasks/dual_arm_box_transport.yaml"


def _write_modified(source: Path, directory: Path, mutate) -> Path:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    mutate(document)
    target = directory / source.name
    target.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf-8")
    return target


class TaskSpecIsSourceOfTruthTest(unittest.TestCase):
    def test_box_placements_come_from_yaml_not_from_python(self) -> None:
        document = yaml.safe_load(BOX_TASK.read_text(encoding="utf-8"))
        expected_start = tuple(document["initial_state"]["box_pose"]["position"])
        expected_target = tuple(document["success_criteria"][0]["params"]["target_position"])

        placement = object_placement_from_spec(document, "box")
        target = target_position_from_spec(document, "box")
        self.assertEqual(placement.position, expected_start)
        self.assertEqual(target, expected_target)

        # 改 YAML -> 读出来的值必须跟着变（写死就会红）
        with tempfile.TemporaryDirectory() as temporary:
            moved = _write_modified(
                BOX_TASK,
                Path(temporary),
                lambda spec: spec["initial_state"]["box_pose"].update(
                    position=[0.71, -0.03, 0.80]
                ),
            )
            moved_document = yaml.safe_load(moved.read_text(encoding="utf-8"))
            moved_placement = object_placement_from_spec(moved_document, "box")
        self.assertEqual(moved_placement.position, (0.71, -0.03, 0.80))
        self.assertNotEqual(moved_placement.position, placement.position)

    def test_randomization_comes_from_yaml(self) -> None:
        document = yaml.safe_load(BOX_TASK.read_text(encoding="utf-8"))
        expected = document["randomization_config"]
        randomization = randomization_from_spec(document)
        self.assertAlmostEqual(randomization.xy_m, float(expected["box_xy_m"]))
        self.assertAlmostEqual(randomization.yaw_rad, float(expected["box_yaw_rad"]))

    def test_template_task_reads_spec(self) -> None:
        """模板本身也要能跑：说明新任务照抄就有 yaml 驱动。"""

        from mrs_arena.tasks._template_task import TemplateTask, default_template_task_path

        task = TemplateTask(default_template_task_path(), object_id="replace_me_object")
        self.assertEqual(task.object_placement().position, (0.0, 0.0, 0.0))
        self.assertEqual(task.target_position(), (0.0, 0.0, 0.0))
        self.assertEqual(task.randomization().xy_m, 0.0)
        self.assertIsNone(task.get_mimic_env_cfg(None))

    def test_box_task_parameters_come_from_yaml(self) -> None:
        """搬箱任务的阈值也必须来自 YAML（原来全写在 Python 常量里）。"""

        from openflex_isaac_contract.task_spec import load_task_spec

        from mrs_arena.tasks.box_transport_spec import box_transport_parameters_from_task

        parameters = box_transport_parameters_from_task(load_task_spec(BOX_TASK))
        self.assertEqual(parameters.initial_position, (0.65, 0.0, 0.80))
        self.assertEqual(parameters.target_position, (0.90, 0.0, 0.80))
        self.assertAlmostEqual(parameters.min_lift_height_m, 0.10)
        self.assertAlmostEqual(parameters.min_carry_distance_m, 0.20)
        self.assertAlmostEqual(parameters.position_tolerance_m, 0.05)
        self.assertAlmostEqual(parameters.episode_length_s, 60.0)
        self.assertAlmostEqual(parameters.box_randomization_xy_m, 0.02)
        self.assertAlmostEqual(parameters.box_randomization_yaw_rad, 0.10)

        def mutate(spec):
            spec["success_criteria"][0]["params"].update(
                min_lift_height=0.15, target_position=[0.95, 0.02, 0.80]
            )
            spec["episode_length_s"] = 45.0

        with tempfile.TemporaryDirectory() as temporary:
            moved = _write_modified(BOX_TASK, Path(temporary), mutate)
            moved_parameters = box_transport_parameters_from_task(load_task_spec(moved))

        self.assertAlmostEqual(moved_parameters.min_lift_height_m, 0.15)
        self.assertEqual(moved_parameters.target_position, (0.95, 0.02, 0.80))
        self.assertAlmostEqual(moved_parameters.episode_length_s, 45.0)
        self.assertNotEqual(
            moved_parameters.min_lift_height_m, parameters.min_lift_height_m
        )

    def test_template_files_are_not_registered_as_real_tasks(self) -> None:
        """模板不能被当成真任务跑起来（文件名下划线开头 + registry 是显式列表）。"""

        from mrs_arena.environments.openflex_task_registry import list_openflex_task_profiles

        task_ids = {profile.task_id for profile in list_openflex_task_profiles()}
        self.assertNotIn("replace_me", task_ids)
        self.assertFalse(any(task_id.startswith("_") for task_id in task_ids))


if __name__ == "__main__":
    unittest.main()
