"""Kit-dependent adapter checks; use the CPU-only transport_logic tests otherwise."""

from __future__ import annotations

import argparse
import os
import unittest


@unittest.skipUnless(
    os.environ.get("MRS_RUN_ISAAC_APP_TESTS") == "1",
    "requires Isaac Sim Kit; run scripts/smoke_openflex_dual_arm_box.py on a supported GPU host",
)
class TestDualArmBoxTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.headless = True
        cls.app_launcher = AppLauncher(args)
        from mrs_arena.tasks.dual_arm_box_transport import DualArmBoxTransportTask

        cls.task_class = DualArmBoxTransportTask

    @classmethod
    def tearDownClass(cls):
        cls.app_launcher.app.close()

    def setUp(self):
        self.task = self.task_class()

    def test_task_defines_bilateral_contact_sensors_and_success_metrics(self):
        scene_cfg = self.task.get_scene_cfg()
        self.assertTrue(hasattr(scene_cfg, "transport_box_left_contact"))
        self.assertTrue(hasattr(scene_cfg, "transport_box_right_contact"))

        # 验证接触传感器的 filter 路径配置正确
        left_sensor_cfg = scene_cfg.transport_box_left_contact
        right_sensor_cfg = scene_cfg.transport_box_right_contact

        # 每侧应该有 2 个 finger filter 路径
        self.assertEqual(len(left_sensor_cfg.filter_prim_paths_expr), 2)
        self.assertEqual(len(right_sensor_cfg.filter_prim_paths_expr), 2)

        # 验证路径是完整的 USD 层级，从 Robot/Geometry/base_link 到 link7
        expected_left_paths = {
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/lift_carriage_link/"
            "openarmx_left_link1/openarmx_left_link2/openarmx_left_link3/"
            "openarmx_left_link4/openarmx_left_link5/openarmx_left_link6/"
            "openarmx_left_link7/openarmx_left_left_finger",
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/lift_carriage_link/"
            "openarmx_left_link1/openarmx_left_link2/openarmx_left_link3/"
            "openarmx_left_link4/openarmx_left_link5/openarmx_left_link6/"
            "openarmx_left_link7/openarmx_left_right_finger",
        }
        expected_right_paths = {
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/lift_carriage_link/"
            "openarmx_right_link1/openarmx_right_link2/openarmx_right_link3/"
            "openarmx_right_link4/openarmx_right_link5/openarmx_right_link6/"
            "openarmx_right_link7/openarmx_right_left_finger",
            "{ENV_REGEX_NS}/Robot/Geometry/base_link/lift_carriage_link/"
            "openarmx_right_link1/openarmx_right_link2/openarmx_right_link3/"
            "openarmx_right_link4/openarmx_right_link5/openarmx_right_link6/"
            "openarmx_right_link7/openarmx_right_right_finger",
        }

        self.assertEqual(set(left_sensor_cfg.filter_prim_paths_expr), expected_left_paths)
        self.assertEqual(set(right_sensor_cfg.filter_prim_paths_expr), expected_right_paths)

        self.assertEqual(len(self.task.get_metrics()), 1)
        self.assertEqual(self.task.get_metrics()[0].name, "success_rate")

    def test_success_criterion_matches_frozen_baseline_thresholds(self):
        success = self.task.get_termination_cfg().success
        self.assertEqual(success.params["initial_position"], (0.65, 0.0, 0.80))
        self.assertEqual(success.params["target_position"], (0.90, 0.0, 0.80))
        self.assertEqual(success.params["min_lift_height_m"], 0.10)
        self.assertEqual(success.params["min_carry_distance_m"], 0.20)
        self.assertEqual(success.params["stable_seconds"], 1.0)

    def test_mimic_is_not_claimed_until_bimanual_datagen_adapter_exists(self):
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode

        self.assertIsNone(self.task.get_mimic_env_cfg(ArmMode.DUAL_ARM))


if __name__ == "__main__":
    unittest.main()
