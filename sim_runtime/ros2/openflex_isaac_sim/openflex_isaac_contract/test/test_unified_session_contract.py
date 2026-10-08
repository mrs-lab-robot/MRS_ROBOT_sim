"""Tests for the framework-independent single-entry session contract."""

from __future__ import annotations

import unittest

from openflex_isaac_contract.embodiment_spec import EmbodimentSpec
from openflex_isaac_contract.preflight import run_preflight
from openflex_isaac_contract.session_config import (
    FrequencyConfig,
    SensorConfig,
    SessionConfig,
)


class UnifiedSessionContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embodiment = EmbodimentSpec.from_dict(
            {
                "spec_version": "1.0",
                "embodiment_id": "openflex",
                "display_name": "OpenFleX",
                "asset_path": "sim_runtime/assets/robots/openflex_robot.usda",
                "teleop_modes": ["vr", "keyboard"],
                "sensors": [
                    {"sensor_id": "base_camera", "sensor_type": "camera"},
                    {"sensor_id": "imu", "sensor_type": "imu"},
                ],
                "capabilities": ["capture", "replay"],
            }
        )
        self.config = SessionConfig(
            session_id="session-001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120, 30, 30),
            sensors=(
                SensorConfig("base_camera", "camera", frequency_hz=30),
                SensorConfig("imu", "imu", frequency_hz=30),
            ),
            machine_id="local",
            simulation_repo_root="/workspace/MRS_ROBOT_sim",
            isaac_lab_python="/workspace/MRS_ROBOT_sim/.venv/bin/python",
            lerobot_python="/opt/lerobot/bin/python",
            episode_root="/data/episodes",
            dataset_root="/data/lerobot",
            dataset_id="openflex_sim/navigation_to_goal",
            sim_client_address="10.0.0.8",
            teleop_mode="vr",
            enable_rviz=True,
            random_seed=7,
        )

    def test_session_round_trip_preserves_runtime_and_capture_choices(self):
        self.config.validate()

        restored = SessionConfig.from_dict(self.config.to_dict())

        self.assertEqual(restored, self.config)
        self.assertEqual(restored.config_hash, self.config.config_hash)
        self.assertEqual(len(restored.sensors), 2)
        self.assertTrue(restored.enable_rviz)
        self.assertEqual(restored.dataset_id, "openflex_sim/navigation_to_goal")
        self.assertEqual(restored.sim_client_address, "10.0.0.8")

    def test_config_hash_is_independent_of_metadata_mapping_order(self):
        first = self.config.to_dict()
        second = self.config.to_dict()
        first["metadata"] = {"z": 1, "nested": {"b": 2, "a": 3}}
        second["metadata"] = {"nested": {"a": 3, "b": 2}, "z": 1}

        self.assertEqual(
            SessionConfig.from_dict(first).config_hash,
            SessionConfig.from_dict(second).config_hash,
        )

    def test_embodiment_declares_sensor_and_teleop_capabilities(self):
        self.assertTrue(self.embodiment.supports_sensor("base_camera"))
        self.assertFalse(self.embodiment.supports_sensor("mid360"))
        self.assertTrue(self.embodiment.supports_teleop("keyboard"))
        self.assertEqual(self.embodiment.sensor_ids, ("base_camera", "imu"))

    def test_preflight_rejects_missing_required_sensor_before_kit_launch(self):
        report = run_preflight(
            self.config,
            embodiment=self.embodiment,
            task={
                "task_id": "navigation_to_goal",
                "capabilities": ["capture"],
                "required_sensors": ["base_camera", "mid360"],
                "allowed_teleop_modes": ["vr"],
            },
        )

        self.assertFalse(report.passed)
        self.assertIn("required_sensors", report.failed_checks)
        self.assertIn("mid360", report.failed_checks["required_sensors"])

    def test_preflight_accepts_a_capture_session_without_arena(self):
        report = run_preflight(
            self.config,
            embodiment=self.embodiment,
            task={
                "task_id": "navigation_to_goal",
                "capabilities": ["capture"],
                "required_sensors": ["base_camera", "imu"],
                "allowed_teleop_modes": ["vr", "keyboard"],
            },
        )

        self.assertTrue(report.passed, report.failed_checks)
        self.assertEqual(report.failed_checks, {})

    def test_baseline_task_declares_single_entry_capture_requirements(self):
        from openflex_isaac_contract.task_spec import load_task_spec

        task_path = (
            __import__("pathlib").Path(__file__).resolve().parents[4]
            / "config/tasks/navigation_to_goal.yaml"
        )
        task = load_task_spec(task_path)

        self.assertIn("capture", task.capabilities)
        self.assertIn("base_camera", task.required_sensors)
        self.assertEqual(task.allowed_teleop_modes, ("vr", "keyboard"))
        self.assertEqual(task.maturity, "draft")


if __name__ == "__main__":
    unittest.main()
