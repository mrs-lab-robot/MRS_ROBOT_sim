#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = (
    REPO_DIR
    / "sim_runtime"
    / "ros2"
    / "openflex_isaac_sim"
    / "openflex_isaac_contract"
)
sys.path.insert(0, str(PACKAGE_DIR))

import openflex_isaac_contract.manifest as manifest_module  # noqa: E402
from openflex_isaac_contract.manifest import (  # noqa: E402
    ManifestError,
    action_dimension,
    component_names,
    load_manifest,
    validate_manifest,
)


class ManifestSchemaTest(unittest.TestCase):
    def _validation_errors(self, manifest: dict[object, object]) -> list[str]:
        try:
            return validate_manifest(manifest)
        except (TypeError, ValueError) as error:
            self.fail(f"validate_manifest raised {type(error).__name__}: {error}")

    def test_default_manifest_has_expected_identity_and_action_dimension(self) -> None:
        manifest = load_manifest()

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["robot"]["id"], "openflex")
        self.assertEqual(manifest["robot"]["display_name"], "OpenFleX")
        self.assertEqual(action_dimension(manifest), 22)

    def test_action_components_are_ordered_and_unique(self) -> None:
        manifest = load_manifest()

        self.assertEqual(
            component_names(manifest, "actions"),
            [
                "base_twist",
                "lift_position",
                "lift_velocity",
                "head_position",
                "left_arm_position",
                "left_gripper_position",
                "right_arm_position",
                "right_gripper_position",
            ],
        )
        self.assertEqual(validate_manifest(manifest), [])

    def test_required_sensor_roles_are_present(self) -> None:
        manifest = load_manifest()
        sensors = {sensor["role"]: sensor for sensor in manifest["sensors"]}

        self.assertEqual(
            sorted(sensors),
            [
                "base_camera",
                "head_camera",
                "imu",
                "left_wrist_camera",
                "lidar",
                "odom",
                "right_wrist_camera",
            ],
        )
        self.assertEqual(sensors["head_camera"]["frame_id"], "head_yaw_link")
        self.assertEqual(sensors["left_wrist_camera"]["frame_id"], "openarmx_left_hand")
        self.assertEqual(sensors["right_wrist_camera"]["frame_id"], "openarmx_right_hand")

    def test_default_manifest_has_canonical_runtime_domains_and_action_contracts(self) -> None:
        manifest = load_manifest()

        self.assertIn("runtime", manifest)
        self.assertEqual(manifest["runtime"]["isaac_sim_version"], "6.0")
        self.assertEqual(manifest["domains"]["physical"]["ros_domain_id"], 0)
        self.assertEqual(manifest["domains"]["simulation"]["ros_domain_id"], 49)

        for action in manifest["actions"]:
            with self.subTest(action=action["name"]):
                self.assertEqual(len(action["fields"]), action["dimension"])
                self.assertEqual(len(action["units"]), action["dimension"])
                self.assertEqual(len(action["limits"]), action["dimension"])
                self.assertGreater(action["default_command_frequency_hz"], 0)
                self.assertTrue(action["safety_clipping"])

        base_twist = manifest["actions"][0]
        self.assertEqual(base_twist["limits"][0], {"minimum": -0.8, "maximum": 0.8})
        self.assertIn("max_wheel_speed", base_twist["limit_source"])
        self.assertIn("not a physical-hardware rating", base_twist["limit_source"])

    def test_joint_state_observation_layout_is_explicit_and_ordered(self) -> None:
        manifest = load_manifest()
        self.assertIn("observation_layout", manifest)
        layouts = {layout["observation"]: layout for layout in manifest["observation_layout"]}
        joint_state = layouts["joint_state"]

        self.assertEqual(joint_state["dimension"], 38)
        self.assertEqual(len(joint_state["fields"]), 38)
        self.assertEqual(joint_state["order"], joint_state["fields"])
        self.assertEqual(len(joint_state["units"]), 38)
        self.assertEqual(
            joint_state["fields"][:3],
            [
                "position.lift_joint",
                "position.openarmx_head_yaw_joint",
                "position.openarmx_head_pitch_joint",
            ],
        )

    def test_validate_manifest_reports_duplicate_action_names(self) -> None:
        manifest = load_manifest()
        duplicate = dict(manifest)
        duplicate["actions"] = [dict(item) for item in manifest["actions"]]
        duplicate["actions"][1]["name"] = "base_twist"

        self.assertIn(
            "actions contains duplicate name: base_twist",
            validate_manifest(duplicate),
        )

    def test_validate_manifest_rejects_missing_required_action(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["actions"] = manifest["actions"][1:]

        self.assertIn(
            "actions is missing required name: base_twist",
            validate_manifest(manifest),
        )

    def test_validate_manifest_rejects_missing_required_sensor_role(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["sensors"] = [sensor for sensor in manifest["sensors"] if sensor["role"] != "imu"]

        self.assertIn(
            "sensors is missing required role: imu",
            validate_manifest(manifest),
        )

    def test_validate_manifest_rejects_duplicate_joint_across_actions(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["actions"][-1]["joints"] = ["openarmx_left_joint1"]

        self.assertIn(
            "actions contains duplicate joint: openarmx_left_joint1",
            validate_manifest(manifest),
        )

    def test_validate_manifest_rejects_unknown_action_unit(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["actions"][0]["units"][0] = "km/h"

        self.assertIn(
            "actions base_twist contains unsupported unit: km/h",
            validate_manifest(manifest),
        )

    def test_validate_manifest_rejects_unregistered_action_topic(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["actions"][0]["topic"] = "/missing/cmd_vel"

        self.assertIn(
            "actions base_twist topic is not declared in ros2_topics: /missing/cmd_vel",
            validate_manifest(manifest),
        )

    def test_validate_manifest_rejects_non_numeric_action_dimension_without_raising(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["actions"][0]["dimension"] = "bad"

        self.assertIn(
            "actions base_twist dimension must be a positive integer",
            self._validation_errors(manifest),
        )
        self.assertEqual(action_dimension(manifest), 19)

    def test_validate_manifest_rejects_non_string_action_unit_without_raising(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["actions"][0]["units"][0] = ["invalid"]

        self.assertIn(
            "actions base_twist contains unsupported unit type: list",
            self._validation_errors(manifest),
        )

    def test_validate_manifest_rejects_non_string_observation_layout_unit_without_raising(self) -> None:
        manifest = deepcopy(load_manifest())
        manifest["observation_layout"][0]["units"][0] = {"invalid": "unit"}

        self.assertIn(
            "observation_layout joint_state contains unsupported unit type: dict",
            self._validation_errors(manifest),
        )

    def test_load_manifest_raises_on_missing_file(self) -> None:
        with self.assertRaises(ManifestError):
            load_manifest(PACKAGE_DIR / "config" / "missing.yaml")

    def test_default_manifest_path_resolves_typical_install_layout(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            install_prefix = Path(temporary_directory) / "install"
            module_path = (
                install_prefix
                / "lib"
                / "python3.10"
                / "site-packages"
                / "openflex_isaac_contract"
                / "manifest.py"
            )
            manifest_path = (
                install_prefix
                / "share"
                / "openflex_isaac_contract"
                / "config"
                / "embodiment.yaml"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.touch()

            with patch.object(manifest_module, "__file__", str(module_path)):
                self.assertEqual(manifest_module.default_manifest_path(), manifest_path)

    def test_default_manifest_path_resolves_ament_install_layout(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            package_prefix = Path(temporary_directory) / "install" / "openflex_isaac_contract"
            module_path = (
                package_prefix
                / "local"
                / "lib"
                / "python3.10"
                / "site-packages"
                / "openflex_isaac_contract"
                / "manifest.py"
            )
            manifest_path = (
                package_prefix
                / "share"
                / "openflex_isaac_contract"
                / "config"
                / "embodiment.yaml"
            )
            manifest_path.parent.mkdir(parents=True)
            manifest_path.touch()

            with patch.object(manifest_module, "__file__", str(module_path)):
                self.assertEqual(manifest_module.default_manifest_path(), manifest_path)


if __name__ == "__main__":
    unittest.main()
