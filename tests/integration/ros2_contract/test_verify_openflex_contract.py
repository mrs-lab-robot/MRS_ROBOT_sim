#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_DIR / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_contract"
BRINGUP_DIR = REPO_DIR / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_bringup"
SENSORS_CONFIG = REPO_DIR / "sim_runtime" / "config" / "sensors" / "sensors.isaac.yaml"
SCRIPT_PATH = PACKAGE_DIR / "scripts" / "verify_embodiment_contract.py"
sys.path.insert(0, str(PACKAGE_DIR))


def load_module():
    spec = importlib.util.spec_from_file_location("verify_openflex_contract", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VerifyOpenFlexContractTest(unittest.TestCase):
    def test_installed_cli_reports_missing_default_configs_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "install"
            site_packages = prefix / "lib" / "python3.10" / "site-packages"
            source_python_package = PACKAGE_DIR / "openflex_isaac_contract"
            installed_package = site_packages / "openflex_isaac_contract"
            installed_script = (
                prefix / "lib" / "openflex_isaac_contract" / "verify_embodiment_contract.py"
            )
            contract_config = prefix / "share" / "openflex_isaac_contract" / "config"
            package_index = prefix / "share" / "ament_index" / "resource_index" / "packages"

            shutil.copytree(source_python_package, installed_package)
            installed_script.parent.mkdir(parents=True)
            shutil.copy2(SCRIPT_PATH, installed_script)
            contract_config.mkdir(parents=True)
            shutil.copy2(PACKAGE_DIR / "config" / "embodiment.yaml", contract_config)
            package_index.mkdir(parents=True)
            (package_index / "openflex_isaac_contract").touch()

            environment = os.environ.copy()
            environment["AMENT_PREFIX_PATH"] = str(prefix)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(site_packages), environment.get("PYTHONPATH", ""))
            )
            outside_worktree = Path(temporary_directory) / "outside-worktree"
            outside_worktree.mkdir()
            result = subprocess.run(
                [sys.executable, str(installed_script)],
                cwd=outside_worktree,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            self.assertNotIn("Traceback", output)
            self.assertIn("controllers.isaac.mobile_base.yaml", output)
            self.assertIn("openflex_isaac_bringup", output)
            self.assertIn("--controllers", output)
            self.assertIn("--sensors", output)

    def test_installed_cli_uses_ament_bringup_share_from_non_worktree_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            prefix = Path(temporary_directory) / "install"
            site_packages = prefix / "lib" / "python3.10" / "site-packages"
            source_python_package = PACKAGE_DIR / "openflex_isaac_contract"
            installed_package = site_packages / "openflex_isaac_contract"
            installed_script = (
                prefix / "lib" / "openflex_isaac_contract" / "verify_embodiment_contract.py"
            )
            contract_config = prefix / "share" / "openflex_isaac_contract" / "config"
            bringup_config = prefix / "share" / "openflex_isaac_bringup" / "config"
            package_index = prefix / "share" / "ament_index" / "resource_index" / "packages"

            self.assertTrue(source_python_package.is_dir())
            shutil.copytree(source_python_package, installed_package)
            installed_script.parent.mkdir(parents=True)
            shutil.copy2(SCRIPT_PATH, installed_script)
            contract_config.mkdir(parents=True)
            shutil.copy2(PACKAGE_DIR / "config" / "embodiment.yaml", contract_config)
            bringup_config.mkdir(parents=True)
            shutil.copy2(
                BRINGUP_DIR / "config" / "controllers.isaac.mobile_base.yaml",
                bringup_config,
            )
            shutil.copy2(SENSORS_CONFIG, bringup_config)
            package_index.mkdir(parents=True)
            for package_name in ("openflex_isaac_contract", "openflex_isaac_bringup"):
                (package_index / package_name).touch()

            environment = os.environ.copy()
            environment["AMENT_PREFIX_PATH"] = str(prefix)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(site_packages), environment.get("PYTHONPATH", ""))
            )
            outside_worktree = Path(temporary_directory) / "outside-worktree"
            outside_worktree.mkdir()
            result = subprocess.run(
                [sys.executable, str(installed_script)],
                cwd=outside_worktree,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout,
                "[OK] OpenFleX embodiment contract matches Isaac controller and sensor configs\n",
            )

    def test_repository_isaac_configs_match_contract(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        controllers = verifier.load_yaml(
            BRINGUP_DIR / "config" / "controllers.isaac.mobile_base.yaml"
        )
        sensors = verifier.load_yaml(
            SENSORS_CONFIG
        )

        self.assertEqual(verifier.verify_controller_contract(manifest, controllers), [])
        self.assertEqual(verifier.verify_sensor_contract(manifest, sensors), [])

    def test_controller_verifier_reports_wrong_left_arm_joint_order(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        controllers = verifier.load_yaml(
            BRINGUP_DIR / "config" / "controllers.isaac.mobile_base.yaml"
        )
        controllers["left_forward_position_controller"]["ros__parameters"]["joints"] = [
            "openarmx_left_joint2",
            "openarmx_left_joint1",
            "openarmx_left_joint3",
            "openarmx_left_joint4",
            "openarmx_left_joint5",
            "openarmx_left_joint6",
            "openarmx_left_joint7",
            "openarmx_left_finger_joint1",
        ]

        self.assertEqual(
            verifier.verify_controller_contract(manifest, controllers),
            ["controller left_forward_position_controller joints do not match contract"],
        )

    def test_controller_verifier_derives_base_topic_from_manifest(self) -> None:
        verifier = load_module()
        manifest = deepcopy(verifier.load_manifest())
        controllers = verifier.load_yaml(
            BRINGUP_DIR / "config" / "controllers.isaac.mobile_base.yaml"
        )
        manifest["actions"][0]["topic"] = "/contract/cmd_vel"
        controllers["swerve_drive_controller"]["ros__parameters"]["cmd_vel_topic"] = "/contract/cmd_vel"

        self.assertEqual(verifier.verify_controller_contract(manifest, controllers), [])

    def test_controller_verifier_reports_update_rate_drift(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        controllers = verifier.load_yaml(
            BRINGUP_DIR / "config" / "controllers.isaac.mobile_base.yaml"
        )
        controllers["controller_manager"]["ros__parameters"]["update_rate"] = 99

        self.assertEqual(
            verifier.verify_controller_contract(manifest, controllers),
            ["controller_manager update_rate does not match contract runtime"],
        )

    def test_sensor_verifier_reports_wrong_head_topic(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        sensors = verifier.load_yaml(
            SENSORS_CONFIG
        )
        sensors["target_topics"]["head_camera"]["color_image"] = "/wrong/head"

        self.assertEqual(
            verifier.verify_sensor_contract(manifest, sensors),
            ["sensor role head_camera color_topic /cam_head/color/image is missing from sensors.isaac.yaml"],
        )

    def test_sensor_verifier_reports_camera_info_drift(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        sensors = verifier.load_yaml(
            SENSORS_CONFIG
        )
        sensors["target_topics"]["base_camera"]["color_camera_info"] = "/wrong/base/camera_info"

        self.assertEqual(
            verifier.verify_sensor_contract(manifest, sensors),
            [
                "sensor role base_camera camera_info_topic "
                "/cam_base/color/camera_info is missing from sensors.isaac.yaml"
            ],
        )

    def test_sensor_verifier_reports_imu_topic_drift(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        sensors = verifier.load_yaml(
            SENSORS_CONFIG
        )
        sensors["target_topics"]["imu"]["livox"] = "/wrong/livox/imu"

        self.assertEqual(
            verifier.verify_sensor_contract(manifest, sensors),
            ["imu topic does not match contract"],
        )

    def test_sensor_verifier_reports_urdf_frame_mapping_drift(self) -> None:
        verifier = load_module()
        manifest = verifier.load_manifest()
        sensors = verifier.load_yaml(
            SENSORS_CONFIG
        )
        sensors["isaac_urdf_sensors"]["head_link"] = "wrong_head_frame"

        self.assertEqual(
            verifier.verify_sensor_contract(manifest, sensors),
            ["sensor role head_camera frame_id does not match isaac_urdf_sensors"],
        )

    def test_cli_reports_invalid_manifest_yaml_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            contract = Path(temporary_directory) / "invalid-contract.yaml"
            contract.write_text("actions: [unterminated\n", encoding="utf-8")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(PACKAGE_DIR), environment.get("PYTHONPATH", ""))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--contract",
                    str(contract),
                    "--controllers",
                    str(BRINGUP_DIR / "config" / "controllers.isaac.mobile_base.yaml"),
                    "--sensors",
                    str(SENSORS_CONFIG),
                ],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
            )

            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            self.assertNotIn("Traceback", output)
            self.assertIn("[ERROR]", output)


if __name__ == "__main__":
    unittest.main()
