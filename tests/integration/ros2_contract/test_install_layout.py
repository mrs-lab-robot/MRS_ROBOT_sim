#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import unittest


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_DIR / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_contract"


class InstallLayoutTest(unittest.TestCase):
    def test_description_install_directories_exist_in_a_clean_checkout(self) -> None:
        package_dir = (
            REPO_DIR
            / "sim_runtime"
            / "ros2"
            / "openflex_isaac_sim"
            / "openflex_isaac_description"
        )

        for directory in ("meshes", "launch"):
            with self.subTest(directory=directory):
                self.assertTrue(
                    (package_dir / directory).is_dir(),
                    f"CMake installs {directory}/ but it is missing from the source checkout",
                )

    def _installed_package_prefix(self) -> Path:
        prefixes = [
            Path(prefix)
            for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
            if prefix
        ]
        for prefix in prefixes:
            if (prefix / "share" / "openflex_isaac_contract").is_dir():
                return prefix
        self.skipTest(
            "No installed openflex_isaac_contract package found in "
            "AMENT_PREFIX_PATH; source the install/setup.bash before running "
            "this test."
        )

    def test_installs_config_file(self) -> None:
        prefix = self._installed_package_prefix()
        config = prefix / "share" / "openflex_isaac_contract" / "config" / "embodiment.yaml"

        self.assertTrue(config.is_file(), f"Installed config is missing: {config}")

    def test_installs_executable_verify_script(self) -> None:
        prefix = self._installed_package_prefix()
        verifier = prefix / "lib" / "openflex_isaac_contract" / "verify_embodiment_contract.py"

        self.assertTrue(verifier.is_file(), f"Installed verifier is missing: {verifier}")
        self.assertTrue(os.access(verifier, os.X_OK), f"Installed verifier is not executable: {verifier}")

    def test_package_declares_yaml_runtime_dependency(self) -> None:
        package_xml = (PACKAGE_DIR / "package.xml").read_text(encoding="utf-8")

        self.assertIn("<exec_depend>python3-yaml</exec_depend>", package_xml)
        self.assertIn("<buildtool_depend>ament_cmake_python</buildtool_depend>", package_xml)


if __name__ == "__main__":
    unittest.main()
