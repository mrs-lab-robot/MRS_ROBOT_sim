#!/usr/bin/env python3
"""Guard the Isaac wrapper against stale integrated-description filenames."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
ROBOT_XACRO = (
    ROOT
    / "sim_runtime"
    / "ros2"
    / "openflex_isaac_sim"
    / "openflex_isaac_description"
    / "urdf"
    / "openflex_robot.urdf.xacro"
)
XACRO_NS = "{http://www.ros.org/wiki/xacro}"
INTEGRATED_XACRO = (
    "$(find openarmx_integrated_description)/urdf/"
    "openarmx_integrated_robot.urdf.xacro"
)


class IntegratedDescriptionSourceTest(unittest.TestCase):
    def test_wrapper_uses_installed_integrated_xacro_without_duplicate_hands(self) -> None:
        root = ET.parse(ROBOT_XACRO).getroot()
        include_paths = [
            include.get("filename")
            for include in root.findall(f"{XACRO_NS}include")
        ]

        self.assertIn(
            INTEGRATED_XACRO,
            include_paths,
            "the wrapper must include the integrated xacro installed by the current OpenFlex workspace",
        )
        self.assertFalse(
            any(path and path.endswith("openarmx_integrated_robot_o6.urdf.xacro") for path in include_paths),
            "the obsolete O6 xacro filename is not installed in the current OpenFlex description package",
        )
        self.assertEqual(
            [],
            root.findall(f".//{XACRO_NS}openarmx_hand"),
            "the current integrated xacro already mounts both hands; the wrapper must not add them again",
        )


if __name__ == "__main__":
    unittest.main()
