from __future__ import annotations

from pathlib import Path
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RVIZ_CONFIG = (
    REPO_ROOT
    / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/rviz/robot_control_only.rviz"
)


def _flatten_displays(displays):
    for display in displays:
        yield display
        yield from _flatten_displays(display.get("Displays", []))


class RvizSensorDisplayConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(RVIZ_CONFIG.read_text(encoding="utf-8"))
        cls.displays = list(
            _flatten_displays(cls.config["Visualization Manager"]["Displays"])
        )

    def test_control_layout_does_not_open_camera_image_displays(self):
        image_displays = [
            display
            for display in self.displays
            if display.get("Class") == "rviz_default_plugins/Image"
        ]
        self.assertEqual(image_displays, [])

    def test_rviz_lidar_displays_keep_sensor_data_qos(self):
        expected = {
            "rviz_default_plugins/PointCloud2": "/livox/lidar_points",
            "rviz_default_plugins/LaserScan": "/scan",
        }
        for display_class, topic in expected.items():
            matches = [
                display
                for display in self.displays
                if display.get("Class") == display_class
                and display.get("Topic", {}).get("Value") == topic
            ]
            self.assertTrue(matches, f"Missing RViz display for {topic}")
            self.assertEqual(
                matches[0]["Topic"].get("Reliability Policy"),
                "Best Effort",
                f"{topic} must accept sensor-data best-effort publishers",
            )
            self.assertIs(
                matches[0].get("Enabled"),
                False,
                f"{topic} must stay hidden unless the user opts in from simulation control",
            )

    def test_control_layout_uses_a_control_sensor_tabs_panel_disabled_by_default(self):
        panels = self.config["Panels"]
        panel = next(
            item
            for item in panels
            if item.get("Class") == "openflex_isaac_bringup/SimulationControlSensorsPanel"
        )
        self.assertIs(panel.get("Show Cameras"), False)
        self.assertIs(panel.get("Show Lidar"), False)
        self.assertNotIn(
            "swerve_base_panel/SwerveBasePanel",
            [item.get("Class") for item in panels],
        )
        self.assertNotIn(
            "lift_slide_panel/LiftPanel",
            [item.get("Class") for item in panels],
        )


if __name__ == "__main__":
    unittest.main()
