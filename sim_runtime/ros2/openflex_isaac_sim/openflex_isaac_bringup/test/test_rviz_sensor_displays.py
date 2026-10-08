from pathlib import Path
import os
import unittest

import yaml


CONFIG_PATH = Path(__file__).parents[1] / "rviz" / "robot_control_only.rviz"
SENSOR_CONFIG_PATH = Path(__file__).parents[1] / "rviz" / "sensor_monitor.rviz"


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


class RvizSensorDisplaysTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.groups = [
            entry
            for entry in _walk(cls.config)
            if entry.get("Class") == "rviz_common/Group"
        ]

    def test_sensor_group_is_enabled(self):
        sensor_group = next(
            group for group in self.groups if group.get("Name") == "Sensors"
        )
        self.assertIs(sensor_group.get("Enabled"), True)

    def test_control_layout_has_no_camera_image_displays_or_docks(self):
        displays = self.config["Visualization Manager"]["Displays"]
        self.assertFalse(
            any(item.get("Class") == "rviz_default_plugins/Image" for item in _walk(displays)),
            "camera image displays belong to the sensor view, not the control layout",
        )
        geometry = self.config.get("Window Geometry", {})
        for name in (
            "Head Camera Image",
            "Left Wrist Camera Image",
            "Right Wrist Camera Image",
            "Base Camera Image",
        ):
            with self.subTest(dock=name):
                self.assertNotIn(name, geometry)

    def test_control_layout_lidar_displays_use_best_effort_qos(self):
        sensor_group = next(
            group for group in self.groups if group.get("Name") == "Sensors"
        )
        sensor_displays = sensor_group["Displays"]
        expected = {
            "/livox/lidar_points": (sensor_displays, "rviz_default_plugins/PointCloud2", False),
            "/scan": (sensor_displays, "rviz_default_plugins/LaserScan", False),
        }
        for topic, (displays, display_class, enabled) in expected.items():
            with self.subTest(topic=topic):
                display = next(
                    item
                    for item in displays
                    if item.get("Class") == display_class
                    and item.get("Topic", {}).get("Value") == topic
                )
                self.assertIs(display.get("Enabled"), enabled)
                self.assertEqual(
                    display["Topic"].get("Reliability Policy"), "Best Effort"
                )

    def test_rviz_window_saves_its_control_panel_docking_state(self):
        geometry = self.config.get("Window Geometry", {})
        self.assertTrue(geometry.get("QMainWindow State"))

    def test_control_layout_contains_composite_left_and_two_right_panels(self):
        panels = {
            panel["Name"]: panel["Class"]
            for panel in self.config["Panels"]
            if panel.get("Name")
        }
        self.assertEqual(
            panels,
            {
                "SimulationControlSensorsPanel": "openflex_isaac_bringup/SimulationControlSensorsPanel",
                "HeadJointSliderPanel": "openarmx_head_joint_slider_panel/HeadJointSliderPanel",
                "JointSliderPanel": "openarmx_joint_slider_panel/JointSliderPanel",
            },
            "the left dock holds switchable base/lift and sensor pages; head and arms stay on the right",
        )
        panel = next(item for item in self.config["Panels"] if item["Name"] == "SimulationControlSensorsPanel")
        self.assertIs(panel.get("Show Cameras"), False)
        self.assertIs(panel.get("Show Lidar"), False)
        self.assertIn("BasePanel", panel)
        self.assertIn("LiftPanel", panel)
        geometry = self.config.get("Window Geometry", {})
        self.assertNotIn("Displays", geometry)
        self.assertNotIn("Views", geometry)

    def test_control_panels_are_docked_left_and_right_like_the_robot_ui(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtCore import QByteArray, Qt
        from PyQt5.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow

        panels = {
            panel["Name"]: panel["Class"]
            for panel in self.config["Panels"]
            if panel.get("Name")
        }
        self.assertEqual(
            panels["SimulationControlSensorsPanel"],
            "openflex_isaac_bringup/SimulationControlSensorsPanel",
        )
        self.assertEqual(
            panels["HeadJointSliderPanel"],
            "openarmx_head_joint_slider_panel/HeadJointSliderPanel",
        )
        self.assertEqual(
            panels["JointSliderPanel"],
            "openarmx_joint_slider_panel/JointSliderPanel",
        )

        state = self.config.get("Window Geometry", {}).get("QMainWindow State", "")
        self.assertTrue(state, "RViz must save an explicit control-panel dock layout")
        app = QApplication.instance() or QApplication([])
        window = QMainWindow()
        dock_names = (
            "SimulationControlSensorsPanel",
            "HeadJointSliderPanel",
            "JointSliderPanel",
        )
        docks = {}
        for name in dock_names:
            dock = QDockWidget(name, window)
            dock.setObjectName(name)
            dock.setWidget(QLabel(name))
            window.addDockWidget(Qt.LeftDockWidgetArea, dock)
            docks[name] = dock

        self.assertTrue(window.restoreState(QByteArray.fromHex(state.encode("ascii"))))
        self.assertEqual(window.dockWidgetArea(docks["SimulationControlSensorsPanel"]), Qt.LeftDockWidgetArea)
        self.assertEqual(
            window.dockWidgetArea(docks["HeadJointSliderPanel"]), Qt.RightDockWidgetArea
        )
        self.assertEqual(
            window.dockWidgetArea(docks["JointSliderPanel"]), Qt.RightDockWidgetArea
        )
        self.assertTrue(
            all(
                window.dockWidgetArea(dock) in (Qt.LeftDockWidgetArea, Qt.RightDockWidgetArea)
                for dock in docks.values()
            ),
            "all control docks must remain on the sides, with nothing docked below",
        )
        self.assertIsNot(
            window.dockWidgetArea(docks["SimulationControlSensorsPanel"]),
            window.dockWidgetArea(docks["HeadJointSliderPanel"]),
        )
        window.close()
        app.processEvents()

    def test_sensor_monitor_layout_shows_camera_and_both_lidar_views(self):
        config = yaml.safe_load(SENSOR_CONFIG_PATH.read_text(encoding="utf-8"))
        groups = [
            entry
            for entry in _walk(config)
            if entry.get("Class") == "rviz_common/Group"
            and entry.get("Name") == "Sensors"
        ]
        self.assertEqual(len(groups), 1)
        sensor_group = groups[0]
        self.assertIs(sensor_group.get("Enabled"), True)
        top_level_displays = config["Visualization Manager"]["Displays"]
        displays = sensor_group["Displays"]
        expected = {
            "/livox/lidar_points": "rviz_default_plugins/PointCloud2",
            "/scan": "rviz_default_plugins/LaserScan",
        }
        for topic, display_class in expected.items():
            with self.subTest(topic=topic):
                display = next(
                    item
                    for item in displays
                    if item.get("Class") == display_class
                    and item.get("Topic", {}).get("Value") == topic
                )
                self.assertIs(display.get("Enabled"), True)
                self.assertEqual(
                    display["Topic"].get("Reliability Policy"), "Best Effort"
                )

        image_displays = [item for item in top_level_displays if item.get("Class") == "rviz_default_plugins/Image"]
        self.assertEqual(len(image_displays), 1)
        self.assertEqual(
            image_displays[0].get("Topic", {}).get("Value"), "/cam_head/color/image"
        )
        self.assertIs(image_displays[0].get("Enabled"), True)
        self.assertEqual(
            image_displays[0]["Topic"].get("Reliability Policy"), "Best Effort"
        )
        self.assertNotIn("QMainWindow State", config.get("Window Geometry", {}))


if __name__ == "__main__":
    unittest.main()
