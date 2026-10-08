#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import yaml


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_DIR / "ros2_pkgs" / "openflex_isaac_sim" / "openflex_isaac_bringup"
LAUNCH_PATH = PACKAGE_DIR / "launch" / "rviz_only.launch.py"
RVIZ_CONFIG_PATH = PACKAGE_DIR / "rviz" / "robot_control_only.rviz"


class _FakeAction:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def load_launch_module():
    launch_module = types.ModuleType("launch")
    launch_module.LaunchDescription = _FakeAction

    launch_actions_module = types.ModuleType("launch.actions")
    launch_actions_module.DeclareLaunchArgument = _FakeAction
    launch_actions_module.OpaqueFunction = _FakeAction
    launch_actions_module.SetEnvironmentVariable = _FakeAction
    launch_actions_module.UnsetEnvironmentVariable = _FakeAction

    launch_substitutions_module = types.ModuleType("launch.substitutions")
    launch_substitutions_module.LaunchConfiguration = _FakeAction

    launch_ros_module = types.ModuleType("launch_ros")
    launch_ros_actions_module = types.ModuleType("launch_ros.actions")
    launch_ros_actions_module.Node = _FakeAction

    launch_ros_parameters_module = types.ModuleType("launch_ros.parameter_descriptions")
    launch_ros_parameters_module.ParameterValue = _FakeAction

    ament_module = types.ModuleType("ament_index_python")
    ament_packages_module = types.ModuleType("ament_index_python.packages")
    ament_packages_module.get_package_share_directory = lambda _name: "/tmp"

    modules = {
        "ament_index_python": ament_module,
        "ament_index_python.packages": ament_packages_module,
        "launch": launch_module,
        "launch.actions": launch_actions_module,
        "launch.substitutions": launch_substitutions_module,
        "launch_ros": launch_ros_module,
        "launch_ros.actions": launch_ros_actions_module,
        "launch_ros.parameter_descriptions": launch_ros_parameters_module,
    }
    spec = importlib.util.spec_from_file_location("rviz_only_launch", LAUNCH_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class RvizOnlyLaunchTest(unittest.TestCase):
    def test_rviz_declares_lidar_scan_and_odometry_displays(self) -> None:
        text = RVIZ_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("Name: LivoxPointCloud", text)
        self.assertIn("Value: /livox/lidar_points", text)
        self.assertIn("Name: LaserScan", text)
        self.assertIn("Value: /scan", text)
        self.assertIn("Name: Odometry", text)
        self.assertIn("Value: /odom", text)
        self.assertIn("Name: FAST-LIO Estimated Path", text)
        self.assertIn("Value: /fastlio2/lio_path", text)

    def test_rviz_display_error_explains_missing_gui_environment(self) -> None:
        launch_file = load_launch_module()

        with mock.patch.dict(os.environ, {}, clear=True):
            error = launch_file._rviz_display_error("")

        self.assertIsNotNone(error)
        assert error is not None
        self.assertIn("DISPLAY", error)
        self.assertIn("qt_qpa_platform", error)

    def test_rviz_display_error_allows_display_wayland_or_explicit_qt_platform(self) -> None:
        launch_file = load_launch_module()

        with mock.patch.dict(os.environ, {"DISPLAY": ":0"}, clear=True):
            self.assertIsNone(launch_file._rviz_display_error(""))
        with mock.patch.dict(os.environ, {"WAYLAND_DISPLAY": "wayland-0"}, clear=True):
            self.assertIsNone(launch_file._rviz_display_error(""))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(launch_file._rviz_display_error("offscreen"))
        with mock.patch.dict(os.environ, {"QT_QPA_PLATFORM": "xcb"}, clear=True):
            self.assertIsNone(launch_file._rviz_display_error(""))

    def test_rviz_node_environment_passes_simulation_mode_and_joint_state_topic(self) -> None:
        launch_file = load_launch_module()

        self.assertEqual(
            launch_file._rviz_node_environment(""),
            {
                "OPENFLEX_LIFT_SIMULATION_MODE": "true",
                "OPENFLEX_LIFT_JOINT_STATES_TOPIC": "/joint_states",
            },
        )
        self.assertEqual(
            launch_file._rviz_node_environment(
                "offscreen", simulation_mode="false", joint_states_topic="/robot/joint_states"
            ),
            {
                "QT_QPA_PLATFORM": "offscreen",
                "OPENFLEX_LIFT_SIMULATION_MODE": "false",
                "OPENFLEX_LIFT_JOINT_STATES_TOPIC": "/robot/joint_states",
            },
        )

    def test_runtime_rviz_config_switches_sensor_displays_without_changing_control_pages(self):
        launch_file = load_launch_module()
        base_config = yaml.safe_load(RVIZ_CONFIG_PATH.read_text(encoding="utf-8"))

        control = launch_file._apply_sensor_options(base_config, False, False)
        camera_only = launch_file._apply_sensor_options(base_config, True, False)
        lidar_only = launch_file._apply_sensor_options(base_config, False, True)

        def panel_settings(config):
            return next(
                panel
                for panel in config["Panels"]
                if panel.get("Class")
                == "openflex_isaac_bringup/SimulationControlSensorsPanel"
            )

        self.assertFalse(panel_settings(control)["Show Cameras"])
        self.assertFalse(panel_settings(control)["Show Lidar"])
        self.assertTrue(panel_settings(camera_only)["Show Cameras"])
        self.assertFalse(panel_settings(camera_only)["Show Lidar"])
        self.assertFalse(panel_settings(lidar_only)["Show Cameras"])
        self.assertTrue(panel_settings(lidar_only)["Show Lidar"])

        lidar_displays = [
            display
            for display in _flatten_displays(control["Visualization Manager"]["Displays"])
            if display.get("Name") in {"LivoxPointCloud", "LaserScan"}
        ]
        self.assertTrue(lidar_displays)
        self.assertTrue(all(display["Enabled"] is False for display in lidar_displays))
        enabled_lidar_displays = [
            display
            for display in _flatten_displays(lidar_only["Visualization Manager"]["Displays"])
            if display.get("Name") in {"LivoxPointCloud", "LaserScan"}
        ]
        self.assertEqual(len(enabled_lidar_displays), len(lidar_displays))
        self.assertTrue(all(display["Enabled"] is True for display in enabled_lidar_displays))


def _flatten_displays(displays):
    for display in displays:
        yield display
        yield from _flatten_displays(display.get("Displays", []))


if __name__ == "__main__":
    unittest.main()
