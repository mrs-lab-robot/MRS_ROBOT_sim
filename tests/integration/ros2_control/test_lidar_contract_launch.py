#!/usr/bin/env python3
"""Regression coverage for the runtime LiDAR ROS 2 contract wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SIM_LAUNCH = (
    ROOT
    / "sim_runtime"
    / "ros2"
    / "openflex_isaac_sim"
    / "openflex_isaac_bringup"
    / "launch"
    / "sim.launch.py"
)


class _FakeAction:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class _FakeLaunchConfiguration:
    def __init__(self, name: str, **_kwargs) -> None:
        self.name = name

    def perform(self, context) -> str:
        return context.launch_configurations[self.name]


def _load_launch_module():
    launch_module = types.ModuleType("launch")
    launch_module.LaunchDescription = _FakeAction

    launch_actions = types.ModuleType("launch.actions")
    for name in (
        "DeclareLaunchArgument",
        "ExecuteProcess",
        "OpaqueFunction",
        "RegisterEventHandler",
        "SetEnvironmentVariable",
        "Shutdown",
        "TimerAction",
        "UnsetEnvironmentVariable",
    ):
        setattr(launch_actions, name, _FakeAction)

    launch_event_handlers = types.ModuleType("launch.event_handlers")
    launch_event_handlers.OnProcessExit = _FakeAction

    launch_substitutions = types.ModuleType("launch.substitutions")
    launch_substitutions.LaunchConfiguration = _FakeLaunchConfiguration

    launch_ros = types.ModuleType("launch_ros")
    launch_ros_actions = types.ModuleType("launch_ros.actions")
    launch_ros_actions.Node = _FakeAction
    launch_ros_parameters = types.ModuleType("launch_ros.parameter_descriptions")
    launch_ros_parameters.ParameterFile = _FakeAction
    launch_ros_parameters.ParameterValue = _FakeAction

    ament = types.ModuleType("ament_index_python")
    ament_packages = types.ModuleType("ament_index_python.packages")
    ament_packages.get_package_share_directory = lambda _name: "/tmp"
    ament_packages.get_package_prefix = lambda _name: "/tmp"

    modules = {
        "ament_index_python": ament,
        "ament_index_python.packages": ament_packages,
        "launch": launch_module,
        "launch.actions": launch_actions,
        "launch.event_handlers": launch_event_handlers,
        "launch.substitutions": launch_substitutions,
        "launch_ros": launch_ros,
        "launch_ros.actions": launch_ros_actions,
        "launch_ros.parameter_descriptions": launch_ros_parameters,
    }
    spec = importlib.util.spec_from_file_location("openflex_sim_launch", SIM_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class LidarContractLaunchTest(unittest.TestCase):
    def test_sim_launch_starts_custommsg_sensor_bridge_for_lio_and_rviz_topics(self) -> None:
        launch_module = _load_launch_module()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bringup_share = root / "bringup"
            (bringup_share / "config").mkdir(parents=True)
            (bringup_share / "config" / "controllers.isaac.mobile_base.yaml").touch()
            stage = bringup_share / "config" / "empty_stage.usd"
            stage.touch()
            scripts_share = root / "isaac_ros2_scripts"
            scripts_share.mkdir()
            isaac = root / "isaac"
            isaac.mkdir()
            (isaac / "python.sh").touch()
            prefix = root / "prefix"
            start_script = prefix / "lib" / "openflex_isaac_bringup" / "start_robot_control_sim.py"
            start_script.parent.mkdir(parents=True)
            start_script.touch()
            robot_usd = root / "openflex_robot.usda"
            robot_usd.touch()
            generated_urdf = root / "robot.urdf"
            sensor_assets = root / "sensor_assets"
            sensor_assets.mkdir()

            values = {
                "sensor_profile": "none",
                "lidar_mount_mode": "parented",
                "generated_urdf": str(generated_urdf),
                "robot_usd": str(robot_usd),
                "stage": str(stage),
                "use_sim_time": "false",
                "controller_use_sim_time": "false",
                "isaac_path": str(isaac),
                "api_host": "127.0.0.1",
                "api_port": "8086",
                "sensor_control_port": "8087",
                "physics_hz": "60",
                "render_hz": "30",
                "headless": "false",
                "sensor_asset_dir": str(sensor_assets),
                "lidar_transport": "historical-helper",
                "lidar_object_id_map": "",
                "x": "0",
                "y": "0",
                "z": "0",
                "roll": "0",
                "pitch": "0",
                "yaw": "0",
                "fixed": "false",
                "spawn_wait_timeout": "120",
                "state_topic": "/openflex/joint_states",
                "sim_ready_timeout": "120",
            }
            context = types.SimpleNamespace(launch_configurations=values)
            with (
                mock.patch.object(
                    launch_module,
                    "get_package_share_directory",
                    side_effect=lambda name: str(
                        bringup_share if name == "openflex_isaac_bringup" else scripts_share
                    ),
                ),
                mock.patch.object(
                    launch_module,
                    "get_package_prefix",
                    side_effect=lambda _name: str(prefix),
                ),
                mock.patch.object(launch_module, "_generate_sensor_free_urdf", return_value="<robot/>") ,
                mock.patch.object(launch_module, "_assert_api_port_available"),
            ):
                actions = launch_module.launch_setup(context)

        lidar_nodes = [
            action
            for action in actions
            if isinstance(action, _FakeAction)
            and action.kwargs.get("executable") == "isaacsim_compat_bridge.py"
        ]
        self.assertEqual(
            len(lidar_nodes),
            1,
            "the base launch must convert Isaac's raw cloud into the real Livox CustomMsg contract",
        )
        parameters = lidar_nodes[0].kwargs["parameters"][0]
        self.assertTrue(parameters["sensor_only_mode"])
        self.assertEqual(parameters["sim_lidar_topic"], "/openflex/livox_frame/lidar")
        self.assertEqual(parameters["livox_lidar_topic"], "/livox/lidar")
        self.assertEqual(parameters["livox_lidar_mode"], "custom")
        self.assertEqual(parameters["pointcloud_topics"], ["/livox/lidar_points"])
        self.assertEqual(parameters["scan_topics"], ["/scan"])
        self.assertFalse(parameters["publish_livox_imu"])
        self.assertFalse(parameters["publish_battery_state"])
        self.assertFalse(
            any(
                isinstance(action, _FakeAction)
                and action.kwargs.get("executable") == "lidar_contract_publisher.py"
                for action in actions
            ),
            "a second PointCloud2 publisher must not collide with /livox/lidar CustomMsg",
        )


if __name__ == "__main__":
    unittest.main()
