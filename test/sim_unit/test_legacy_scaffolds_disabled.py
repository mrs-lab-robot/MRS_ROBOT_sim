from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[2]
ROS_ROOT = ROOT / "sim_runtime" / "ros2" / "openflex_isaac_sim"
SENSORS_ROOT = ROS_ROOT / "openflex_isaac_sensors"
BRIDGE_ROOT = ROS_ROOT / "openflex_isaac_bridge"

sys.path.insert(0, str(SENSORS_ROOT))
sys.path.insert(0, str(BRIDGE_ROOT))


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LegacyExecutableEntryPointTest(unittest.TestCase):
    def test_sensor_entrypoints_fail_before_initializing_ros_or_a_publisher(self):
        cases = (
            ("openflex_isaac_sensors.camera_publisher", "CameraPublisher"),
            ("openflex_isaac_sensors.lidar_publisher", "LidarPublisher"),
            ("openflex_isaac_sensors.imu_publisher", "ImuPublisher"),
        )
        for module_name, node_class in cases:
            with self.subTest(module=module_name):
                module = __import__(module_name, fromlist=["main"])
                ros = MagicMock()
                node_constructor = MagicMock()
                with patch.object(module, "rclpy", ros, create=True), patch.object(
                    module, node_class, node_constructor, create=True
                ):
                    with self.assertRaisesRegex(RuntimeError, "sim.launch.py"):
                        module.main()
                ros.init.assert_not_called()
                node_constructor.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "sim.launch.py"):
                    getattr(module, node_class)()

    def test_legacy_bridge_entrypoint_fails_before_advertising_services(self):
        module = _load_module(
            "legacy_sim_bridge_under_test",
            BRIDGE_ROOT / "openflex_isaac_bridge" / "sim_bridge_node.py",
        )
        ros = MagicMock()
        node_constructor = MagicMock()
        with patch.object(module, "rclpy", ros, create=True), patch.object(
            module, "IsaacSimBridgeNode", node_constructor, create=True
        ):
            with self.assertRaisesRegex(RuntimeError, "sim.launch.py"):
                module.main()
        ros.init.assert_not_called()
        node_constructor.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "sim.launch.py"):
            module.IsaacSimBridgeNode()


class LegacyLaunchEntryPointTest(unittest.TestCase):
    def test_legacy_launches_fail_during_description_generation(self):
        paths = (
            SENSORS_ROOT / "launch" / "sensors.launch.py",
            ROS_ROOT / "openflex_isaac_bringup" / "launch" / "isaac_sim.launch.py",
            BRIDGE_ROOT / "launch" / "bridge.launch.py",
        )
        for path in paths:
            with self.subTest(launch=path.name):
                module = _load_module(f"legacy_{path.stem}_under_test", path)
                with self.assertRaisesRegex(RuntimeError, "sim.launch.py"):
                    module.generate_launch_description()


if __name__ == "__main__":
    unittest.main()
