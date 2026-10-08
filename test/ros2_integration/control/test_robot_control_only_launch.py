#!/usr/bin/env python3
"""Static checks for the standalone Isaac Sim 6 launch contract."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
BRINGUP = ROOT / "ros2_pkgs" / "openflex_isaac_sim" / "openflex_isaac_bringup"
SENSORS = ROOT / "ros2_pkgs" / "openflex_isaac_sim" / "openflex_isaac_sensors"
SIM_LAUNCH = BRINGUP / "launch" / "sim.launch.py"
START_SCRIPT = BRINGUP / "scripts" / "start_robot_control_sim.py"


class RobotControlLaunchTest(unittest.TestCase):
    def test_sim_launch_python_syntax_is_valid(self) -> None:
        source = SIM_LAUNCH.read_text(encoding="utf-8")
        compile(source, str(SIM_LAUNCH), "exec")

    def test_core_launch_is_sensor_free_and_has_runtime_sensor_lifecycle(self) -> None:
        text = SIM_LAUNCH.read_text(encoding="utf-8")
        start_text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('default_value="none"', text)
        self.assertIn('choices=["none"]', text)
        self.assertNotIn('executable="camera_contract_publisher.py"', text)
        self.assertIn('executable="isaacsim_compat_bridge.py"', text)
        self.assertIn('"sensor_only_mode": True', text)
        self.assertIn('"enable_sensors": False', text)
        self.assertIn('RobotSensorRuntime(', start_text)
        self.assertIn('_RUNTIME_SENSOR_MANAGER.mark_ready(error=sensor_catalog_error)', start_text)
        self.assertNotIn('create_robot_sensor_suite(', start_text)
        self.assertNotIn('bootstrap_camera_gates(', start_text)
        self.assertIn('sensor_profile = "none"', text)
        self.assertIn('default_value="parented"', text)

    def test_launch_keeps_rviz_separate_and_starts_vla_contract_bridge(self) -> None:
        text = SIM_LAUNCH.read_text(encoding="utf-8")

        self.assertNotIn('package="rviz2"', text)
        self.assertIn('executable="vla_contract_bridge.py"', text)
        self.assertNotIn('executable="camera_contract_publisher.py"', text)
        self.assertIn('executable="isaacsim_compat_bridge.py"', text)
        self.assertIn('"livox_lidar_mode": "custom"', text)
        self.assertTrue((BRINGUP / "launch" / "rviz_only.launch.py").is_file())

    def test_isaac_startup_enables_multitick_motion_bvh(self) -> None:
        text = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"--/rtx/hydra/supportMultiTickRate=true"', text)
        self.assertIn('"--/renderer/raytracingMotion/enabled=true"', text)
        self.assertIn('"--/renderer/multiGpu/enabled=false"', text)

    def test_mid360_profile_keeps_variant_cadence_and_performance_limits(self) -> None:
        text = (SENSORS / "openflex_isaac_sensors" / "mid360.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"Profile01_20Hz_0p5deg"', text)
        self.assertNotIn('"omni:sensor:Core:patternFiringRateHz": 20000', text)
        self.assertNotIn('"omni:sensor:Core:patternFiringRateHz": 36000', text)
        self.assertIn('"omni:sensor:Core:nearRangeM": 0.1', text)
        self.assertIn('"omni:sensor:Core:farRangeM": 40.0', text)
        self.assertIn('"OPENFLEX_ISAAC_LIDAR_CONFIG", "Example_Rotary"', text)
        self.assertIn('mount_mode: str = "parented"', text)

    def test_controller_target_is_90_hz(self) -> None:
        text = (BRINGUP / "config" / "controllers.isaac.mobile_base.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("update_rate: 90", text)


if __name__ == "__main__":
    unittest.main()
