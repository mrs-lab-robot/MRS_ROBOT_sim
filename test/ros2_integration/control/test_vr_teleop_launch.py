#!/usr/bin/env python3
"""Static contract tests for the Isaac-specific VR teleoperation launch."""

from pathlib import Path
import unittest


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_DIR / "ros2_pkgs" / "openflex_isaac_sim" / "openflex_isaac_bringup"
LAUNCH_FILE = PACKAGE_DIR / "launch" / "vr_teleop.launch.py"


class VrTeleopLaunchTest(unittest.TestCase):
    def test_isaac_arm_adapter_is_executable(self) -> None:
        adapter = PACKAGE_DIR / "scripts" / "isaacsim_vr_arm_node.py"

        self.assertTrue(adapter.stat().st_mode & 0o111)

    def test_launch_starts_complete_vr_control_path(self) -> None:
        script = LAUNCH_FILE.read_text(encoding="utf-8")

        for package, executable in (
            ("openflex_vr_bridge", "pico_pose_bridge_node"),
            ("openflex_isaac_bringup", "isaacsim_vr_arm_node.py"),
            ("openarmx_head_teleop_vr_pico", "head_teleop_node"),
            ("swerve_bringup", "vr_teleop_node"),
            ("swerve_bringup", "vr_lift_control_node"),
        ):
            with self.subTest(package=package, executable=executable):
                self.assertIn(f'package="{package}"', script)
                self.assertIn(f'executable="{executable}"', script)

    def test_isaac_arm_adapter_preserves_the_working_hardware_ik_frames(self) -> None:
        adapter = PACKAGE_DIR / "scripts" / "isaacsim_vr_arm_node.py"
        script = LAUNCH_FILE.read_text(encoding="utf-8")

        self.assertTrue(adapter.is_file())
        adapter_text = adapter.read_text(encoding="utf-8")
        self.assertIn('"openarmx_left_link7_pico"', adapter_text)
        self.assertIn('"openarmx_right_link7_pico"', adapter_text)
        self.assertNotIn('"openarmx_left_hand_tcp"', adapter_text)
        self.assertNotIn('"openarmx_right_hand_tcp"', adapter_text)
        self.assertIn('"urdf_path": isaac_urdf', script)

    def test_isaac_keeps_the_parallel_gripper_in_every_pose_mode(self) -> None:
        script = LAUNCH_FILE.read_text(encoding="utf-8")
        controller_config = (
            PACKAGE_DIR / "config" / "controllers.isaac.mobile_base.yaml"
        ).read_text(encoding="utf-8")

        self.assertIn('"robot_type": "gripper"', script)
        self.assertIn("openarmx_left_finger_joint1", controller_config)
        self.assertIn("openarmx_right_finger_joint1", controller_config)

    def test_launch_uses_isaac_urdf_and_conservative_chassis_defaults(self) -> None:
        script = LAUNCH_FILE.read_text(encoding="utf-8")

        self.assertIn('"ISAACSIM_ROBOT_GENERATED_DIR"', script)
        self.assertIn('"robot_control_only.urdf"', script)
        self.assertIn('LaunchConfiguration("isaac_urdf")', script)
        self.assertNotIn('"openflex_isaac_robot.urdf"', script)
        self.assertIn('DeclareLaunchArgument("max_linear_speed", default_value="0.35")', script)
        self.assertIn('DeclareLaunchArgument("boost_linear_speed", default_value="0.50")', script)
        self.assertIn('DeclareLaunchArgument("max_angular_speed", default_value="0.60")', script)
        self.assertIn('DeclareLaunchArgument("acceleration_time", default_value="2.0")', script)
        self.assertIn('DeclareLaunchArgument("listen_port", default_value="5100")', script)

    def test_launch_checks_generated_urdf_before_stopping_existing_vr_nodes(self) -> None:
        script = LAUNCH_FILE.read_text(encoding="utf-8")
        start = script[script.index("def _start_vr_stack"):script.index("def generate_launch_description")]

        self.assertIn("os.path.isfile(urdf_path)", start)
        self.assertIn("start the simulation first", start.lower())
        self.assertLess(start.index("os.path.isfile(urdf_path)"), start.index("stop_existing_vr"))

    def test_launch_stops_existing_vr_stack_before_creating_nodes(self) -> None:
        script = LAUNCH_FILE.read_text(encoding="utf-8")

        self.assertIn(
            'DeclareLaunchArgument("stop_existing_vr", default_value="true")',
            script,
        )
        self.assertIn('"stop_stale_vr_processes.py"', script)
        self.assertIn("function=_start_vr_stack", script)


if __name__ == "__main__":
    unittest.main()
