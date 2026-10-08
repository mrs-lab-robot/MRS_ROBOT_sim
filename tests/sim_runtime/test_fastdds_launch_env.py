from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from launch import LaunchContext


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_FILE = (
    REPO_ROOT
    / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/launch/sim.launch.py"
)
RVIZ_LAUNCH_FILE = (
    REPO_ROOT
    / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/launch/rviz_only.launch.py"
)


def _load_launch_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIM_LAUNCH = _load_launch_module("openflex_sim_launch", LAUNCH_FILE)
RVIZ_LAUNCH = _load_launch_module("openflex_rviz_launch", RVIZ_LAUNCH_FILE)


class FastDDSEnvironmentTest(unittest.TestCase):
    def test_local_default_removes_inherited_remote_profile(self) -> None:
        for module in (SIM_LAUNCH, RVIZ_LAUNCH):
            with self.subTest(launch=module.__name__):
                local_context = LaunchContext()
                local_context.launch_configurations["fastdds_profiles_file"] = ""
                local_context.environment["FASTRTPS_DEFAULT_PROFILES_FILE"] = (
                    "/remote/fastdds.xml"
                )

                for action in module._apply_fastdds_profile(local_context):
                    action.execute(local_context)

                self.assertNotIn(
                    "FASTRTPS_DEFAULT_PROFILES_FILE", local_context.environment
                )

    def test_explicit_profile_replaces_inherited_remote_profile(self) -> None:
        for module in (SIM_LAUNCH, RVIZ_LAUNCH):
            with self.subTest(launch=module.__name__):
                local_context = LaunchContext()
                local_context.launch_configurations["fastdds_profiles_file"] = (
                    "/selected/fastdds.xml"
                )
                local_context.environment["FASTRTPS_DEFAULT_PROFILES_FILE"] = (
                    "/old/fastdds.xml"
                )

                for action in module._apply_fastdds_profile(local_context):
                    action.execute(local_context)

                self.assertEqual(
                    local_context.environment["FASTRTPS_DEFAULT_PROFILES_FILE"],
                    "/selected/fastdds.xml",
                )

    def test_local_sim_launch_overrides_a_remote_shell_to_local_only(self) -> None:
        context = LaunchContext()
        context.launch_configurations["ros_localhost_only"] = "1"
        context.environment["ROS_LOCALHOST_ONLY"] = "0"

        for action in SIM_LAUNCH._apply_ros_localhost_only(context):
            action.execute(context)

        self.assertEqual(context.environment["ROS_LOCALHOST_ONLY"], "1")

    def test_sim_launch_keeps_remote_discovery_an_explicit_override(self) -> None:
        context = LaunchContext()
        context.launch_configurations["ros_localhost_only"] = "0"
        context.environment["ROS_LOCALHOST_ONLY"] = "1"

        for action in SIM_LAUNCH._apply_ros_localhost_only(context):
            action.execute(context)

        self.assertEqual(context.environment["ROS_LOCALHOST_ONLY"], "0")


if __name__ == "__main__":
    unittest.main()
