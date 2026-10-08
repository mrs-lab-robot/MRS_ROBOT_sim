#!/usr/bin/env python3
"""Launch only the OpenFleX RViz interface."""

from __future__ import annotations

import atexit
import copy
import os
from pathlib import Path
import tempfile

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetEnvironmentVariable,
    UnsetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _rviz_display_error(qt_qpa_platform: str) -> str | None:
    if qt_qpa_platform.strip():
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    if os.environ.get("QT_QPA_PLATFORM"):
        return None
    return (
        "RViz needs a GUI display, but DISPLAY and WAYLAND_DISPLAY are unset. "
        "Run this launch from a desktop terminal or SSH with X forwarding, or pass "
        "qt_qpa_platform:=offscreen/vnc for non-interactive diagnostics."
    )


def _rviz_node_environment(
    qt_qpa_platform: str,
    simulation_mode: str = "true",
    joint_states_topic: str = "/joint_states",
) -> dict[str, str]:
    value = qt_qpa_platform.strip()
    environment = {
        "OPENFLEX_LIFT_SIMULATION_MODE": simulation_mode.strip().lower(),
        "OPENFLEX_LIFT_JOINT_STATES_TOPIC": joint_states_topic.strip() or "/joint_states",
    }
    if value:
        environment["QT_QPA_PLATFORM"] = value
    return environment


def _fastdds_profile_action(profile_path: str):
    profile_path = profile_path.strip()
    if profile_path:
        return SetEnvironmentVariable(
            name="FASTRTPS_DEFAULT_PROFILES_FILE",
            value=profile_path,
        )
    return UnsetEnvironmentVariable(name="FASTRTPS_DEFAULT_PROFILES_FILE")


def _apply_sensor_options(config: dict, show_cameras: bool, show_lidar: bool) -> dict:
    """Return a per-run RViz config with only the requested sensor panels enabled."""
    configured = copy.deepcopy(config)
    panel = next(
        (
            item
            for item in configured.get("Panels", [])
            if item.get("Class")
            == "openflex_isaac_bringup/SimulationControlSensorsPanel"
        ),
        None,
    )
    if panel is None:
        raise RuntimeError("RViz config is missing SimulationControlSensorsPanel")
    panel["Show Cameras"] = bool(show_cameras)
    panel["Show Lidar"] = bool(show_lidar)

    def update_lidar_displays(displays):
        for display in displays:
            if display.get("Name") in {"LivoxPointCloud", "LaserScan"}:
                display["Enabled"] = bool(show_lidar)
            update_lidar_displays(display.get("Displays", []))

    update_lidar_displays(
        configured.get("Visualization Manager", {}).get("Displays", [])
    )
    return configured


def _write_runtime_rviz_config(source: Path, show_cameras: bool, show_lidar: bool) -> Path:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = _apply_sensor_options(config, show_cameras, show_lidar)
    fd, temporary_path = tempfile.mkstemp(prefix="openflex_sim_rviz_", suffix=".rviz")
    path = Path(temporary_path)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
    atexit.register(lambda path=path: path.unlink(missing_ok=True))
    return path


def _apply_fastdds_profile(context, *args, **kwargs):
    profile_path = LaunchConfiguration("fastdds_profiles_file").perform(context)
    return [_fastdds_profile_action(profile_path)]


def launch_setup(context, *args, **kwargs):
    bringup_share = Path(get_package_share_directory("openflex_isaac_bringup"))
    show_cameras = LaunchConfiguration("show_cameras").perform(context).strip().lower() in {
        "1", "true", "yes", "on"
    }
    show_lidar = LaunchConfiguration("show_lidar").perform(context).strip().lower() in {
        "1", "true", "yes", "on"
    }
    rviz_config_source = bringup_share / "rviz" / "robot_control_only.rviz"
    rviz_config = _write_runtime_rviz_config(
        rviz_config_source, show_cameras, show_lidar
    )
    qt_qpa_platform = LaunchConfiguration("qt_qpa_platform").perform(context)
    simulation_mode = LaunchConfiguration("simulation_mode").perform(context)
    joint_states_topic = LaunchConfiguration("joint_states_topic").perform(context)

    if not rviz_config.is_file():
        raise RuntimeError(f"RViz config does not exist: {rviz_config}")
    display_error = _rviz_display_error(qt_qpa_platform)
    if display_error is not None:
        raise RuntimeError(display_error)

    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="both",
            arguments=["-d", str(rviz_config)],
            additional_env=_rviz_node_environment(
                qt_qpa_platform, simulation_mode, joint_states_topic
            ),
            parameters=[
                {
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "pos_min": ParameterValue(LaunchConfiguration("pos_min"), value_type=float),
                    "pos_max": ParameterValue(LaunchConfiguration("pos_max"), value_type=float),
                    "simulation_mode": ParameterValue(
                        LaunchConfiguration("simulation_mode"), value_type=bool
                    ),
                    "joint_states_topic": LaunchConfiguration("joint_states_topic"),
                    "position_command_topic": LaunchConfiguration("position_command_topic"),
                }
            ],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use the simulation clock when RViz is attached to an Isaac Sim session.",
            ),
            DeclareLaunchArgument(
                "qt_qpa_platform",
                default_value="",
                description=(
                    "Optional Qt platform override for RViz, e.g. offscreen or vnc. "
                    "Leave empty for normal desktop xcb/Wayland detection."
                ),
            ),
            DeclareLaunchArgument(
                "pos_min",
                default_value="-0.65",
                description="Minimum simulated lift_joint position in metres.",
            ),
            DeclareLaunchArgument(
                "pos_max",
                default_value="0.3",
                description="Maximum simulated lift_joint position in metres.",
            ),
            DeclareLaunchArgument("simulation_mode", default_value="true"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument(
                "show_cameras",
                default_value="false",
                description="Show camera tabs only for camera topics that publish images.",
            ),
            DeclareLaunchArgument(
                "show_lidar",
                default_value="false",
                description="Enable LiDAR RViz displays and the optional radar tab.",
            ),
            DeclareLaunchArgument(
                "position_command_topic",
                default_value="/lift_position_controller/commands",
            ),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="49",
                description="ROS 2 domain of the running Isaac Sim session.",
            ),
            DeclareLaunchArgument(
                "ros_localhost_only",
                default_value="1",
                description=(
                    "ROS_LOCALHOST_ONLY value used by the Isaac Sim session. "
                    "The default local-only mode matches the simulation launch environment."
                ),
            ),
            DeclareLaunchArgument(
                "fastdds_profiles_file",
                default_value="",
                description=(
                    "Fast DDS XML profile file; empty selects local/default discovery "
                    "and clears any inherited remote-only profile."
                ),
            ),
            SetEnvironmentVariable(
                name="ROS_DOMAIN_ID",
                value=LaunchConfiguration("ros_domain_id"),
            ),
            SetEnvironmentVariable(
                name="ROS_LOCALHOST_ONLY",
                value=LaunchConfiguration("ros_localhost_only"),
            ),
            OpaqueFunction(function=_apply_fastdds_profile),
            OpaqueFunction(function=launch_setup),
        ]
    )
