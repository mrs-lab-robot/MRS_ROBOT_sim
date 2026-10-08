#!/usr/bin/env python3
"""Run the Pico/Quest VR control path against the Isaac Sim controllers."""

import os
from pathlib import Path
import subprocess
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackagePrefix, FindPackageShare


def _start_vr_stack(context, nodes):
    if IfCondition(LaunchConfiguration("enable_arms")).evaluate(context):
        urdf_path = LaunchConfiguration("isaac_urdf").perform(context)
        if not os.path.isfile(urdf_path):
            raise RuntimeError(
                f"Isaac robot URDF not found at {urdf_path}; start the simulation first "
                "or pass isaac_urdf:=/absolute/path/to/robot_control_only.urdf"
            )
    if IfCondition(LaunchConfiguration("stop_existing_vr")).evaluate(context):
        cleanup_script = PathJoinSubstitution(
            [
                FindPackagePrefix("openflex_isaac_bringup"),
                "lib",
                "openflex_isaac_bringup",
                "stop_stale_vr_processes.py",
            ]
        ).perform(context)
        subprocess.run([sys.executable, cleanup_script], check=True)
    return nodes


def _default_isaac_urdf() -> str:
    generated_dir = os.environ.get("ISAACSIM_ROBOT_GENERATED_DIR", "").strip()
    if generated_dir:
        return str(Path(generated_dir).expanduser() / "robot_control_only.urdf")
    repository_root = os.environ.get("ISAACSIM_ROBOT_ROOT", "").strip()
    if repository_root:
        return str(
            Path(repository_root).expanduser()
            / "reports"
            / "runtime"
            / "generated"
            / "robot_control_only.urdf"
        )
    cache_root = os.environ.get("XDG_CACHE_HOME", "").strip()
    if cache_root:
        return str(Path(cache_root).expanduser() / "mrs_robot/isaacsim_generated/robot_control_only.urdf")
    return str(
        Path.home()
        / ".cache"
        / "mrs_robot"
        / "isaacsim_generated"
        / "robot_control_only.urdf"
    )


def generate_launch_description() -> LaunchDescription:
    isaac_urdf = LaunchConfiguration("isaac_urdf")
    arm_config = PathJoinSubstitution(
        [FindPackageShare("openarmx_teleop_vr"), "config", "teleop_params.yaml"]
    )

    arguments = [
        DeclareLaunchArgument(
            "isaac_urdf",
            default_value=_default_isaac_urdf(),
            description="Generated URDF used by the running Isaac robot and its VR IK node",
        ),
        DeclareLaunchArgument("listen_address", default_value="0.0.0.0"),
        DeclareLaunchArgument("listen_port", default_value="5100"),
        DeclareLaunchArgument("stop_existing_vr", default_value="true"),
        DeclareLaunchArgument("enable_arms", default_value="true"),
        DeclareLaunchArgument("enable_head", default_value="true"),
        DeclareLaunchArgument("enable_chassis", default_value="true"),
        DeclareLaunchArgument("enable_lift", default_value="true"),
        DeclareLaunchArgument("max_linear_speed", default_value="0.35"),
        DeclareLaunchArgument("boost_linear_speed", default_value="0.50"),
        DeclareLaunchArgument("max_angular_speed", default_value="0.60"),
        DeclareLaunchArgument("acceleration_time", default_value="2.0"),
    ]

    bridge = Node(
        package="openflex_vr_bridge",
        executable="pico_pose_bridge_node",
        name="pico_pose_bridge",
        output="screen",
        parameters=[
            {
                "listen_address": LaunchConfiguration("listen_address"),
                "listen_port": ParameterValue(LaunchConfiguration("listen_port"), value_type=int),
                "use_sim_time": False,
            }
        ],
    )
    arms = Node(
        package="openflex_isaac_bringup",
        executable="isaacsim_vr_arm_node.py",
        name="openarmx_teleop_vr_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_arms")),
        parameters=[
            arm_config,
            {
                # Isaac's ROS controllers always include the parallel gripper
                # joint, regardless of whether the VR pose source changes mode.
                "robot_type": "gripper",
                "urdf_path": isaac_urdf,
                "use_sim_time": False,
            },
        ],
    )
    head = Node(
        package="openarmx_head_teleop_vr_pico",
        executable="head_teleop_node",
        name="openarmx_head_teleop_vr_pico_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_head")),
        parameters=[{"use_sim_time": False, "publish_visualization_tf": False}],
    )
    chassis = Node(
        package="swerve_bringup",
        executable="vr_teleop_node",
        name="vr_teleop_chassis",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_chassis")),
        parameters=[
            {
                "cmd_vel_topic": "/cmd_vel",
                "max_linear_speed": ParameterValue(
                    LaunchConfiguration("max_linear_speed"), value_type=float
                ),
                "boost_linear_speed": ParameterValue(
                    LaunchConfiguration("boost_linear_speed"), value_type=float
                ),
                "max_angular_speed": ParameterValue(
                    LaunchConfiguration("max_angular_speed"), value_type=float
                ),
                "enable_scurve": True,
                "acceleration_time": ParameterValue(
                    LaunchConfiguration("acceleration_time"), value_type=float
                ),
                "smoothness": 0.5,
                "estop_toggle_topic": "",
                "use_vr_chassis_speed_config": False,
                "use_sim_time": False,
            }
        ],
    )
    lift = Node(
        package="swerve_bringup",
        executable="vr_lift_control_node",
        name="vr_lift_control",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_lift")),
        parameters=[
            {
                "jog_command_topic": "/lift_manual_position_controller/jog_command",
                "lift_speed": 0.05,
                "use_sim_time": False,
            }
        ],
    )

    start_vr_stack = OpaqueFunction(
        function=_start_vr_stack,
        args=[[bridge, arms, head, chassis, lift]],
    )
    return LaunchDescription([*arguments, start_vr_stack])
