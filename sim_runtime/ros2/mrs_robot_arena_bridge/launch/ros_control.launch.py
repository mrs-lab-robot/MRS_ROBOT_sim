"""Start ROS command relay, robot TF, and the existing OpenFlex RViz controls."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    relay = Node(
        package="mrs_robot_arena_bridge",
        executable="arena_relay",
        name="mrs_robot_arena_relay",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("mrs_robot_arena_bridge"), "config", "ros_control.yaml"
            ]),
            {
                "control_mode": LaunchConfiguration("control_mode"),
                "arena_host": LaunchConfiguration("sim_host"),
                "command_port": LaunchConfiguration("command_port"),
                "state_port": LaunchConfiguration("state_port"),
                "deadman_timeout": LaunchConfiguration("deadman_timeout"),
            },
        ],
    )
    robot_description = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(
                    Command([
                        "xacro ",
                        PathJoinSubstitution([
                            FindPackageShare("openflex_isaac_description"),
                            "urdf",
                            "openflex_robot.urdf.xacro",
                        ]),
                    ]),
                    value_type=str,
                ),
                "use_sim_time": False,
            }
        ],
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("openflex_isaac_bringup"), "launch", "rviz_only.launch.py"
            ])
        ),
        launch_arguments={
            "use_sim_time": "false",
            "simulation_mode": "true",
            "ros_domain_id": LaunchConfiguration("ros_domain_id"),
            "ros_localhost_only": LaunchConfiguration("ros_localhost_only"),
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("control_mode", default_value="ros", choices=["ros", "vr"]),
            DeclareLaunchArgument("sim_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("command_port", default_value="24102"),
            DeclareLaunchArgument("state_port", default_value="24103"),
            DeclareLaunchArgument("deadman_timeout", default_value="0.25"),
            DeclareLaunchArgument("ros_domain_id", default_value="49"),
            DeclareLaunchArgument("ros_localhost_only", default_value="1"),
            SetEnvironmentVariable("ROS_DOMAIN_ID", LaunchConfiguration("ros_domain_id")),
            SetEnvironmentVariable("ROS_LOCALHOST_ONLY", LaunchConfiguration("ros_localhost_only")),
            robot_description,
            relay,
            rviz,
        ]
    )
