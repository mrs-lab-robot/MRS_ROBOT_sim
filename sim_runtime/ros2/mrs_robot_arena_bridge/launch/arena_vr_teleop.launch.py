"""Launch VR input/solvers and the Arena relay, without starting robot hardware."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    vr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("openarmx_integrated_bringup"),
                "launch",
                "integrated_vr_teleop.launch.py",
            ])
        ),
        launch_arguments={
            "enable_joystick_control": "true",
            "enable_button_lift": "true",
            "enable_head_teleop": "true",
            "enable_waist_control": "false",
        }.items(),
    )
    relay = Node(
        package="mrs_robot_arena_bridge",
        executable="arena_relay",
        name="mrs_robot_arena_relay",
        output="screen",
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("mrs_robot_arena_bridge"), "config", "arena_vr_teleop.yaml"
            ]),
            {
                "arena_host": LaunchConfiguration("arena_host"),
                "state_bind_host": LaunchConfiguration("state_bind_host"),
                "camera_bind_host": LaunchConfiguration("camera_bind_host"),
            },
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument("arena_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("state_bind_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("camera_bind_host", default_value="127.0.0.1"),
        vr_launch,
        relay,
    ])
