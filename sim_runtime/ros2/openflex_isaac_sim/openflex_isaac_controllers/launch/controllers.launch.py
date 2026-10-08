#!/usr/bin/env python3
"""Launch ros2_control controllers for Isaac Sim."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Generate launch description for controllers."""

    # Get config file path
    config_file = PathJoinSubstitution([
        FindPackageShare('openflex_isaac_controllers'),
        'config',
        'ros2_control.yaml'
    ])

    # Controller manager
    controller_manager_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[config_file],
    )

    # Joint state broadcaster (start immediately)
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Left arm controller (delayed start)
    left_arm_controller_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['left_arm_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    # Right arm controller (delayed start)
    right_arm_controller_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['right_arm_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    # Lift controller (delayed start)
    lift_controller_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['lift_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    # Head controller (delayed start)
    head_controller_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['head_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    # Velocity controller (delayed start)
    velocity_controller_spawner = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['velocity_controller', '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    # Joint command bridge
    joint_command_bridge = Node(
        package='openflex_isaac_controllers',
        executable='joint_command_bridge',
        name='joint_command_bridge',
        output='screen',
    )

    # Velocity controller bridge
    velocity_controller_bridge = Node(
        package='openflex_isaac_controllers',
        executable='velocity_controller',
        name='velocity_controller',
        output='screen',
    )

    return LaunchDescription([
        controller_manager_node,
        joint_state_broadcaster_spawner,
        left_arm_controller_spawner,
        right_arm_controller_spawner,
        lift_controller_spawner,
        head_controller_spawner,
        velocity_controller_spawner,
        joint_command_bridge,
        velocity_controller_bridge,
    ])
