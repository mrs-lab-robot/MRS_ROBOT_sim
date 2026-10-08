#!/usr/bin/env python3
"""
Joint Command Bridge - Bridges ROS2 control commands to Isaac Sim.

This node receives joint trajectory commands from ros2_control and
forwards them to Isaac Sim's articulation API.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer
import threading


class JointCommandBridge(Node):
    """Bridge between ros2_control and Isaac Sim joint commands."""

    def __init__(self):
        super().__init__('joint_command_bridge')

        self.get_logger().info('Initializing Joint Command Bridge')

        # Joint names for the full robot
        self.joint_names = []
        self.current_positions = {}
        self.current_velocities = {}
        self.lock = threading.Lock()

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/isaac/joint_states',
            self.joint_state_callback,
            10
        )

        # Publishers - send commands to Isaac Sim
        self.joint_command_pub = self.create_publisher(
            JointState,
            '/isaac/joint_commands',
            10
        )

        # Subscribe to trajectory commands from controllers
        self.trajectory_sub = self.create_subscription(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            self.trajectory_callback,
            10
        )

        # Action server for FollowJointTrajectory
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            self.execute_trajectory
        )

        self.get_logger().info('Joint Command Bridge initialized')

    def joint_state_callback(self, msg):
        """Update current joint states from Isaac Sim."""
        with self.lock:
            for i, name in enumerate(msg.name):
                if i < len(msg.position):
                    self.current_positions[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.current_velocities[name] = msg.velocity[i]

    def trajectory_callback(self, msg):
        """Handle incoming trajectory commands."""
        if not msg.points:
            return

        # Use the last point as the target position
        target_point = msg.points[-1]

        # Create joint command message
        cmd_msg = JointState()
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
        cmd_msg.name = msg.joint_names
        cmd_msg.position = list(target_point.positions)

        if target_point.velocities:
            cmd_msg.velocity = list(target_point.velocities)

        self.joint_command_pub.publish(cmd_msg)
        self.get_logger().debug(f'Published joint commands for {len(msg.joint_names)} joints')

    def execute_trajectory(self, goal_handle):
        """Execute a trajectory action."""
        self.get_logger().info('Executing trajectory action')

        trajectory = goal_handle.request.trajectory

        # Execute each point in the trajectory
        for point in trajectory.points:
            cmd_msg = JointState()
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            cmd_msg.name = trajectory.joint_names
            cmd_msg.position = list(point.positions)

            if point.velocities:
                cmd_msg.velocity = list(point.velocities)

            self.joint_command_pub.publish(cmd_msg)

            # Sleep for the time_from_start duration
            # (simplified - in real implementation, use proper timing)
            if point.time_from_start.sec > 0 or point.time_from_start.nanosec > 0:
                sleep_time = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
                rclpy.spin_once(self, timeout_sec=min(sleep_time, 0.1))

        goal_handle.succeed()

        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
