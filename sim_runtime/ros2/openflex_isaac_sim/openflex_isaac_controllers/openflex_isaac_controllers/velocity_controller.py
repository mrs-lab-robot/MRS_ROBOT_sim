#!/usr/bin/env python3
"""
Velocity Controller - Handles base velocity commands for mobile chassis.

Converts twist commands to wheel velocities for the swerve drive system.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float64MultiArray
import numpy as np


class VelocityController(Node):
    """Controller for chassis velocity commands."""

    def __init__(self):
        super().__init__('velocity_controller')

        self.get_logger().info('Initializing Velocity Controller')

        # Parameters
        self.declare_parameter('wheel_base', 0.5)  # Distance between front and rear wheels
        self.declare_parameter('wheel_track', 0.5)  # Distance between left and right wheels
        self.declare_parameter('wheel_radius', 0.1)

        self.wheel_base = self.get_parameter('wheel_base').value
        self.wheel_track = self.get_parameter('wheel_track').value
        self.wheel_radius = self.get_parameter('wheel_radius').value

        # Subscriber for cmd_vel
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Publisher for wheel velocities to Isaac Sim
        self.wheel_vel_pub = self.create_publisher(
            Float64MultiArray,
            '/isaac/wheel_velocities',
            10
        )

        # Publisher for velocity feedback
        self.vel_feedback_pub = self.create_publisher(
            TwistStamped,
            '/velocity_controller/state',
            10
        )

        self.get_logger().info('Velocity Controller initialized')
        self.get_logger().info(f'Wheel base: {self.wheel_base}m, Track: {self.wheel_track}m')

    def cmd_vel_callback(self, msg):
        """Convert twist command to wheel velocities."""
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        angular_z = msg.angular.z

        # Swerve drive kinematics (simplified)
        # Each wheel can steer and drive independently
        # For now, use differential drive approximation

        # Calculate wheel velocities for 4-wheel swerve
        # Wheel positions: FL, FR, RL, RR
        l_x = self.wheel_base / 2.0
        l_y = self.wheel_track / 2.0

        # Velocity at each wheel (before converting to angular velocity)
        v_fl_x = linear_x - angular_z * l_y
        v_fl_y = linear_y + angular_z * l_x

        v_fr_x = linear_x + angular_z * l_y
        v_fr_y = linear_y + angular_z * l_x

        v_rl_x = linear_x - angular_z * l_y
        v_rl_y = linear_y - angular_z * l_x

        v_rr_x = linear_x + angular_z * l_y
        v_rr_y = linear_y - angular_z * l_x

        # Magnitude and angle for each wheel
        v_fl = np.sqrt(v_fl_x**2 + v_fl_y**2)
        v_fr = np.sqrt(v_fr_x**2 + v_fr_y**2)
        v_rl = np.sqrt(v_rl_x**2 + v_rl_y**2)
        v_rr = np.sqrt(v_rr_x**2 + v_rr_y**2)

        # Convert to angular velocities (rad/s)
        omega_fl = v_fl / self.wheel_radius
        omega_fr = v_fr / self.wheel_radius
        omega_rl = v_rl / self.wheel_radius
        omega_rr = v_rr / self.wheel_radius

        # Steering angles
        theta_fl = np.arctan2(v_fl_y, v_fl_x) if v_fl > 0.001 else 0.0
        theta_fr = np.arctan2(v_fr_y, v_fr_x) if v_fr > 0.001 else 0.0
        theta_rl = np.arctan2(v_rl_y, v_rl_x) if v_rl > 0.001 else 0.0
        theta_rr = np.arctan2(v_rr_y, v_rr_x) if v_rr > 0.001 else 0.0

        # Publish wheel commands
        wheel_cmd = Float64MultiArray()
        wheel_cmd.data = [
            omega_fl, theta_fl,  # FL: driving velocity, steering angle
            omega_fr, theta_fr,  # FR
            omega_rl, theta_rl,  # RL
            omega_rr, theta_rr,  # RR
        ]

        self.wheel_vel_pub.publish(wheel_cmd)

        # Publish velocity feedback
        feedback = TwistStamped()
        feedback.header.stamp = self.get_clock().now().to_msg()
        feedback.header.frame_id = 'base_link'
        feedback.twist = msg
        self.vel_feedback_pub.publish(feedback)

        self.get_logger().debug(
            f'Cmd: vx={linear_x:.2f}, vy={linear_y:.2f}, wz={angular_z:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VelocityController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
