#!/usr/bin/env python3
"""ros2_direct 修复项 C 补测用的模拟 VR 命令流发布节点。

在 A6000 无头环境下驱动 openarmx_leader/relay 链路的"GUI 等价"动作源：
真实 GUI 流程里这些命令话题由 openarmx_integrated_bringup 的 VR 节点
(pico_pose_bridge + openarmx_teleop_vr) 发布；无 Pico 时由本节点以相同
话题/字段/频率(20Hz)发布小幅度正弦命令，使 relay→sim→/joint_states→
LeRobot 直采的整条链路有活数据。

话题（与 relay_node.py / teleop config.py 默认一致）:
  /left_forward_position_controller/commands  Float64MultiArray(8: 7关节+手指)
  /right_forward_position_controller/commands Float64MultiArray(8)
  /head_forward_position_controller/commands  Float64MultiArray(2)
  /lift_position_controller/commands          Float64MultiArray(1)
  /cmd_vel                                    Twist(0)
  /pico_left_controller/pose, /pico_right_controller/pose  PoseStamped(保 deadman)

用法(A6000, 系统 python3.10 + humble):
  source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=49 ROS_LOCALHOST_ONLY=1
  python3 leader_driver.py
  # Ctrl+C 退出；录满 1 集后再退出
"""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

RATE_HZ = 20.0


class LeaderDriver(Node):
    def __init__(self) -> None:
        super().__init__('leader_driver_sim')
        self._left_pub = self.create_publisher(
            Float64MultiArray, '/left_forward_position_controller/commands', 10)
        self._right_pub = self.create_publisher(
            Float64MultiArray, '/right_forward_position_controller/commands', 10)
        self._head_pub = self.create_publisher(
            Float64MultiArray, '/head_forward_position_controller/commands', 10)
        self._lift_pub = self.create_publisher(
            Float64MultiArray, '/lift_position_controller/commands', 10)
        self._left_pose_pub = self.create_publisher(
            PoseStamped, '/pico_left_controller/pose', 10)
        self._right_pose_pub = self.create_publisher(
            PoseStamped, '/pico_right_controller/pose', 10)
        self._t0 = time.monotonic()
        self._seq = 0
        self.create_timer(1.0 / RATE_HZ, self._tick)
        self.get_logger().info('leader_driver ready @20Hz')

    def _sweep(self, amp: float, phase: float, period: float = 8.0) -> float:
        t = time.monotonic() - self._t0
        return amp * math.sin(2.0 * math.pi * t / period + phase)

    def _publish_array(self, publisher, values: list[float]) -> None:
        message = Float64MultiArray()
        message.data = [float(v) for v in values]
        publisher.publish(message)

    def _publish_pose(self, publisher) -> None:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = 'world'
        message.pose.orientation.w = 1.0
        publisher.publish(message)

    def _tick(self) -> None:
        self._seq += 1
        left = [self._sweep(0.08, 0.5 * index) for index in range(7)] + [0.0]
        right = [self._sweep(0.08, 0.5 * index + math.pi) for index in range(7)] + [0.0]
        finger = 0.02 + 0.012 * math.sin(2.0 * math.pi * (time.monotonic() - self._t0) / 4.0)
        left[7] = finger
        right[7] = finger
        head = [self._sweep(0.08, 0.0), self._sweep(0.04, 1.0)]
        lift = [0.0]
        self._publish_array(self._left_pub, left)
        self._publish_array(self._right_pub, right)
        self._publish_array(self._head_pub, head)
        self._publish_array(self._lift_pub, lift)
        self._publish_pose(self._left_pose_pub)
        self._publish_pose(self._right_pose_pub)


def main() -> None:
    rclpy.init()
    node = LeaderDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
