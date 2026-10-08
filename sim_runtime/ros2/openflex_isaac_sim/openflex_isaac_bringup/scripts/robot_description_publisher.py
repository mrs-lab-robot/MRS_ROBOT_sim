#!/usr/bin/env python3
"""Publish the generated robot URDF with transient-local durability."""

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class RobotDescriptionPublisher(Node):
    def __init__(self) -> None:
        super().__init__("openflex_robot_description_publisher")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("topic", "/robot_description")
        self.declare_parameter("republish_period_sec", 0.0)

        urdf_path = Path(str(self.get_parameter("urdf_path").value)).expanduser()
        topic = str(self.get_parameter("topic").value)
        period = float(self.get_parameter("republish_period_sec").value)
        if not urdf_path.is_file():
            raise RuntimeError(f"robot description URDF does not exist: {urdf_path}")

        self._message = String(data=urdf_path.read_text(encoding="utf-8"))
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(String, topic, qos)
        self._publisher.publish(self._message)
        self._timer = self.create_timer(period, self._publish) if period > 0.0 else None
        self.get_logger().info(
            f"Publishing {len(self._message.data)}-byte robot description on {topic} "
            f"from {urdf_path}"
        )

    def _publish(self) -> None:
        self._publisher.publish(self._message)


def main() -> None:
    rclpy.init()
    node = RobotDescriptionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
