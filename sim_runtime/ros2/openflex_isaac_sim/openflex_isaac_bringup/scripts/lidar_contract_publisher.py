#!/usr/bin/env python3
"""Publish the real-robot MID360 PointCloud2 and LaserScan contracts."""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2


class LidarContractPublisher(Node):
    def __init__(self) -> None:
        super().__init__("openflex_lidar_contract_publisher")
        self.declare_parameter("input_topic", "/openflex/livox_frame/lidar")
        self.declare_parameter("pointcloud_topic", "/livox/lidar")
        self.declare_parameter("pointcloud_alias_topics", ["/livox/lidar_points"])
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("frame_id", "livox_frame")
        self.declare_parameter("scan_angle_min", -math.pi)
        self.declare_parameter("scan_angle_max", math.pi)
        self.declare_parameter("scan_angle_increment", 0.0058)
        self.declare_parameter("scan_range_min", 0.05)
        self.declare_parameter("scan_range_max", 20.0)
        self.declare_parameter("scan_height_min", -0.35)
        self.declare_parameter("scan_height_max", 1.5)

        self._frame_id = str(self.get_parameter("frame_id").value)
        pointcloud_topics = [str(self.get_parameter("pointcloud_topic").value)]
        pointcloud_topics.extend(
            str(topic)
            for topic in self.get_parameter("pointcloud_alias_topics").value
            if str(topic)
        )
        self._pointcloud_publishers = [
            self.create_publisher(PointCloud2, topic, qos_profile_sensor_data)
            for topic in dict.fromkeys(pointcloud_topics)
        ]
        self._scan_publisher = self.create_publisher(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            qos_profile_sensor_data,
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._on_cloud,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            "OpenFleX LiDAR contract publisher started: "
            f"{self.get_parameter('input_topic').value} -> "
            f"{', '.join(dict.fromkeys(pointcloud_topics))}, "
            f"{self.get_parameter('scan_topic').value}"
        )

    @staticmethod
    def _field_offsets(message: PointCloud2) -> dict[str, int]:
        return {field.name: int(field.offset) for field in message.fields}

    @staticmethod
    def _field_view(message: PointCloud2, offset: int) -> np.ndarray:
        byte_order = ">" if message.is_bigendian else "<"
        count = int(message.width) * int(message.height)
        return np.ndarray(
            shape=(count,),
            dtype=np.dtype(byte_order + "f4"),
            buffer=message.data,
            offset=offset,
            strides=(int(message.point_step),),
        )

    def _on_cloud(self, message: PointCloud2) -> None:
        if self._frame_id:
            message.header.frame_id = self._frame_id
        for publisher in self._pointcloud_publishers:
            publisher.publish(message)
        if self._scan_publisher.get_subscription_count() == 0:
            return

        offsets = self._field_offsets(message)
        if not {"x", "y", "z"}.issubset(offsets):
            self.get_logger().error("MID360 PointCloud2 is missing x/y/z fields")
            return

        x = self._field_view(message, offsets["x"])
        y = self._field_view(message, offsets["y"])
        z = self._field_view(message, offsets["z"])
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

        angle_min = float(self.get_parameter("scan_angle_min").value)
        angle_max = float(self.get_parameter("scan_angle_max").value)
        angle_increment = max(
            1.0e-6, float(self.get_parameter("scan_angle_increment").value)
        )
        range_min = float(self.get_parameter("scan_range_min").value)
        range_max = float(self.get_parameter("scan_range_max").value)
        height_min = float(self.get_parameter("scan_height_min").value)
        height_max = float(self.get_parameter("scan_height_max").value)

        distances = np.hypot(x, y)
        angles = np.arctan2(y, x)
        valid = (
            finite
            & (z >= height_min)
            & (z <= height_max)
            & (distances >= range_min)
            & (distances <= range_max)
            & (angles >= angle_min)
            & (angles < angle_max)
        )
        bin_count = max(1, int(math.ceil((angle_max - angle_min) / angle_increment)))
        ranges = np.full(bin_count, np.inf, dtype=np.float32)
        if np.any(valid):
            indices = ((angles[valid] - angle_min) / angle_increment).astype(np.int64)
            inside = (indices >= 0) & (indices < bin_count)
            np.minimum.at(ranges, indices[inside], distances[valid][inside])

        scan = LaserScan()
        scan.header = message.header
        scan.angle_min = angle_min
        scan.angle_max = angle_min + bin_count * angle_increment
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = range_min
        scan.range_max = range_max
        scan.ranges = ranges.tolist()
        self._scan_publisher.publish(scan)


def main() -> None:
    rclpy.init()
    node = LidarContractPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
