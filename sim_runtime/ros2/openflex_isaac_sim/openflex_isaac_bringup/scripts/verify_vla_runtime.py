#!/usr/bin/env python3
"""Verify the Isaac Sim runtime interfaces required by OpenFleX VLA."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import time

from PIL import Image as PILImage
import rclpy
from livox_ros_driver2.msg import CustomMsg
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu, JointState, LaserScan, PointCloud2


EXPECTED_TYPES = {
    "/joint_states": "sensor_msgs/msg/JointState",
    "/fastlio2/lio_odom": "nav_msgs/msg/Odometry",
    "/livox/lidar": "livox_ros_driver2/msg/CustomMsg",
    "/livox/lidar_points": "sensor_msgs/msg/PointCloud2",
    "/scan": "sensor_msgs/msg/LaserScan",
    "/livox/imu": "sensor_msgs/msg/Imu",
    "/cam_left/color/image/compressed": "sensor_msgs/msg/CompressedImage",
    "/cam_right/color/image/compressed": "sensor_msgs/msg/CompressedImage",
    "/cam_head/color/image/compressed": "sensor_msgs/msg/CompressedImage",
    "/cam_base/color/image/compressed": "sensor_msgs/msg/CompressedImage",
    "/left_forward_position_controller/commands": "std_msgs/msg/Float64MultiArray",
    "/right_forward_position_controller/commands": "std_msgs/msg/Float64MultiArray",
    "/head_forward_position_controller/commands": "std_msgs/msg/Float64MultiArray",
    "/lift_position_controller/commands": "std_msgs/msg/Float64MultiArray",
    "/velocity_controller/commands": "std_msgs/msg/Float64MultiArray",
    "/cmd_vel": "geometry_msgs/msg/Twist",
}

EXPECTED_IMAGE_SIZES = {
    "/cam_left/color/image/compressed": (320, 240),
    "/cam_right/color/image/compressed": (320, 240),
    "/cam_head/color/image/compressed": (640, 480),
    "/cam_base/color/image/compressed": (320, 240),
}

REQUIRED_JOINTS = {
    "lift_joint",
    "openarmx_head_yaw_joint",
    "openarmx_head_pitch_joint",
    *(f"openarmx_left_joint{index}" for index in range(1, 8)),
    "openarmx_left_finger_joint1",
    *(f"openarmx_right_joint{index}" for index in range(1, 8)),
    "openarmx_right_finger_joint1",
}


class VlaRuntimeVerifier(Node):
    def __init__(self) -> None:
        super().__init__("openflex_vla_runtime_verifier")
        self.joint_state: JointState | None = None
        self.odom: Odometry | None = None
        self.livox_lidar: CustomMsg | None = None
        self.lidar_points: PointCloud2 | None = None
        self.scan: LaserScan | None = None
        self.imu: Imu | None = None
        self.livox_point_counts: list[int] = []
        self.livox_times: list[float] = []
        self.pointcloud_counts: list[int] = []
        self.pointcloud_times: list[float] = []
        self.images: dict[str, dict] = {}
        self.image_counts = {topic: 0 for topic in EXPECTED_IMAGE_SIZES}
        self.image_first_times: dict[str, float | None] = {
            topic: None for topic in EXPECTED_IMAGE_SIZES
        }
        self.image_last_times: dict[str, float | None] = {
            topic: None for topic in EXPECTED_IMAGE_SIZES
        }
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(Odometry, "/fastlio2/lio_odom", self._on_odom, 10)
        self.create_subscription(CustomMsg, "/livox/lidar", self._on_livox_lidar, 10)
        self.create_subscription(
            PointCloud2, "/livox/lidar_points", self._on_lidar_points, qos_profile_sensor_data
        )
        self.create_subscription(LaserScan, "/scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Imu, "/livox/imu", self._on_imu, qos_profile_sensor_data)
        self._image_subscriptions = [
            self.create_subscription(
                CompressedImage,
                topic,
                lambda message, current_topic=topic: self._on_image(current_topic, message),
                qos_profile_sensor_data,
            )
            for topic in EXPECTED_IMAGE_SIZES
        ]

    def _on_joint_state(self, message: JointState) -> None:
        self.joint_state = message

    def _on_odom(self, message: Odometry) -> None:
        self.odom = message

    def _on_livox_lidar(self, message: CustomMsg) -> None:
        self.livox_lidar = message
        self.livox_point_counts.append(int(message.point_num))
        self.livox_times.append(time.monotonic())

    def _on_lidar_points(self, message: PointCloud2) -> None:
        self.lidar_points = message
        self.pointcloud_counts.append(int(message.width) * int(message.height))
        self.pointcloud_times.append(time.monotonic())

    def _on_scan(self, message: LaserScan) -> None:
        self.scan = message

    def _on_imu(self, message: Imu) -> None:
        self.imu = message

    def _on_image(self, topic: str, message: CompressedImage) -> None:
        try:
            with PILImage.open(BytesIO(bytes(message.data))) as image:
                rgb_image = image.convert("RGB")
                extrema = rgb_image.getextrema()
                self.images[topic] = {
                    "size": image.size,
                    "non_black": any(maximum > 0 for _, maximum in extrema),
                }
            now = time.monotonic()
            self.image_counts[topic] += 1
            if self.image_first_times[topic] is None:
                self.image_first_times[topic] = now
            self.image_last_times[topic] = now
        except Exception as error:
            self.get_logger().error(f"Failed to decode {topic}: {error}")

    @staticmethod
    def _rate_hz(times: list[float]) -> float:
        if len(times) < 2 or times[-1] <= times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def collect(
        self,
        timeout_s: float,
        sample_seconds: float,
        min_lidar_points: int,
        min_lidar_hz: float,
    ) -> dict:
        started = time.monotonic()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (
                self.joint_state is not None
                and self.odom is not None
                and self.livox_lidar is not None
                and self.lidar_points is not None
                and self.scan is not None
                and self.imu is not None
                and len(self.images) == 4
                and time.monotonic() - started >= sample_seconds
            ):
                break

        graph = dict(self.get_topic_names_and_types())
        type_checks = {}
        for topic, expected_type in EXPECTED_TYPES.items():
            actual_types = graph.get(topic, [])
            type_checks[topic] = {
                "expected": expected_type,
                "actual": actual_types,
                "ok": actual_types == [expected_type],
            }

        missing_joints = sorted(
            REQUIRED_JOINTS - set(self.joint_state.name if self.joint_state is not None else [])
        )
        image_checks = {}
        for topic, expected_size in EXPECTED_IMAGE_SIZES.items():
            first_time = self.image_first_times[topic]
            last_time = self.image_last_times[topic]
            span = last_time - first_time if first_time is not None and last_time is not None else 0.0
            rate_hz = (
                (self.image_counts[topic] - 1) / span
                if self.image_counts[topic] > 1 and span > 0.0
                else 0.0
            )
            observation = self.images.get(topic)
            size = observation["size"] if observation is not None else None
            non_black = bool(observation and observation["non_black"])
            image_checks[topic] = {
                "expected": list(expected_size),
                "actual": list(size) if size is not None else None,
                "count": self.image_counts[topic],
                "rate_hz": rate_hz,
                "non_black": non_black,
                "ok": size == expected_size and non_black and rate_hz >= 13.0,
            }
        odom_ok = self.odom is not None and bool(self.odom.header.frame_id)
        joint_ok = self.joint_state is not None and not missing_joints
        livox_offsets = (
            [int(point.offset_time) for point in self.livox_lidar.points]
            if self.livox_lidar is not None
            else []
        )
        livox_points_finite = bool(
            self.livox_lidar
            and all(
                math.isfinite(value)
                for point in self.livox_lidar.points[:100]
                for value in (point.x, point.y, point.z)
            )
        )
        livox_rate_hz = self._rate_hz(self.livox_times)
        livox_median_points = (
            float(statistics.median(self.livox_point_counts)) if self.livox_point_counts else 0.0
        )
        livox_ok = bool(
            self.livox_lidar
            and self.livox_lidar.point_num == len(self.livox_lidar.points)
            and self.livox_lidar.timebase > 0
            and bool(self.livox_lidar.header.frame_id)
            and livox_points_finite
            and livox_offsets == sorted(livox_offsets)
            and len(self.livox_point_counts) >= 3
            and livox_rate_hz >= min_lidar_hz
            and livox_median_points >= min_lidar_points
        )
        pointcloud_count = (
            int(self.lidar_points.width) * int(self.lidar_points.height)
            if self.lidar_points is not None
            else 0
        )
        pointcloud_rate_hz = self._rate_hz(self.pointcloud_times)
        pointcloud_median_points = (
            float(statistics.median(self.pointcloud_counts)) if self.pointcloud_counts else 0.0
        )
        pointcloud_ok = bool(
            self.lidar_points
            and len(self.lidar_points.data) > 0
            and bool(self.lidar_points.header.frame_id)
            and len(self.pointcloud_counts) >= 3
            and pointcloud_rate_hz >= min_lidar_hz
            and pointcloud_median_points >= min_lidar_points
        )
        finite_scan_ranges = (
            sum(math.isfinite(value) for value in self.scan.ranges)
            if self.scan is not None
            else 0
        )
        scan_ok = bool(self.scan and len(self.scan.ranges) > 0 and finite_scan_ranges > 0)
        imu_ok = bool(self.imu and bool(self.imu.header.frame_id))
        result = {
            "type_checks": type_checks,
            "joint_states": {
                "received": self.joint_state is not None,
                "joint_count": len(self.joint_state.name) if self.joint_state is not None else 0,
                "missing_required_joints": missing_joints,
                "ok": joint_ok,
            },
            "odom": {
                "received": self.odom is not None,
                "frame_id": self.odom.header.frame_id if self.odom is not None else "",
                "child_frame_id": self.odom.child_frame_id if self.odom is not None else "",
                "ok": odom_ok,
                "source_semantics": "simulated /odom compatibility relay; not FAST-LIO estimation",
            },
            "livox_lidar": {
                "received": self.livox_lidar is not None,
                "sample_count": len(self.livox_point_counts),
                "rate_hz": livox_rate_hz,
                "min_point_num": min(self.livox_point_counts) if self.livox_point_counts else 0,
                "median_point_num": livox_median_points,
                "max_point_num": max(self.livox_point_counts) if self.livox_point_counts else 0,
                "required_median_point_num": min_lidar_points,
                "required_rate_hz": min_lidar_hz,
                "timebase": int(self.livox_lidar.timebase) if self.livox_lidar else 0,
                "frame_id": self.livox_lidar.header.frame_id if self.livox_lidar else "",
                "offsets_monotonic": livox_offsets == sorted(livox_offsets),
                "sample_points_finite": livox_points_finite,
                "ok": livox_ok,
            },
            "lidar_points": {
                "received": self.lidar_points is not None,
                "sample_count": len(self.pointcloud_counts),
                "rate_hz": pointcloud_rate_hz,
                "min_point_count": min(self.pointcloud_counts) if self.pointcloud_counts else 0,
                "median_point_count": pointcloud_median_points,
                "max_point_count": max(self.pointcloud_counts) if self.pointcloud_counts else 0,
                "required_median_point_count": min_lidar_points,
                "required_rate_hz": min_lidar_hz,
                "last_point_count": pointcloud_count,
                "frame_id": self.lidar_points.header.frame_id if self.lidar_points else "",
                "ok": pointcloud_ok,
            },
            "scan": {
                "received": self.scan is not None,
                "bin_count": len(self.scan.ranges) if self.scan else 0,
                "finite_range_count": finite_scan_ranges,
                "ok": scan_ok,
            },
            "imu": {
                "received": self.imu is not None,
                "frame_id": self.imu.header.frame_id if self.imu else "",
                "ok": imu_ok,
            },
            "compressed_images": image_checks,
        }
        result["pass"] = (
            all(check["ok"] for check in type_checks.values())
            and joint_ok
            and odom_ok
            and livox_ok
            and pointcloud_ok
            and scan_ok
            and imu_ok
            and all(check["ok"] for check in image_checks.values())
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sample-seconds", type=float, default=10.0)
    parser.add_argument("--min-lidar-points", type=int, default=10000)
    parser.add_argument("--min-lidar-hz", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rclpy.init()
    node = VlaRuntimeVerifier()
    try:
        result = node.collect(
            args.timeout,
            max(1.0, args.sample_seconds),
            max(1, args.min_lidar_points),
            max(0.0, args.min_lidar_hz),
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
