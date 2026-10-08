#!/usr/bin/python3
"""Runtime smoke test for the OpenFleX Isaac Sim upper-body bringup."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from livox_ros_driver2.msg import CustomMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, JointState, LaserScan, PointCloud2
from std_msgs.msg import Float64MultiArray


REQUIRED_JOINTS = {
    "lift_joint",
    "openarmx_left_joint1",
    "openarmx_left_joint7",
    "openarmx_left_finger_joint1",
    "openarmx_left_finger_joint2",
    "openarmx_right_joint1",
    "openarmx_right_joint7",
    "openarmx_right_finger_joint1",
    "openarmx_right_finger_joint2",
    "openarmx_head_yaw_joint",
    "openarmx_head_pitch_joint",
}


class RuntimeVerifier(Node):
    def __init__(self, require_sensors: bool) -> None:
        super().__init__("openflex_isaac_upper_body_runtime_verifier")
        self.require_sensors = require_sensors
        self.last_clock: Clock | None = None
        self.last_joint_state: JointState | None = None
        self.last_livox_lidar: CustomMsg | None = None
        self.last_livox_points: PointCloud2 | None = None
        self.last_scan: LaserScan | None = None
        self.last_camera_info: dict[str, CameraInfo] = {}

        self.create_subscription(Clock, "/clock", self._clock_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)
        self.create_subscription(CustomMsg, "/livox/lidar", self._livox_lidar_cb, 10)
        self.create_subscription(
            PointCloud2,
            "/livox/lidar_points",
            self._livox_points_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(LaserScan, "/scan", self._scan_cb, qos_profile_sensor_data)
        for topic in (
            "/cam_left/color/camera_info",
            "/cam_right/color/camera_info",
            "/cam_head/color/camera_info",
            "/cam_base/color/camera_info",
        ):
            self.create_subscription(
                CameraInfo,
                topic,
                lambda msg, camera_topic=topic: self._camera_info_cb(camera_topic, msg),
                qos_profile_sensor_data,
            )

        self.lift_command_pub = self.create_publisher(
            Float64MultiArray, "/lift_position_controller/commands", 10
        )

    def _clock_cb(self, msg: Clock) -> None:
        self.last_clock = msg

    def _joint_state_cb(self, msg: JointState) -> None:
        self.last_joint_state = msg

    def _livox_lidar_cb(self, msg: CustomMsg) -> None:
        self.last_livox_lidar = msg

    def _livox_points_cb(self, msg: PointCloud2) -> None:
        self.last_livox_points = msg

    def _scan_cb(self, msg: LaserScan) -> None:
        self.last_scan = msg

    def _camera_info_cb(self, topic: str, msg: CameraInfo) -> None:
        self.last_camera_info[topic] = msg

    def wait_for_messages(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._messages_ready():
                self._assert_required_joints()
                return
        missing = self._missing_topics()
        raise RuntimeError(f"Timed out waiting for messages from: {', '.join(missing)}")

    def _messages_ready(self) -> bool:
        ready = self.last_clock is not None and self.last_joint_state is not None
        if self.require_sensors:
            ready = ready and all(
                (
                    self.last_livox_lidar is not None,
                    self.last_livox_points is not None,
                    self.last_scan is not None,
                    "/cam_left/color/camera_info" in self.last_camera_info,
                    "/cam_right/color/camera_info" in self.last_camera_info,
                    "/cam_head/color/camera_info" in self.last_camera_info,
                    "/cam_base/color/camera_info" in self.last_camera_info,
                )
            )
        return ready

    def _missing_topics(self) -> list[str]:
        missing = []
        if self.last_clock is None:
            missing.append("/clock")
        if self.last_joint_state is None:
            missing.append("/joint_states")
        if self.require_sensors:
            if self.last_livox_lidar is None:
                missing.append("/livox/lidar")
            if self.last_livox_points is None:
                missing.append("/livox/lidar_points")
            if self.last_scan is None:
                missing.append("/scan")
            for topic in (
                "/cam_left/color/camera_info",
                "/cam_right/color/camera_info",
                "/cam_head/color/camera_info",
                "/cam_base/color/camera_info",
            ):
                if topic not in self.last_camera_info:
                    missing.append(topic)
        return missing

    def _assert_required_joints(self) -> None:
        if self.last_joint_state is None:
            raise RuntimeError("No /joint_states received yet")
        missing = sorted(REQUIRED_JOINTS - set(self.last_joint_state.name))
        if missing:
            raise RuntimeError(f"Missing expected joints in /joint_states: {', '.join(missing)}")

    def get_joint_position(self, joint_name: str) -> float:
        if self.last_joint_state is None:
            raise RuntimeError("No /joint_states received yet")
        try:
            idx = self.last_joint_state.name.index(joint_name)
        except ValueError as exc:
            raise RuntimeError(f"Joint '{joint_name}' not found in /joint_states") from exc
        return float(self.last_joint_state.position[idx])

    def stream_lift_command(self, target: float, duration_sec: float, rate_hz: float = 10.0) -> None:
        period = 1.0 / rate_hz
        end_time = time.monotonic() + duration_sec
        message = Float64MultiArray(data=[target])
        while time.monotonic() < end_time:
            self.lift_command_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=period)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lift-target", type=float, default=0.12)
    parser.add_argument("--publish-seconds", type=float, default=2.0)
    parser.add_argument("--joint-timeout", type=float, default=10.0)
    parser.add_argument("--required-lift-delta", type=float, default=0.01)
    parser.add_argument("--require-sensors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = RuntimeVerifier(require_sensors=args.require_sensors)
    try:
        node.wait_for_messages(timeout_sec=args.joint_timeout)
        before = node.get_joint_position("lift_joint")
        node.stream_lift_command(args.lift_target, args.publish_seconds)

        settle_deadline = time.monotonic() + args.joint_timeout
        after = before
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            after = node.get_joint_position("lift_joint")
            if abs(after - before) >= args.required_lift_delta:
                break

        delta = after - before
        print(
            {
                "clock": "ok",
                "joint_states": "ok",
                "required_joints": "ok",
                "lift_before": before,
                "lift_after": after,
                "lift_delta": delta,
                "sensor_checks": "enabled" if args.require_sensors else "skipped",
            }
        )
        if abs(delta) < args.required_lift_delta:
            raise RuntimeError(
                f"lift_joint moved only {delta:.6f}, below threshold {args.required_lift_delta:.6f}"
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"verification_failed: {exc}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
