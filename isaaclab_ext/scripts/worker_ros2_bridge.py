#!/usr/bin/env python3
"""Bridge the existing Pico ROS 2 teleop topics to the Isaac Lab Worker API.

This process owns ROS subscriptions and publishes the current Isaac Lab joint
state back to ``/joint_states`` for the existing VR IK node. It never starts a
ROS controller manager or creates a second SimulationContext.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def _request_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="GET" if data is None else "POST",
    )
    with urlopen(request, timeout=0.4) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Worker API response must be a JSON object")
    return result


def _finite_values(values: Any, expected: int) -> list[float] | None:
    try:
        if len(values) != expected:
            return None
        converted = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return converted if all(math.isfinite(value) for value in converted) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--task-id", choices=("navigation_to_goal", "dual_arm_box_transport"), required=True)
    parser.add_argument("--control-hz", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not args.api_url.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("--api-url must point to the Worker loopback API")
    if not math.isfinite(args.control_hz) or args.control_hz <= 0.0:
        parser.error("--control-hz must be a positive finite rate")

    try:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Float64MultiArray
    except ImportError as error:
        print(f"ROS 2 Python packages unavailable: {error}", file=sys.stderr, flush=True)
        return 3

    class WorkerRos2Bridge(Node):
        def __init__(self) -> None:
            super().__init__("mrs_robot_lab_worker_ros2_bridge")
            self._started_at = time.monotonic()
            self._api_url = args.api_url.rstrip("/")
            self._task_id = args.task_id
            self._ready = False
            self._startup_failed = False
            self._missing_stack_checks = 0
            self._source_seq = 0
            self._estop = False
            self._latest_twist: tuple[float, float, float] | None = None
            self._latest_twist_time = 0.0
            self._latest_left: list[float] | None = None
            self._latest_left_time = 0.0
            self._latest_right: list[float] | None = None
            self._latest_right_time = 0.0
            self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
            self.create_subscription(Bool, "/vr_estop_active", self._on_estop, 10)
            if self._task_id == "navigation_to_goal":
                self.create_subscription(Twist, "/cmd_vel", self._on_twist, 10)
                expected_nodes = ("pico_pose_bridge", "vr_teleop_chassis")
            else:
                self.create_subscription(
                    Float64MultiArray,
                    "/left_forward_position_controller/commands",
                    self._on_left_arm,
                    10,
                )
                self.create_subscription(
                    Float64MultiArray,
                    "/right_forward_position_controller/commands",
                    self._on_right_arm,
                    10,
                )
                expected_nodes = ("pico_pose_bridge", "openarmx_teleop_vr_node")
            self._expected_nodes = set(expected_nodes)
            rate = min(float(args.control_hz), 60.0)
            self.create_timer(1.0 / rate, self._tick)
            self.create_timer(1.0, self._check_stack)
            self.get_logger().info(
                f"Worker ROS 2 bridge started for {self._task_id}; "
                f"waiting for {', '.join(expected_nodes)}"
            )

        def _on_estop(self, message) -> None:
            self._estop = bool(message.data)

        def _on_twist(self, message) -> None:
            values = (message.linear.x, message.linear.y, message.angular.z)
            if all(math.isfinite(float(value)) for value in values):
                self._latest_twist = tuple(float(value) for value in values)
                self._latest_twist_time = time.monotonic()

        def _on_left_arm(self, message) -> None:
            self._latest_left = _finite_values(message.data, 8)
            self._latest_left_time = time.monotonic()

        def _on_right_arm(self, message) -> None:
            self._latest_right = _finite_values(message.data, 8)
            self._latest_right_time = time.monotonic()

        def _check_stack(self) -> None:
            observed = {name.rsplit("/", 1)[-1] for name in self.get_node_names()}
            missing = sorted(self._expected_nodes - observed)
            if not missing:
                self._missing_stack_checks = 0
                if not self._ready:
                    self._ready = True
                    self._send_ready(True, "Pico ROS 2 control nodes are ready")
                    self.get_logger().info("Pico VR ROS 2 control stack is ready")
                return
            self._missing_stack_checks += 1
            message = "VR ROS 2 节点未就绪：" + ", ".join(missing)
            if not self._ready and time.monotonic() - self._started_at > 45.0:
                self._startup_failed = True
                self._send_ready(False, message)
                self.get_logger().error(message)
                rclpy.shutdown()
            elif self._ready and self._missing_stack_checks >= 3:
                self._send_ready(False, message)
                self.get_logger().error(message)
                rclpy.shutdown()

        def _tick(self) -> None:
            if self._ready:
                self._publish_worker_joint_state()
                self._publish_command()

        def _publish_worker_joint_state(self) -> None:
            try:
                state = _request_json(self._api_url + "/api/v1/teleop/state")
            except (URLError, TimeoutError, ValueError) as error:
                self.get_logger().warning(f"Worker state API unavailable: {error}")
                return
            joint_state = state.get("joint_state")
            if not isinstance(joint_state, dict):
                return
            names = joint_state.get("names")
            positions = _finite_values(joint_state.get("position", ()), len(names or ()))
            velocities = _finite_values(joint_state.get("velocity", ()), len(names or ()))
            if not names or positions is None or velocities is None:
                return
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = [str(name) for name in names]
            message.position = positions
            message.velocity = velocities
            self._joint_state_pub.publish(message)

        def _publish_command(self) -> None:
            self._source_seq += 1
            now = time.monotonic()
            if self._estop:
                payload = {"kind": "estop", "active": True}
            elif self._task_id == "navigation_to_goal":
                twist = self._latest_twist
                if twist is None or now - self._latest_twist_time > 0.5:
                    twist = (0.0, 0.0, 0.0)
                payload = {"kind": "base_twist", "values": twist}
            else:
                left = self._latest_left if now - self._latest_left_time <= 0.5 else None
                right = self._latest_right if now - self._latest_right_time <= 0.5 else None
                payload = {"kind": "dual_arm", "left": left, "right": right}
            payload["source_seq"] = self._source_seq
            try:
                _request_json(self._api_url + "/api/v1/teleop/command", payload)
            except (URLError, TimeoutError, ValueError) as error:
                self.get_logger().warning(f"Worker command API unavailable: {error}")

        def _send_ready(self, ready: bool, message: str) -> None:
            try:
                _request_json(
                    self._api_url + "/api/v1/teleop/ready",
                    {"ready": ready, "message": message},
                )
            except (URLError, TimeoutError, ValueError) as error:
                self.get_logger().error(f"Could not report VR sidecar health: {error}")

    rclpy.init(args=None)
    node = WorkerRos2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._ready:
            node._send_ready(False, "VR ROS 2 bridge stopped")
        node.destroy_node()
        rclpy.try_shutdown()
    return 8 if node._startup_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
