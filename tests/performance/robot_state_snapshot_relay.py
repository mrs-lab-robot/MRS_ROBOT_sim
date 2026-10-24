#!/usr/bin/env python3
"""将物理权威进程的机器人状态转发给本机相机副本进程。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import socket
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState


@dataclass
class RelayMetrics:
    joint_messages: int = 0
    odom_messages: int = 0
    snapshots_sent: int = 0
    datagrams_sent: int = 0
    latest_joint_stamp_ns: int | None = None
    latest_odom_stamp_ns: int | None = None
    start_wall_ns: int = 0


class SnapshotRelay(Node):
    """System-Python ROS subscriber with a deliberately tiny UDP IPC payload."""

    def __init__(self, host: str, ports: list[int], state_topic: str, odom_topic: str) -> None:
        super().__init__("robot_state_snapshot_relay")
        self._targets = [(host, port) for port in dict.fromkeys(ports)]
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._metrics = RelayMetrics(start_wall_ns=time.monotonic_ns())
        self._latest_odom: dict[str, object] | None = None
        self.create_subscription(JointState, state_topic, self._joint_callback, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_callback, 10)

    @staticmethod
    def _stamp_ns(message: object) -> int:
        stamp = message.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _odom_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        self._latest_odom = {
            "stamp_ns": self._stamp_ns(message),
            "position": [float(position.x), float(position.y), float(position.z)],
            "orientation_xyzw": [
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ],
        }
        self._metrics.odom_messages += 1
        self._metrics.latest_odom_stamp_ns = int(self._latest_odom["stamp_ns"])

    def _joint_callback(self, message: JointState) -> None:
        count = min(len(message.name), len(message.position))
        if count == 0:
            return
        stamp_ns = self._stamp_ns(message)
        payload = {
            "schema_version": 1,
            "relay_wall_ns": time.monotonic_ns(),
            "joint_stamp_ns": stamp_ns,
            "names": list(message.name[:count]),
            "positions": [float(value) for value in message.position[:count]],
            "odom": self._latest_odom,
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for target in self._targets:
            self._socket.sendto(encoded, target)
            self._metrics.datagrams_sent += 1
        self._metrics.joint_messages += 1
        self._metrics.snapshots_sent += 1
        self._metrics.latest_joint_stamp_ns = stamp_ns

    def metrics(self) -> dict[str, object]:
        elapsed_s = max(0.0, (time.monotonic_ns() - self._metrics.start_wall_ns) / 1e9)
        return {
            "schema_version": 1,
            "status": "complete",
            "elapsed_wall_s": elapsed_s,
            "joint_messages": self._metrics.joint_messages,
            "odom_messages": self._metrics.odom_messages,
            "snapshots_sent": self._metrics.snapshots_sent,
            "datagrams_sent": self._metrics.datagrams_sent,
            "target_count": len(self._targets),
            "targets": [f"{host}:{port}" for host, port in self._targets],
            "snapshot_wall_hz": self._metrics.snapshots_sent / elapsed_s if elapsed_s else None,
            "latest_joint_stamp_ns": self._metrics.latest_joint_stamp_ns,
            "latest_odom_stamp_ns": self._metrics.latest_odom_stamp_ns,
        }

    def close(self) -> None:
        self._socket.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24101)
    parser.add_argument(
        "--additional-port",
        type=int,
        action="append",
        default=[],
        help="additional local UDP snapshot target; may be repeated",
    )
    parser.add_argument("--state-topic", default="/openflex/joint_states")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ports = [args.port, *args.additional_port]
    if any(not 1 <= port <= 65535 for port in ports):
        parser.error("all ports must be between 1 and 65535")

    rclpy.init()
    relay = SnapshotRelay(args.host, ports, args.state_topic, args.odom_topic)
    stopping = False

    def _stop(_signal_number, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while rclpy.ok() and not stopping:
            rclpy.spin_once(relay, timeout_sec=0.1)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(relay.metrics(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        relay.close()
        relay.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
