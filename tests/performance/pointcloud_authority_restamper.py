#!/usr/bin/env python3
"""Restamp an internal MID360 PointCloud2 stream with the authority /clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import PointCloud2


SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    reliability=ReliabilityPolicy.BEST_EFFORT,
)


class AuthorityPointCloudRestamper(Node):
    def __init__(self, source_topic: str, output_topic: str) -> None:
        super().__init__("mid360_authority_clock_restamper")
        self._latest_clock: Clock | None = None
        self._publisher = self.create_publisher(PointCloud2, output_topic, SENSOR_QOS)
        self.create_subscription(Clock, "/clock", self._clock_callback, 10)
        self.create_subscription(PointCloud2, source_topic, self._pointcloud_callback, SENSOR_QOS)
        self.start_wall_ns = time.monotonic_ns()
        self.first_input_wall_ns: int | None = None
        self.last_input_wall_ns: int | None = None
        self.first_output_wall_ns: int | None = None
        self.last_output_wall_ns: int | None = None
        self.clock_messages = 0
        self.input_messages = 0
        self.output_messages = 0
        self.dropped_without_clock = 0
        self.last_input_stamp_ns: int | None = None
        self.last_output_stamp_ns: int | None = None
        self.last_point_count = 0

    @staticmethod
    def _stamp_ns(stamp: object) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _clock_callback(self, message: Clock) -> None:
        self._latest_clock = message
        self.clock_messages += 1

    def _pointcloud_callback(self, message: PointCloud2) -> None:
        callback_wall_ns = time.monotonic_ns()
        if self.first_input_wall_ns is None:
            self.first_input_wall_ns = callback_wall_ns
        self.last_input_wall_ns = callback_wall_ns
        self.input_messages += 1
        self.last_input_stamp_ns = self._stamp_ns(message.header.stamp)
        self.last_point_count = int(message.width) * max(1, int(message.height))
        if self._latest_clock is None:
            self.dropped_without_clock += 1
            return
        message.header.stamp = self._latest_clock.clock
        self.last_output_stamp_ns = self._stamp_ns(message.header.stamp)
        self._publisher.publish(message)
        if self.first_output_wall_ns is None:
            self.first_output_wall_ns = callback_wall_ns
        self.last_output_wall_ns = callback_wall_ns
        self.output_messages += 1

    @staticmethod
    def _active_rate(count: int, first_wall_ns: int | None, last_wall_ns: int | None) -> float | None:
        if count < 2 or first_wall_ns is None or last_wall_ns is None:
            return None
        duration_s = max(0.0, (last_wall_ns - first_wall_ns) / 1e9)
        return count / duration_s if duration_s > 0.0 else None

    @staticmethod
    def _active_duration(first_wall_ns: int | None, last_wall_ns: int | None) -> float | None:
        if first_wall_ns is None or last_wall_ns is None:
            return None
        return max(0.0, (last_wall_ns - first_wall_ns) / 1e9)

    def _since_start(self, wall_ns: int | None) -> float | None:
        if wall_ns is None:
            return None
        return max(0.0, (wall_ns - self.start_wall_ns) / 1e9)

    def metrics(self) -> dict[str, object]:
        elapsed_s = max(0.0, (time.monotonic_ns() - self.start_wall_ns) / 1e9)
        return {
            "schema_version": 2,
            "status": "complete",
            "elapsed_wall_s": elapsed_s,
            "startup_to_first_input_wall_s": (
                (self.first_input_wall_ns - self.start_wall_ns) / 1e9
                if self.first_input_wall_ns is not None else None
            ),
            "startup_to_first_output_wall_s": (
                (self.first_output_wall_ns - self.start_wall_ns) / 1e9
                if self.first_output_wall_ns is not None else None
            ),
            "input_first_wall_since_start_s": self._since_start(self.first_input_wall_ns),
            "input_last_wall_since_start_s": self._since_start(self.last_input_wall_ns),
            "output_first_wall_since_start_s": self._since_start(self.first_output_wall_ns),
            "output_last_wall_since_start_s": self._since_start(self.last_output_wall_ns),
            "clock_messages": self.clock_messages,
            "input_messages": self.input_messages,
            "output_messages": self.output_messages,
            "dropped_without_clock": self.dropped_without_clock,
            "input_active_duration_wall_s": self._active_duration(
                self.first_input_wall_ns, self.last_input_wall_ns
            ),
            "output_active_duration_wall_s": self._active_duration(
                self.first_output_wall_ns, self.last_output_wall_ns
            ),
            "input_active_wall_hz": self._active_rate(
                self.input_messages, self.first_input_wall_ns, self.last_input_wall_ns
            ),
            "output_active_wall_hz": self._active_rate(
                self.output_messages, self.first_output_wall_ns, self.last_output_wall_ns
            ),
            "output_wall_hz": self.output_messages / elapsed_s if elapsed_s else None,
            "last_input_stamp_ns": self.last_input_stamp_ns,
            "last_output_stamp_ns": self.last_output_stamp_ns,
            "last_point_count": self.last_point_count,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-topic", required=True)
    parser.add_argument("--output-topic", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_topic == args.output_topic:
        parser.error("source and output topics must differ")

    rclpy.init()
    node = AuthorityPointCloudRestamper(args.source_topic, args.output_topic)
    stopping = False

    def _stop(_signal_number: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    try:
        while rclpy.ok() and not stopping:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(node.metrics(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
