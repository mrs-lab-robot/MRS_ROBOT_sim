#!/usr/bin/env python3
"""Reject missing, all-black RGB, and invalid depth camera streams."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


CAMERAS = ("base", "head", "left", "right")


def topics() -> list[str]:
    return [
        f"/cam_{camera}/{kind}/image"
        for camera in CAMERAS
        for kind in ("color", "depth")
    ]


def color_stats(message: Image) -> dict[str, object]:
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        return {"ok": False, "reason": f"unsupported color encoding: {message.encoding}"}
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)[..., :3]
    nonzero_ratio = float(np.count_nonzero(image) / image.size)
    standard_deviation = float(image.std())
    return {
        "ok": bool(image.max() > 0 and standard_deviation > 0.5 and nonzero_ratio > 0.001),
        "min": int(image.min()),
        "max": int(image.max()),
        "mean": float(image.mean()),
        "std": standard_deviation,
        "nonzero_ratio": nonzero_ratio,
    }


def depth_stats(message: Image) -> dict[str, object]:
    if message.encoding.upper() != "32FC1":
        return {"ok": False, "reason": f"unsupported depth encoding: {message.encoding}"}
    dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
    rows = np.frombuffer(bytes(message.data), dtype=dtype).reshape(message.height, message.step // 4)
    image = rows[:, : message.width]
    valid = np.isfinite(image) & (image > 0.0)
    values = image[valid]
    result: dict[str, object] = {
        "ok": bool(values.size and valid.mean() > 0.001),
        "finite_ratio": float(np.isfinite(image).mean()),
        "positive_ratio": float(valid.mean()),
    }
    if values.size:
        result.update(
            min=float(values.min()),
            max=float(values.max()),
            mean=float(values.mean()),
            std=float(values.std()),
        )
    return result


class CameraImageVerifier(Node):
    def __init__(self, expected_topics: list[str]) -> None:
        super().__init__("openflex_camera_image_verifier")
        self.messages: dict[str, Image] = {}
        self._image_subscriptions = [
            self.create_subscription(
                Image,
                topic,
                lambda message, current_topic=topic: self.messages.setdefault(current_topic, message),
                qos_profile_sensor_data,
            )
            for topic in expected_topics
        ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    expected_topics = topics()

    rclpy.init()
    node = CameraImageVerifier(expected_topics)
    try:
        deadline = time.monotonic() + args.timeout
        while len(node.messages) < len(expected_topics) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)

        results: dict[str, dict[str, object]] = {}
        for topic in expected_topics:
            message = node.messages.get(topic)
            if message is None:
                results[topic] = {"ok": False, "reason": "timeout"}
                continue
            stats = depth_stats(message) if "/depth/" in topic else color_stats(message)
            results[topic] = {
                "width": int(message.width),
                "height": int(message.height),
                "encoding": message.encoding,
                **stats,
            }

        summary = {
            "expected": len(expected_topics),
            "received": len(node.messages),
            "passed": sum(bool(result.get("ok")) for result in results.values()),
            "all_pass": all(bool(result.get("ok")) for result in results.values()),
            "topics": results,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["all_pass"] else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
