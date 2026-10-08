#!/usr/bin/env python3
"""Publish the real-robot compressed camera contract from Isaac RGB images."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from PIL import Image as PILImage


CAMERA_OUTPUT_SIZES = {
    "base": (320, 240),
    "head": (640, 480),
    "left": (320, 240),
    "right": (320, 240),
}


@dataclass
class CameraRelay:
    name: str
    publisher: object
    output_size: tuple[int, int]
    next_publish_time: float = 0.0
    compression_in_flight: bool = False


class CameraContractPublisher(Node):
    def __init__(self) -> None:
        super().__init__("openflex_camera_contract_publisher")
        self.declare_parameter("max_rate_hz", 15.0)
        self.declare_parameter("jpeg_quality", 80)
        self._workers = ThreadPoolExecutor(
            max_workers=len(CAMERA_OUTPUT_SIZES),
            thread_name_prefix="openflex_camera",
        )
        self._relays: dict[str, CameraRelay] = {}
        self._image_subscriptions = []

        for camera, output_size in CAMERA_OUTPUT_SIZES.items():
            publisher = self.create_publisher(
                CompressedImage,
                f"/cam_{camera}/color/image/compressed",
                qos_profile_sensor_data,
            )
            relay = CameraRelay(camera, publisher, output_size)
            self._relays[camera] = relay
            self._image_subscriptions.append(
                self.create_subscription(
                    Image,
                    f"/cam_{camera}/color/image",
                    lambda message, current=relay: self._on_image(current, message),
                    qos_profile_sensor_data,
                )
            )

        self.get_logger().info("OpenFleX compressed camera contract publisher started")

    def _on_image(self, relay: CameraRelay, message: Image) -> None:
        if relay.publisher.get_subscription_count() == 0 or relay.compression_in_flight:
            return
        max_rate_hz = float(self.get_parameter("max_rate_hz").value)
        now = time.monotonic()
        if max_rate_hz > 0.0:
            period = 1.0 / max_rate_hz
            if relay.next_publish_time and now < relay.next_publish_time:
                return
            if relay.next_publish_time:
                relay.next_publish_time += period
                if relay.next_publish_time < now - period:
                    relay.next_publish_time = now + period
            else:
                relay.next_publish_time = now + period
        relay.compression_in_flight = True
        future = self._workers.submit(self._compress, message, relay.output_size)
        future.add_done_callback(
            lambda completed, current=relay: self._publish_result(current, completed)
        )

    def _compress(self, message: Image, output_size: tuple[int, int]) -> CompressedImage:
        formats = {
            "rgb8": ("RGB", "RGB", 3),
            "bgr8": ("RGB", "BGR", 3),
            "rgba8": ("RGBA", "RGBA", 4),
            "bgra8": ("RGBA", "BGRA", 4),
            "mono8": ("L", "L", 1),
            "8uc1": ("L", "L", 1),
        }
        encoding = message.encoding.lower()
        if encoding not in formats:
            raise ValueError(f"unsupported color encoding: {message.encoding}")
        mode, raw_mode, channels = formats[encoding]
        row_bytes = int(message.width) * channels
        if int(message.step) < row_bytes:
            raise ValueError(f"invalid image step {message.step} for {message.width}x{channels}")
        data = bytes(message.data)
        if int(message.step) != row_bytes:
            data = b"".join(
                data[offset : offset + row_bytes]
                for offset in range(0, int(message.step) * int(message.height), int(message.step))
            )
        image = PILImage.frombytes(mode, (int(message.width), int(message.height)), data, "raw", raw_mode)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        if image.size != output_size:
            resampling = getattr(PILImage, "Resampling", PILImage)
            image = image.resize(output_size, resampling.LANCZOS)
        output = BytesIO()
        quality = max(1, min(100, int(self.get_parameter("jpeg_quality").value)))
        image.save(output, format="JPEG", quality=quality)

        compressed = CompressedImage()
        compressed.header = message.header
        compressed.format = "jpeg"
        compressed.data = output.getvalue()
        return compressed

    def _publish_result(self, relay: CameraRelay, future) -> None:
        relay.compression_in_flight = False
        try:
            compressed = future.result()
        except Exception as error:
            self.get_logger().warn(f"JPEG compression failed for cam_{relay.name}: {error}")
            return
        if relay.publisher.get_subscription_count() > 0:
            relay.publisher.publish(compressed)

    def destroy_node(self) -> bool:
        self._workers.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main() -> int:
    rclpy.init()
    node = CameraContractPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
