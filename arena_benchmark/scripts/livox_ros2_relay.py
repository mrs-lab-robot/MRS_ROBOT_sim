#!/usr/bin/env python3
"""Convert Arena's ROS 2 PointCloud2 stream to Livox MID360 CustomMsg."""

from __future__ import annotations

import argparse
import copy
import math
import signal
import subprocess
import struct
import sys
import threading
import time
from typing import Any


_FIELD_FORMATS = {
    1: "b",  # INT8
    2: "B",  # UINT8
    3: "h",  # INT16
    4: "H",  # UINT16
    5: "i",  # INT32
    6: "I",  # UINT32
    7: "f",  # FLOAT32
    8: "d",  # FLOAT64
}


def _stamp_ns(header: Any) -> int:
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return max(0, int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0)))


def _read_scalar(data: memoryview, offset: int, field: Any, endian: str) -> int | float:
    fmt = _FIELD_FORMATS.get(int(field.datatype))
    if fmt is None:
        raise ValueError(f"unsupported PointCloud2 field datatype for {field.name}: {field.datatype}")
    try:
        return struct.unpack_from(endian + fmt, data, offset + int(field.offset))[0]
    except struct.error as error:
        raise ValueError(f"invalid PointCloud2 data for field {field.name}: {error}") from error


def _relative_time_ns(
    value: int | float, field_name: str, frame_time_ns: int
) -> tuple[int | None, int]:
    """Return (absolute timestamp, frame-relative offset) in nanoseconds."""
    name = field_name.lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, 0
        if abs(value) >= 1.0e12:
            return int(value), 0
        if name == "offset_time":
            return None, int(round(value))
        if name == "timestamp":
            as_nanoseconds = int(round(value))
            as_seconds = int(round(value * 1.0e9))
            if frame_time_ns and abs(as_nanoseconds - frame_time_ns) <= 1_000_000_000:
                return as_nanoseconds, 0
            if frame_time_ns and abs(as_seconds - frame_time_ns) <= 2_000_000_000:
                return as_seconds, 0
            if abs(value) >= 1.0e8:
                return as_seconds, 0
            if abs(value) >= 1.0e6:
                return None, as_nanoseconds
        return None, int(round(value * 1.0e9))
    integer = int(value)
    if name == "offset_time":
        return None, integer
    if name == "timestamp":
        if abs(integer) >= 1_000_000_000_000:
            return integer, 0
        if frame_time_ns and abs(integer - frame_time_ns) <= 1_000_000_000:
            return integer, 0
        if abs(integer) >= 100_000_000:
            return integer * 1_000_000_000, 0
    return None, integer


def pointcloud_to_custom_msg(
    point_cloud: Any,
    *,
    custom_msg_type: type,
    custom_point_type: type,
    frame_id: str | None = None,
) -> Any:
    """Decode a PointCloud2 buffer into the installed Livox CustomMsg type."""
    fields = {str(field.name).lower(): field for field in point_cloud.fields}
    required = ("x", "y", "z")
    if any(name not in fields for name in required):
        raise ValueError("PointCloud2 must contain x, y, and z fields")

    intensity_field = next(
        (fields[name] for name in ("reflectivity", "intensity") if name in fields), None
    )
    line_field = next(
        (fields[name] for name in ("line", "channel", "channelid", "ring", "laser_id") if name in fields),
        None,
    )
    time_field = next(
        (fields[name] for name in ("offset_time", "timestamp", "time") if name in fields),
        None,
    )
    frame_time_ns = _stamp_ns(getattr(point_cloud, "header", None))

    width = int(point_cloud.width)
    height = int(point_cloud.height)
    point_step = int(point_cloud.point_step)
    row_step = int(getattr(point_cloud, "row_step", width * point_step))
    if width < 0 or height < 0 or point_step <= 0:
        raise ValueError("PointCloud2 dimensions and point_step are invalid")
    data = memoryview(point_cloud.data)
    endian = ">" if bool(getattr(point_cloud, "is_bigendian", False)) else "<"
    decoded: list[tuple[float, float, float, int, int, int | None, int]] = []
    for row in range(height):
        for column in range(width):
            base = row * row_step + column * point_step
            x, y, z = (
                float(_read_scalar(data, base, fields[name], endian)) for name in required
            )
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            reflectivity = (
                int(round(float(_read_scalar(data, base, intensity_field, endian))))
                if intensity_field is not None
                else 0
            )
            line = (
                int(round(float(_read_scalar(data, base, line_field, endian))))
                if line_field is not None
                else 0
            )
            absolute_time = None
            offset_time = 0
            if time_field is not None:
                raw_time = _read_scalar(data, base, time_field, endian)
                absolute_time, offset_time = _relative_time_ns(
                    raw_time, str(time_field.name), frame_time_ns
                )
            decoded.append(
                (
                    x,
                    y,
                    z,
                    min(255, max(0, reflectivity)),
                    min(255, max(0, line)),
                    absolute_time,
                    offset_time,
                )
            )

    absolute_times = [point[5] for point in decoded if point[5] is not None]
    if absolute_times:
        timebase = min(absolute_times)
        point_offsets = [
            (int(point[5]) - timebase) if point[5] is not None else point[6]
            for point in decoded
        ]
    else:
        timebase = frame_time_ns
        point_offsets = [point[6] for point in decoded]

    message = custom_msg_type()
    message.header = copy.copy(getattr(point_cloud, "header", None))
    if message.header is not None and frame_id:
        message.header.frame_id = frame_id
    message.timebase = max(0, int(timebase))
    message.point_num = len(decoded)
    message.lidar_id = 0
    message.rsvd = [0, 0, 0]
    message.points = []
    for decoded_point, offset_time in zip(decoded, point_offsets):
        x, y, z, reflectivity, line, _absolute, _relative = decoded_point
        point = custom_point_type()
        point.offset_time = min(0xFFFFFFFF, max(0, int(offset_time)))
        point.x = x
        point.y = y
        point.z = z
        point.reflectivity = reflectivity
        point.tag = 0
        point.line = line
        message.points.append(point)
    return message


class ManagedLivoxRelay:
    """Own one relay subprocess and expose its readiness to the Kit thread."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        *,
        ready_prefix: str,
        data_ready_prefix: str,
    ) -> None:
        self._process = process
        self._ready_prefix = ready_prefix
        self._ready = threading.Event()
        self._data_ready_prefix = data_ready_prefix
        self._data_ready = threading.Event()
        self._output_lock = threading.Lock()
        self._last_output = ""
        self._reader = threading.Thread(
            target=self._read_output,
            name="arena-livox-relay-output",
            daemon=True,
        )
        self._reader.start()

    @classmethod
    def start(
        cls,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        ready_prefix: str = "ARENA_LIVOX_RELAY_READY",
        data_ready_prefix: str = "ARENA_LIVOX_RELAY_DATA_READY",
    ) -> "ManagedLivoxRelay":
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        return cls(
            process,
            ready_prefix=ready_prefix,
            data_ready_prefix=data_ready_prefix,
        )

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        return self._process.poll()

    @property
    def last_output(self) -> str:
        with self._output_lock:
            return self._last_output

    def _read_output(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip()
                with self._output_lock:
                    self._last_output = line
                if line.startswith(self._ready_prefix):
                    self._ready.set()
                if line.startswith(self._data_ready_prefix):
                    self._data_ready.set()
                # Forward every child line. A traceback is multi-line; filtering
                # only its header discards the exception type and call site.
                print(f"[arena-livox-relay] {line}", file=sys.stderr, flush=True)
        finally:
            stream.close()

    def is_ready(self) -> bool:
        returncode = self._process.poll()
        if returncode is not None:
            detail = self.last_output or f"process exited with status {returncode}"
            raise RuntimeError(f"Livox relay failed before ready: {detail}")
        return self._ready.is_set()

    def wait_ready(self, *, timeout_s: float) -> bool:
        self._ready.wait(max(0.0, float(timeout_s)))
        return self.is_ready()

    def has_data(self) -> bool:
        self.is_ready()
        return self._data_ready.is_set()

    def wait_data_ready(self, *, timeout_s: float) -> bool:
        self._data_ready.wait(max(0.0, float(timeout_s)))
        self.is_ready()
        return self._data_ready.is_set()

    def is_healthy(self) -> bool:
        try:
            return self.is_ready() and self._data_ready.is_set()
        except RuntimeError:
            return False

    def stop(self, *, timeout_s: float = 3.0) -> None:
        if self._process.poll() is None:
            self._process.send_signal(signal.SIGINT)
            try:
                self._process.wait(timeout=max(0.1, float(timeout_s)))
            except subprocess.TimeoutExpired:
                # This is the exact child created and owned by this instance.
                self._process.kill()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    # 清理路径不该抛异常（会盖住真正的退出原因）；SIGKILL 已发出，
                    # 剩下的是内核回收，记一条 warning 就够。
                    print("warning: livox relay 未在 1s 内退出（SIGKILL 已发出）", file=sys.stderr)
        self._reader.join(timeout=1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-topic", default="/livox/lidar_points")
    parser.add_argument("--output-topic", default="/livox/lidar")
    parser.add_argument("--frame-id", default="livox_frame")
    parser.add_argument("--node-name", default="arena_livox_custom_msg_relay")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import rclpy
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2

    class LivoxRelayNode(Node):
        def __init__(self) -> None:
            super().__init__(args.node_name)
            self._data_ready = False
            self._publisher = self.create_publisher(
                CustomMsg, args.output_topic, qos_profile_sensor_data
            )
            self._subscription = self.create_subscription(
                PointCloud2,
                args.source_topic,
                self._on_pointcloud,
                qos_profile_sensor_data,
            )

        def _on_pointcloud(self, point_cloud: PointCloud2) -> None:
            try:
                message = pointcloud_to_custom_msg(
                    point_cloud,
                    custom_msg_type=CustomMsg,
                    custom_point_type=CustomPoint,
                    frame_id=args.frame_id,
                )
            except (TypeError, ValueError) as error:
                self.get_logger().warning(f"dropping invalid PointCloud2: {error}")
                return
            self._publisher.publish(message)
            if message.point_num > 0 and not self._data_ready:
                self._data_ready = True
                print("ARENA_LIVOX_RELAY_DATA_READY", flush=True)

    rclpy.init()
    node = LivoxRelayNode()
    print(
        f"ARENA_LIVOX_RELAY_READY source={args.source_topic} output={args.output_topic}",
        flush=True,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
