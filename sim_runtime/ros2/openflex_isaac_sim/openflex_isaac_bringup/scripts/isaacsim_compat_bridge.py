#!/usr/bin/env python3
"""Publish OpenFleX real-robot-compatible topics from Isaac Sim topics."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import (
    BatteryState,
    CameraInfo,
    CompressedImage,
    Image,
    Imu,
    JointState,
    LaserScan,
    PointCloud2,
    PointField,
)
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import Trigger

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional runtime dependency
    PILImage = None

try:
    from livox_ros_driver2.msg import CustomMsg, CustomPoint
except Exception:  # pragma: no cover - optional workspace package
    CustomMsg = None
    CustomPoint = None

try:
    from lift_slide_msgs.msg import MotorStatus
except Exception:  # pragma: no cover - optional workspace package
    MotorStatus = None

try:
    from interface.srv import RefineMap, SaveMaps, SavePoses
except Exception:  # pragma: no cover - optional workspace package
    RefineMap = None
    SaveMaps = None
    SavePoses = None


SIM_LIVOX_GRAVITY_MPS2 = 9.80665


def _topic_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        topics: list[str] = []
        for item in value:
            topics.extend(_topic_list(item))
        return topics
    return []


def _unique_topics(*values: object) -> list[str]:
    return list(dict.fromkeys(topic for value in values for topic in _topic_list(value)))


def _livox_driver_qos() -> QoSProfile:
    return QoSProfile(depth=256)


def _sim_lidar_input_qos() -> QoSProfile:
    """Match Isaac RTX LiDAR's reliable PointCloud2 writer."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _bounded_sensor_qos(depth: int = 1) -> QoSProfile:
    """Keep sensor relays latest-sample oriented and non-blocking."""
    return QoSProfile(
        depth=max(1, int(depth)),
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def _camera_info_topic(image_topic: str) -> str:
    if image_topic.endswith("/image_raw"):
        return image_topic[: -len("/image_raw")] + "/camera_info"
    if image_topic.endswith("/image"):
        return image_topic[: -len("/image")] + "/camera_info"
    return image_topic.rstrip("/") + "/camera_info"


_POINTFIELD_DTYPES = {
    PointField.INT8: np.dtype("i1"),
    PointField.UINT8: np.dtype("u1"),
    PointField.INT16: np.dtype("i2"),
    PointField.UINT16: np.dtype("u2"),
    PointField.INT32: np.dtype("i4"),
    PointField.UINT32: np.dtype("u4"),
    PointField.FLOAT32: np.dtype("f4"),
    PointField.FLOAT64: np.dtype("f8"),
}


@dataclass(frozen=True)
class DecodedLidarPoint:
    x: float
    y: float
    z: float
    intensity: float
    timestamp_ns: int | None
    emitter_id: int | None
    channel_id: int | None


def _pointfield_dtype(field: PointField, is_bigendian: bool) -> np.dtype:
    try:
        dtype = _POINTFIELD_DTYPES[int(field.datatype)]
    except KeyError as exc:
        raise ValueError(f"unsupported PointCloud2 datatype {field.datatype}") from exc
    if int(field.count) != 1:
        raise ValueError(f"PointCloud2 field '{field.name}' must have count=1")
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if is_bigendian else "<")
    return dtype


def _read_pointcloud_field(cloud: PointCloud2, field: PointField) -> np.ndarray:
    height = max(0, int(cloud.height))
    width = max(0, int(cloud.width))
    if height == 0 or width == 0:
        return np.empty(0, dtype=_pointfield_dtype(field, bool(cloud.is_bigendian)))
    point_step = int(cloud.point_step)
    row_step = int(cloud.row_step) or point_step * width
    if point_step <= 0 or row_step < point_step * width:
        raise ValueError(
            f"invalid PointCloud2 stride: point_step={point_step}, row_step={row_step}, width={width}"
        )
    dtype = _pointfield_dtype(field, bool(cloud.is_bigendian))
    if int(field.offset) < 0 or int(field.offset) + dtype.itemsize > point_step:
        raise ValueError(
            f"PointCloud2 field '{field.name}' exceeds point_step={point_step}"
        )
    data = bytes(cloud.data)
    final_offset = (height - 1) * row_step + (width - 1) * point_step + int(field.offset) + dtype.itemsize
    if final_offset > len(data):
        raise ValueError(
            f"PointCloud2 data is truncated for field '{field.name}': "
            f"need {final_offset} bytes, got {len(data)}"
        )
    return np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=data,
        offset=int(field.offset),
        strides=(row_step, point_step),
    ).reshape(-1)


def _read_pointcloud_field_array(cloud: PointCloud2, field: PointField) -> np.ndarray:
    """Read a fixed-size PointCloud2 array field while honoring row padding."""
    height = max(0, int(cloud.height))
    width = max(0, int(cloud.width))
    if height == 0 or width == 0:
        scalar_dtype = _POINTFIELD_DTYPES.get(int(field.datatype))
        if scalar_dtype is None:
            raise ValueError(f"unsupported PointCloud2 datatype {field.datatype}")
        if scalar_dtype.itemsize > 1:
            scalar_dtype = scalar_dtype.newbyteorder(">" if cloud.is_bigendian else "<")
        return np.empty((0, int(field.count)), dtype=scalar_dtype)
    point_step = int(cloud.point_step)
    row_step = int(cloud.row_step) or point_step * width
    if point_step <= 0 or row_step < point_step * width:
        raise ValueError(
            f"invalid PointCloud2 stride: point_step={point_step}, row_step={row_step}, width={width}"
        )
    if int(field.count) <= 0:
        raise ValueError(f"PointCloud2 field '{field.name}' must have a positive count")
    scalar_dtype = _pointfield_dtype(
        PointField(datatype=field.datatype, count=1), bool(cloud.is_bigendian)
    )
    field_bytes = scalar_dtype.itemsize * int(field.count)
    if int(field.offset) < 0 or int(field.offset) + field_bytes > point_step:
        raise ValueError(f"PointCloud2 field '{field.name}' exceeds point_step={point_step}")
    data = bytes(cloud.data)
    final_offset = (
        (height - 1) * row_step
        + (width - 1) * point_step
        + int(field.offset)
        + field_bytes
    )
    if final_offset > len(data):
        raise ValueError(
            f"PointCloud2 data is truncated for field '{field.name}': "
            f"need {final_offset} bytes, got {len(data)}"
        )
    return np.ndarray(
        shape=(height, width, int(field.count)),
        dtype=scalar_dtype,
        buffer=data,
        offset=int(field.offset),
        strides=(row_step, point_step, scalar_dtype.itemsize),
    ).reshape(-1, int(field.count))


def _decode_lidar_points(
    cloud: PointCloud2,
    *,
    require_livox_metadata: bool,
    max_points: int | None = None,
) -> list[DecodedLidarPoint]:
    fields = {field.name: field for field in cloud.fields}
    required = {"x", "y", "z"}
    if require_livox_metadata:
        required.update({"emitter_id", "channel_id"})
        has_split_timestamp = {"timestamp_0", "timestamp_1"}.issubset(fields)
        has_array_timestamp = "timestamp" in fields
        if not has_split_timestamp and not has_array_timestamp:
            required.add("timestamp")
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"PointCloud2 is missing required LiDAR fields: {', '.join(missing)}")

    used_names = required | {
        "intensity", "timestamp", "timestamp_0", "timestamp_1", "emitter_id", "channel_id"
    }
    arrays = {
        name: _read_pointcloud_field(cloud, field)
        for name, field in fields.items()
        if name in used_names and int(field.count) == 1
    }
    if "timestamp" in fields and int(fields["timestamp"].count) != 1:
        arrays["timestamp"] = _read_pointcloud_field_array(cloud, fields["timestamp"])
    xyz = np.column_stack((arrays["x"], arrays["y"], arrays["z"])).astype(np.float64, copy=False)
    valid = np.isfinite(xyz).all(axis=1)
    if "intensity" in arrays:
        intensity_values = arrays["intensity"].astype(np.float64, copy=False)
        valid &= np.isfinite(intensity_values)
    else:
        intensity_values = np.zeros(len(xyz), dtype=np.float64)

    timestamp_values: np.ndarray | None = None
    if "timestamp_0" in arrays and "timestamp_1" in arrays:
        timestamp_values = (
            arrays["timestamp_0"].astype(np.uint64, copy=False)
            | (arrays["timestamp_1"].astype(np.uint64, copy=False) << np.uint64(32))
        )
    elif "timestamp" in arrays:
        timestamp_array = arrays["timestamp"]
        if int(fields["timestamp"].datatype) != PointField.UINT32:
            raise ValueError("PointCloud2 timestamp field must use UINT32 values")
        if timestamp_array.ndim != 2 or timestamp_array.shape[1] != 2:
            raise ValueError("PointCloud2 timestamp field must contain exactly two UINT32 values")
        timestamp_values = (
            timestamp_array[:, 0].astype(np.uint64, copy=False)
            | (timestamp_array[:, 1].astype(np.uint64, copy=False) << np.uint64(32))
        )
    if require_livox_metadata and timestamp_values is None:
        raise ValueError("PointCloud2 must contain timestamp UINT32[2] or timestamp_0/timestamp_1")

    emitter_values = arrays.get("emitter_id")
    channel_values = arrays.get("channel_id")
    valid_indices = np.flatnonzero(valid)
    if timestamp_values is not None and len(valid_indices):
        valid_timestamps = timestamp_values[valid_indices]
        timebase = int(valid_timestamps.min())
        span = int(valid_timestamps.max()) - timebase
        if span > np.iinfo(np.uint32).max:
            raise ValueError(f"LiDAR frame timestamp span {span} ns exceeds uint32 offset_time")
    else:
        timebase = None

    # Isaac usually emits points in acquisition order. Preserve it, and only
    # repair a stream when the actual timestamp sequence is out of order.
    if timestamp_values is not None and len(valid_indices) > 1:
        selected_timestamps = timestamp_values[valid_indices]
        if np.any(selected_timestamps[1:] < selected_timestamps[:-1]):
            valid_indices = valid_indices[np.argsort(selected_timestamps, kind="stable")]
    if max_points is not None:
        point_limit = max(0, int(max_points))
        if point_limit == 0:
            valid_indices = valid_indices[:0]
        elif len(valid_indices) > point_limit:
            # Preserve the complete scan and its time span. Taking only the
            # first N points turns a rotary frame into a narrow temporal and
            # angular slice, which is invalid input for LiDAR odometry.
            sample_positions = np.linspace(
                0, len(valid_indices) - 1, num=point_limit, dtype=np.int64
            )
            valid_indices = valid_indices[sample_positions]

    points: list[DecodedLidarPoint] = []
    for index in valid_indices:
        raw_intensity = float(intensity_values[index])
        raw_intensity *= 255.0
        timestamp_ns = int(timestamp_values[index]) if timestamp_values is not None else None
        points.append(
            DecodedLidarPoint(
                x=float(xyz[index, 0]),
                y=float(xyz[index, 1]),
                z=float(xyz[index, 2]),
                intensity=max(0.0, min(255.0, raw_intensity)),
                timestamp_ns=timestamp_ns,
                emitter_id=int(emitter_values[index]) if emitter_values is not None else None,
                channel_id=int(channel_values[index]) if channel_values is not None else None,
            )
        )
    return points


@dataclass
class ImageRelay:
    name: str
    source_topic: str
    image_pubs: list
    camera_info_pubs: list
    compressed_pubs: list
    last_compressed_publish_time: float = 0.0
    subscription: object | None = None
    compression_in_flight: bool = False


class OpenFlexSimCompatBridge(Node):
    def __init__(self) -> None:
        super().__init__("openflex_sim_compat_bridge")
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)

        self.declare_parameter("sim_color_topic", "/openflex/d435_color_optical_frame/color/image")
        self.declare_parameter("sim_depth_topic", "/openflex/d435_depth_optical_frame/depth/image")
        self.declare_parameter("sim_base_color_topic", "")
        self.declare_parameter("sim_base_depth_topic", "")
        self.declare_parameter("sim_head_color_topic", "/openflex/head_yaw_link/head/color/image")
        self.declare_parameter("sim_head_depth_topic", "/openflex/head_yaw_link/head/depth/image")
        self.declare_parameter("sim_left_color_topic", "/openflex/openarmx_left_hand/left/color/image")
        self.declare_parameter("sim_left_depth_topic", "/openflex/openarmx_left_hand/left/depth/image")
        self.declare_parameter("sim_right_color_topic", "/openflex/openarmx_right_hand/right/color/image")
        self.declare_parameter("sim_right_depth_topic", "/openflex/openarmx_right_hand/right/depth/image")
        self.declare_parameter("sim_lidar_topic", "/openflex/livox_frame/lidar")
        self.declare_parameter("sim_scan_topic", "/openflex/livox_scan_frame/scan")
        self.declare_parameter("sim_odom_topic", "/odom")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("cam_base_color_topics", ["/cam_base/color/image"])
        self.declare_parameter("cam_base_depth_topics", ["/cam_base/depth/image"])
        self.declare_parameter("cam_head_color_topics", ["/cam_head/color/image"])
        self.declare_parameter("cam_head_depth_topics", ["/cam_head/depth/image"])
        self.declare_parameter("cam_left_color_topics", ["/cam_left/color/image"])
        self.declare_parameter("cam_left_depth_topics", ["/cam_left/depth/image"])
        self.declare_parameter("cam_right_color_topics", ["/cam_right/color/image"])
        self.declare_parameter("cam_right_depth_topics", ["/cam_right/depth/image"])
        self.declare_parameter("cam_base_color_compressed_topics", ["/cam_base/color/image/compressed"])
        self.declare_parameter("cam_head_color_compressed_topics", ["/cam_head/color/image/compressed"])
        self.declare_parameter("cam_left_color_compressed_topics", ["/cam_left/color/image/compressed"])
        self.declare_parameter("cam_right_color_compressed_topics", ["/cam_right/color/image/compressed"])
        self.declare_parameter("color_topics", ["/camera/color/image_raw", "/vision/color/image_raw"])
        self.declare_parameter("depth_topics", ["/camera/depth/image_raw", "/vision/depth/image_raw"])
        self.declare_parameter("camera_compressed_max_rate_hz", 15.0)
        self.declare_parameter("camera_compressed_jpeg_quality", 80)
        self.declare_parameter("sensor_queue_depth", 1)
        self.declare_parameter("scan_topics", ["/scan"])
        self.declare_parameter(
            "odom_topics",
            ["/odom_safe", "/fastlio2/lio_odom", "/pgo/offset"],
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter(
            "pointcloud_topics",
            ["/livox/lidar_points", "/fastlio2/body_cloud"],
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("livox_lidar_topic", "/livox/lidar")
        self.declare_parameter("livox_lidar_mode", "pointcloud")
        self.declare_parameter("camera_horizontal_fov_deg", 60.0)
        self.declare_parameter("livox_frame_id", "livox_frame")
        self.declare_parameter("livox_scan_period", 0.1)
        self.declare_parameter("livox_line_count", 4)
        self.declare_parameter("livox_max_points", 30000)
        self.declare_parameter("publish_scan_from_pointcloud", True)
        self.declare_parameter("scan_angle_min", -math.pi)
        self.declare_parameter("scan_angle_max", math.pi)
        self.declare_parameter("scan_angle_increment", 0.0058)
        self.declare_parameter("scan_range_min", 0.05)
        self.declare_parameter("scan_range_max", 20.0)
        self.declare_parameter("scan_height_min", -0.35)
        self.declare_parameter("scan_height_max", 1.5)
        self.declare_parameter("publish_livox_imu", False)
        self.declare_parameter("publish_lift_status", True)
        self.declare_parameter("publish_mapping_stub_services", True)
        self.declare_parameter("relay_cmd_vel_safe", True)
        self.declare_parameter("publish_battery_state", True)
        self.declare_parameter("battery_state_topic", "/battery_state")
        self.declare_parameter("battery_percentage", 0.85)
        self.declare_parameter("battery_voltage", 48.0)
        self.declare_parameter("battery_publish_rate_hz", 1.0)
        self.declare_parameter("sensor_only_mode", False)

        self._sensor_only_mode = bool(self.get_parameter("sensor_only_mode").value)

        self._image_relays: list[ImageRelay] = []
        self._compression_warning_keys: set[str] = set()
        # Compression is intentionally outside the ROS callback/executor.  A
        # single in-flight job per relay gives a bounded latest-sample queue:
        # slow JPEG work cannot back-pressure Isaac or accumulate old frames.
        self._sensor_workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="openflex_sensor")
        self._lidar_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openflex_lidar")
        self._lidar_in_flight = False
        self._livox_driver_qos = _livox_driver_qos()
        self._sim_lidar_input_qos = _sim_lidar_input_qos()
        self._sensor_qos = _bounded_sensor_qos(self.get_parameter("sensor_queue_depth").value)
        self._image_subscription_timer = None
        if not self._sensor_only_mode:
            self._create_camera_relays()
            self._image_subscription_timer = self.create_timer(
                0.5,
                self._refresh_image_subscriptions,
                clock=self._steady_clock,
            )

        self.scan_pubs = [
            self.create_publisher(LaserScan, topic, self._sensor_qos)
            for topic in _topic_list(self.get_parameter("scan_topics").value)
        ]
        self.odom_pubs = [
            self.create_publisher(Odometry, topic, self._sensor_qos)
            for topic in _topic_list(self.get_parameter("odom_topics").value)
            if not self._sensor_only_mode
        ]

        livox_mode = str(self.get_parameter("livox_lidar_mode").value).lower()
        livox_topic = str(self.get_parameter("livox_lidar_topic").value)
        pointcloud_topics = _topic_list(self.get_parameter("pointcloud_topics").value)
        publish_livox_pointcloud = livox_mode == "pointcloud" or (
            livox_mode == "auto" and CustomMsg is None
        )
        if publish_livox_pointcloud:
            pointcloud_topics.insert(0, livox_topic)

        self.pointcloud_pubs = [
            self.create_publisher(PointCloud2, topic, self._sensor_qos)
            for topic in dict.fromkeys(pointcloud_topics)
        ]
        self.livox_pub = None
        if livox_mode == "custom" and CustomMsg is not None:
            self.livox_pub = self.create_publisher(CustomMsg, livox_topic, self._livox_driver_qos)
        elif livox_mode == "custom":
            self.get_logger().warn("livox_ros_driver2 is not available; /livox/lidar CustomMsg is disabled")
        elif livox_mode == "auto" and CustomMsg is not None:
            self.livox_pub = self.create_publisher(CustomMsg, livox_topic, self._livox_driver_qos)
        elif livox_mode == "auto":
            self.get_logger().warn(
                "livox_ros_driver2 is not available; /livox/lidar falls back to PointCloud2 in auto mode"
            )
        elif livox_mode != "pointcloud":
            self.get_logger().warn(
                f"Unsupported livox_lidar_mode '{livox_mode}', expected 'auto', 'custom', or 'pointcloud'"
            )

        self.latest_lift_position = 0.0
        self.latest_lift_velocity = 0.0
        self._lift_target_position = 0.0
        self._lift_jog_velocity = 0.0
        self._lift_position_pub = (
            self.create_publisher(Float64MultiArray, "/lift_position_controller/commands", 10)
            if not self._sensor_only_mode
            else None
        )
        self._latest_odom_stamp_ns: int | None = None
        self._latest_odom_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._latest_odom_angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._latest_imu_linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_lidar_stamp_ns: int | None = None
        self._last_imu_stamp_ns: int | None = None
        self._odom_imu_warning_emitted = False
        self.homing_state_pub = (
            self.create_publisher(String, "/lift_slide_driver/homing_state", 10)
            if not self._sensor_only_mode
            else None
        )
        self.limit_switch_state_pub = (
            self.create_publisher(String, "/lift_slide_driver/limit_switch_state", 10)
            if not self._sensor_only_mode
            else None
        )
        self.motor_status_pub = (
            self.create_publisher(MotorStatus, "/lift_slide_driver/motor_status", qos_profile_sensor_data)
            if (
                not self._sensor_only_mode
                and MotorStatus is not None
                and bool(self.get_parameter("publish_lift_status").value)
            )
            else None
        )
        self.imu_pub = (
            self.create_publisher(Imu, "/livox/imu", self._livox_driver_qos)
            if bool(self.get_parameter("publish_livox_imu").value)
            else None
        )
        self.battery_state_pub = None
        self._battery_state_timer = None
        if bool(self.get_parameter("publish_battery_state").value):
            battery_topic = self._param_str("battery_state_topic", "/battery_state")
            self.battery_state_pub = self.create_publisher(
                BatteryState, battery_topic, qos_profile_sensor_data
            )
            battery_publish_rate_hz = float(self.get_parameter("battery_publish_rate_hz").value)
            if battery_publish_rate_hz > 0.0:
                self._battery_state_timer = self.create_timer(
                    1.0 / battery_publish_rate_hz,
                    self._publish_battery_state,
                    clock=self._steady_clock,
                )
            else:
                self.get_logger().warn(
                    "battery_publish_rate_hz is not positive; /battery_state publishing is disabled"
                )

        self._sim_lidar_topic = str(self.get_parameter("sim_lidar_topic").value)
        self._sim_scan_topic = str(self.get_parameter("sim_scan_topic").value)
        self._lidar_subscription = None
        self._scan_subscription = None
        self._high_bandwidth_subscription_timer = self.create_timer(
            0.5,
            self._refresh_high_bandwidth_subscriptions,
            clock=self._steady_clock,
        )
        if not self._sensor_only_mode:
            self.create_subscription(
                Odometry,
                str(self.get_parameter("sim_odom_topic").value),
                self._on_odom,
                10,
            )
            self.create_subscription(
                JointState,
                str(self.get_parameter("joint_states_topic").value),
                self._on_joint_states,
                10,
            )
            # RViz's lift panel uses the real robot's manual-controller topics.
            # Isaac's URDF path exposes a standard position controller instead,
            # so translate step/jog commands into position targets here.
            self.create_subscription(
                Float64MultiArray,
                "/lift_manual_position_controller/step_command",
                self._on_lift_step_command,
                10,
            )
            self.create_subscription(
                Float64,
                "/lift_manual_position_controller/jog_command",
                self._on_lift_jog_command,
                10,
            )
        self.cmd_vel_pub = None
        if not self._sensor_only_mode and bool(self.get_parameter("relay_cmd_vel_safe").value):
            self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.create_subscription(Twist, "/cmd_vel_safe", self.cmd_vel_pub.publish, 10)

        if not self._sensor_only_mode:
            self._create_lift_services()
            if bool(self.get_parameter("publish_mapping_stub_services").value):
                self._create_mapping_stub_services()
        # Publish IMU independently of the slow wall-clock /odom relay.  The
        # steady timer guarantees the real-driver contract of >=100 Hz, while
        # each message still carries the newest simulation timestamp and never
        # goes backwards.
        if self.imu_pub is not None:
            self.create_timer(1.0 / 100.0, self._publish_imu, clock=self._steady_clock)
        if not self._sensor_only_mode:
            self.create_timer(0.1, self._publish_lift_jog_target, clock=self._steady_clock)
            self.create_timer(0.1, self._publish_lift_status, clock=self._steady_clock)

        self.get_logger().info(
            "OpenFleX sim compatibility bridge started: "
            f"mode={'sensor-only' if self._sensor_only_mode else 'robot-compat'}, "
            f"LiDAR mode={livox_mode}, lift status={'on' if self.motor_status_pub else 'off'}"
        )

    def _param_str(self, name: str, fallback: str = "") -> str:
        value = str(self.get_parameter(name).value).strip()
        return value or fallback

    def _create_camera_relays(self) -> None:
        legacy_color_topics = self.get_parameter("color_topics").value
        legacy_depth_topics = self.get_parameter("depth_topics").value
        legacy_color_source = str(self.get_parameter("sim_color_topic").value)
        legacy_depth_source = str(self.get_parameter("sim_depth_topic").value)

        self._create_image_relay(
            "base_color",
            self._param_str("sim_base_color_topic", legacy_color_source),
            _unique_topics(self.get_parameter("cam_base_color_topics").value, legacy_color_topics),
            self.get_parameter("cam_base_color_compressed_topics").value,
        )
        self._create_image_relay(
            "base_depth",
            self._param_str("sim_base_depth_topic", legacy_depth_source),
            _unique_topics(self.get_parameter("cam_base_depth_topics").value, legacy_depth_topics),
            [],
        )
        self._create_image_relay(
            "head_color",
            self._param_str("sim_head_color_topic"),
            self.get_parameter("cam_head_color_topics").value,
            self.get_parameter("cam_head_color_compressed_topics").value,
        )
        self._create_image_relay(
            "head_depth",
            self._param_str("sim_head_depth_topic"),
            self.get_parameter("cam_head_depth_topics").value,
            [],
        )
        self._create_image_relay(
            "left_color",
            self._param_str("sim_left_color_topic"),
            self.get_parameter("cam_left_color_topics").value,
            self.get_parameter("cam_left_color_compressed_topics").value,
        )
        self._create_image_relay(
            "left_depth",
            self._param_str("sim_left_depth_topic"),
            self.get_parameter("cam_left_depth_topics").value,
            [],
        )
        self._create_image_relay(
            "right_color",
            self._param_str("sim_right_color_topic"),
            self.get_parameter("cam_right_color_topics").value,
            self.get_parameter("cam_right_color_compressed_topics").value,
        )
        self._create_image_relay(
            "right_depth",
            self._param_str("sim_right_depth_topic"),
            self.get_parameter("cam_right_depth_topics").value,
            [],
        )

    def _create_image_relay(
        self,
        name: str,
        source_topic: str,
        image_topics_value: object,
        compressed_topics_value: object,
    ) -> None:
        image_topics = _topic_list(image_topics_value)
        compressed_topics = _topic_list(compressed_topics_value)
        if not source_topic or (not image_topics and not compressed_topics):
            return

        relay = ImageRelay(
            name=name,
            source_topic=source_topic,
            image_pubs=[
                self.create_publisher(Image, topic, self._sensor_qos)
                for topic in image_topics
            ],
            camera_info_pubs=[
                self.create_publisher(CameraInfo, _camera_info_topic(topic), self._sensor_qos)
                for topic in image_topics
            ],
            compressed_pubs=[
                self.create_publisher(CompressedImage, topic, self._sensor_qos)
                for topic in compressed_topics
            ],
        )
        self._image_relays.append(relay)

    @staticmethod
    def _relay_has_subscribers(relay: ImageRelay) -> bool:
        return any(
            pub.get_subscription_count() > 0
            for pub in (*relay.image_pubs, *relay.camera_info_pubs, *relay.compressed_pubs)
        )

    def _refresh_image_subscriptions(self) -> None:
        for relay in self._image_relays:
            needed = self._relay_has_subscribers(relay)
            if needed and relay.subscription is None:
                relay.subscription = self.create_subscription(
                    Image,
                    relay.source_topic,
                    lambda image, image_relay=relay: self._on_image(image_relay, image),
                    getattr(self, "_sensor_qos", qos_profile_sensor_data),
                )
            elif not needed and relay.subscription is not None:
                self.destroy_subscription(relay.subscription)
                relay.subscription = None

    @staticmethod
    def _any_subscribers(publishers: Iterable) -> bool:
        return any(pub.get_subscription_count() > 0 for pub in publishers)

    def _refresh_high_bandwidth_subscriptions(self) -> None:
        lidar_publishers = list(self.pointcloud_pubs)
        if self.livox_pub is not None:
            lidar_publishers.append(self.livox_pub)
        lidar_needed = self._any_subscribers(lidar_publishers) or (
            bool(self.get_parameter("publish_scan_from_pointcloud").value)
            and self._any_subscribers(self.scan_pubs)
        )
        if lidar_needed and self._lidar_subscription is None:
            self._lidar_subscription = self.create_subscription(
                PointCloud2,
                self._sim_lidar_topic,
                self._on_lidar,
                getattr(self, "_sim_lidar_input_qos", _sim_lidar_input_qos()),
            )
        elif not lidar_needed and self._lidar_subscription is not None:
            self.destroy_subscription(self._lidar_subscription)
            self._lidar_subscription = None

        scan_needed = bool(self._sim_scan_topic) and self._any_subscribers(self.scan_pubs)
        if scan_needed and self._scan_subscription is None:
            self._scan_subscription = self.create_subscription(
                LaserScan,
                self._sim_scan_topic,
                self._on_scan,
                qos_profile_sensor_data,
            )
        elif not scan_needed and self._scan_subscription is not None:
            self.destroy_subscription(self._scan_subscription)
            self._scan_subscription = None

    def _camera_info(self, image: Image) -> CameraInfo:
        width = int(image.width)
        height = int(image.height)
        hfov = math.radians(float(self.get_parameter("camera_horizontal_fov_deg").value))
        fx = width / (2.0 * math.tan(hfov / 2.0)) if width > 0 else 0.0
        fy = fx
        cx = (width - 1.0) / 2.0 if width > 0 else 0.0
        cy = (height - 1.0) / 2.0 if height > 0 else 0.0

        info = CameraInfo()
        info.header = image.header
        info.width = width
        info.height = height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _on_image(self, relay: ImageRelay, image: Image) -> None:
        image_pubs = [pub for pub in relay.image_pubs if pub.get_subscription_count() > 0]
        for pub in image_pubs:
            pub.publish(image)

        camera_info_pubs = [
            pub for pub in relay.camera_info_pubs if pub.get_subscription_count() > 0
        ]
        if camera_info_pubs:
            info = self._camera_info(image)
            for pub in camera_info_pubs:
                pub.publish(info)

        if (
            relay.compressed_pubs
            and any(pub.get_subscription_count() > 0 for pub in relay.compressed_pubs)
            and self._should_publish_compressed(relay)
        ):
            if relay.compression_in_flight:
                return
            relay.compression_in_flight = True
            future = self._sensor_workers.submit(self._compress_color_image, relay.name, image)
            future.add_done_callback(
                lambda completed, image_relay=relay: self._publish_compressed_result(image_relay, completed)
            )

    def _publish_compressed_result(self, relay: ImageRelay, future) -> None:
        relay.compression_in_flight = False
        try:
            compressed = future.result()
        except Exception as exc:  # keep worker failures out of the ROS executor
            self.get_logger().warn(f"JPEG worker failed for {relay.name}: {exc}")
            return
        if compressed is not None:
            for pub in relay.compressed_pubs:
                if pub.get_subscription_count() > 0:
                    pub.publish(compressed)

    def _should_publish_compressed(self, relay: ImageRelay) -> bool:
        max_rate_hz = float(self.get_parameter("camera_compressed_max_rate_hz").value)
        if max_rate_hz <= 0.0:
            return True
        now = time.monotonic()
        min_period = 1.0 / max_rate_hz
        if relay.last_compressed_publish_time and now - relay.last_compressed_publish_time < min_period:
            return False
        relay.last_compressed_publish_time = now
        return True

    def _warn_compression_once(self, key: str, message: str) -> None:
        if key in self._compression_warning_keys:
            return
        self._compression_warning_keys.add(key)
        self.get_logger().warn(message)

    def _compress_color_image(self, relay_name: str, image: Image) -> CompressedImage | None:
        if PILImage is None:
            self._warn_compression_once(
                "no_pillow",
                "Pillow is not available; simulated color compressed image topics are disabled",
            )
            return None

        width = int(image.width)
        height = int(image.height)
        if width <= 0 or height <= 0:
            return None

        pil_image = self._image_to_pil(relay_name, image, width, height)
        if pil_image is None:
            return None
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")

        output = BytesIO()
        quality = max(1, min(100, int(self.get_parameter("camera_compressed_jpeg_quality").value)))
        pil_image.save(output, format="JPEG", quality=quality)

        compressed = CompressedImage()
        compressed.header = image.header
        compressed.format = "jpeg"
        compressed.data = output.getvalue()
        return compressed

    def _image_to_pil(self, relay_name: str, image: Image, width: int, height: int):
        encoding = image.encoding.lower()
        formats = {
            "rgb8": ("RGB", "RGB", 3),
            "bgr8": ("RGB", "BGR", 3),
            "rgba8": ("RGBA", "RGBA", 4),
            "bgra8": ("RGBA", "BGRA", 4),
            "mono8": ("L", "L", 1),
            "8uc1": ("L", "L", 1),
        }
        if encoding not in formats:
            self._warn_compression_once(
                f"unsupported:{relay_name}:{encoding}",
                f"Cannot JPEG-compress {relay_name}: unsupported image encoding '{image.encoding}'",
            )
            return None

        mode, raw_mode, bytes_per_pixel = formats[encoding]
        expected_step = width * bytes_per_pixel
        step = int(image.step) if int(image.step) > 0 else expected_step
        raw = bytes(image.data)
        expected_size = step * height
        if len(raw) < expected_size:
            self._warn_compression_once(
                f"short:{relay_name}:{encoding}",
                f"Cannot JPEG-compress {relay_name}: image data is shorter than width/height/step",
            )
            return None

        if step != expected_step:
            raw = b"".join(
                raw[row * step : row * step + expected_step]
                for row in range(height)
            )
        return PILImage.frombytes(mode, (width, height), raw, "raw", raw_mode)

    def _on_lidar(self, cloud: PointCloud2) -> None:
        stamp_ns = int(cloud.header.stamp.sec) * 1_000_000_000 + int(cloud.header.stamp.nanosec)
        if stamp_ns == getattr(self, "_last_lidar_stamp_ns", None):
            return
        self._last_lidar_stamp_ns = stamp_ns

        frame_id = str(self.get_parameter("livox_frame_id").value)
        if frame_id:
            cloud.header.frame_id = frame_id
        for pub in self.pointcloud_pubs:
            pub.publish(cloud)
        need_scan = (
            bool(self.get_parameter("publish_scan_from_pointcloud").value)
            and any(pub.get_subscription_count() > 0 for pub in self.scan_pubs)
        )
        if (self.livox_pub is not None or need_scan) and not self._lidar_in_flight:
            self._lidar_in_flight = True
            future = self._lidar_worker.submit(
                self._convert_lidar_outputs, cloud, need_scan
            )
            future.add_done_callback(self._publish_lidar_outputs)

    def _convert_lidar_outputs(self, cloud: PointCloud2, need_scan: bool):
        need_livox = self.livox_pub is not None
        max_points = max(1, int(self.get_parameter("livox_max_points").value))
        if need_livox and not need_scan:
            # The CustomMsg-only path is the normal FAST-LIO path.  Avoid
            # constructing an intermediate DecodedLidarPoint object for every
            # point; decode the numeric arrays once and construct ROS points
            # directly below.
            livox = self._livox_from_cloud_fast(cloud, max_points=max_points)
            return livox, None

        points = _decode_lidar_points(
            cloud,
            require_livox_metadata=need_livox,
            max_points=max_points,
        )

        if need_scan:
            angle_min = float(self.get_parameter("scan_angle_min").value)
            angle_max = float(self.get_parameter("scan_angle_max").value)
            angle_increment = max(1e-6, float(self.get_parameter("scan_angle_increment").value))
            range_min = float(self.get_parameter("scan_range_min").value)
            range_max = float(self.get_parameter("scan_range_max").value)
            height_min = float(self.get_parameter("scan_height_min").value)
            height_max = float(self.get_parameter("scan_height_max").value)
            bin_count = max(1, int(math.ceil((angle_max - angle_min) / angle_increment)))
            ranges = [math.inf] * bin_count
            intensities = [0.0] * bin_count

        for point in points:
            if need_scan:
                if point.z < height_min or point.z > height_max:
                    continue
                distance = math.hypot(point.x, point.y)
                if distance < range_min or distance > range_max:
                    continue
                angle = math.atan2(point.y, point.x)
                if angle < angle_min or angle >= angle_max:
                    continue
                bin_index = int((angle - angle_min) / angle_increment)
                if 0 <= bin_index < bin_count and distance < ranges[bin_index]:
                    ranges[bin_index] = float(distance)
                    intensities[bin_index] = point.intensity

        livox = self._livox_from_points(cloud, points) if need_livox else None
        scan = self._scan_from_bins(
            cloud, angle_min, angle_max, angle_increment, range_min, range_max,
            ranges, intensities,
        ) if need_scan else None
        return livox, scan

    def _livox_from_cloud_fast(self, cloud: PointCloud2, *, max_points: int):
        fields = {field.name: field for field in cloud.fields}
        required = {"x", "y", "z", "emitter_id", "channel_id"}
        if {"timestamp_0", "timestamp_1"}.issubset(fields):
            timestamp_name = "split"
        elif "timestamp" in fields:
            timestamp_name = "array"
        else:
            raise ValueError("PointCloud2 is missing Livox timestamp fields")
        missing = sorted(required - fields.keys())
        if missing:
            raise ValueError(f"PointCloud2 is missing required LiDAR fields: {', '.join(missing)}")

        arrays = {
            name: _read_pointcloud_field(cloud, fields[name])
            for name in ("x", "y", "z", "emitter_id", "channel_id")
        }
        if "intensity" in fields:
            intensity_values = _read_pointcloud_field(cloud, fields["intensity"])
        else:
            intensity_values = np.zeros(len(arrays["x"]), dtype=np.float64)
        if timestamp_name == "split":
            timestamp_values = (
                _read_pointcloud_field(cloud, fields["timestamp_0"]).astype(np.uint64, copy=False)
                | (
                    _read_pointcloud_field(cloud, fields["timestamp_1"]).astype(np.uint64, copy=False)
                    << np.uint64(32)
                )
            )
        else:
            timestamp_array = _read_pointcloud_field_array(cloud, fields["timestamp"])
            if int(fields["timestamp"].datatype) != PointField.UINT32:
                raise ValueError("PointCloud2 timestamp field must use UINT32 values")
            if timestamp_array.ndim != 2 or timestamp_array.shape[1] != 2:
                raise ValueError("PointCloud2 timestamp field must contain exactly two UINT32 values")
            timestamp_values = (
                timestamp_array[:, 0].astype(np.uint64, copy=False)
                | (timestamp_array[:, 1].astype(np.uint64, copy=False) << np.uint64(32))
            )

        x_values = arrays["x"].astype(np.float64, copy=False)
        y_values = arrays["y"].astype(np.float64, copy=False)
        z_values = arrays["z"].astype(np.float64, copy=False)
        intensity_values = intensity_values.astype(np.float64, copy=False)
        valid = (
            np.isfinite(x_values)
            & np.isfinite(y_values)
            & np.isfinite(z_values)
            & np.isfinite(intensity_values)
        )
        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            raise ValueError("LiDAR PointCloud2 contains no finite points")
        selected_timestamps = timestamp_values[valid_indices]
        timebase = int(selected_timestamps.min())
        span = int(selected_timestamps.max()) - timebase
        if span > np.iinfo(np.uint32).max:
            raise ValueError(f"LiDAR frame timestamp span {span} ns exceeds uint32 offset_time")
        if len(valid_indices) > 1 and np.any(selected_timestamps[1:] < selected_timestamps[:-1]):
            valid_indices = valid_indices[np.argsort(selected_timestamps, kind="stable")]
        if len(valid_indices) > max_points:
            sample_positions = np.linspace(0, len(valid_indices) - 1, num=max_points, dtype=np.int64)
            valid_indices = valid_indices[sample_positions]

        line_count = max(1, int(self.get_parameter("livox_line_count").value))
        # Convert the selected columns once before constructing ROS messages.
        # Repeated NumPy scalar conversion inside the point loop is expensive
        # for a 30k-point frame and needlessly holds the executor thread.
        selected_timestamps = timestamp_values[valid_indices]
        selected_x = x_values[valid_indices].tolist()
        selected_y = y_values[valid_indices].tolist()
        selected_z = z_values[valid_indices].tolist()
        selected_reflectivity = np.clip(
            np.rint(intensity_values[valid_indices] * 255.0), 0, 255
        ).astype(np.uint8).tolist()
        selected_lines = np.mod(
            arrays["emitter_id"][valid_indices].astype(np.uint64, copy=False), line_count
        ).astype(np.uint8).tolist()
        selected_offsets = (selected_timestamps - np.uint64(timebase)).astype(np.uint32).tolist()

        msg = CustomMsg()
        msg.header = cloud.header
        msg.header.stamp.sec = int(timebase // 1_000_000_000)
        msg.header.stamp.nanosec = int(timebase % 1_000_000_000)
        msg.timebase = timebase
        msg.point_num = len(valid_indices)
        msg.lidar_id = 1
        msg.points = [
            CustomPoint(
                offset_time=offset_time,
                x=x,
                y=y,
                z=z,
                reflectivity=reflectivity,
                tag=0x10,
                line=line,
            )
            for offset_time, x, y, z, reflectivity, line in zip(
                selected_offsets,
                selected_x,
                selected_y,
                selected_z,
                selected_reflectivity,
                selected_lines,
            )
        ]
        return msg

    def _livox_from_points(self, cloud: PointCloud2, points: list[DecodedLidarPoint]):
        msg = CustomMsg()
        msg.header = cloud.header
        timestamps = [point.timestamp_ns for point in points if point.timestamp_ns is not None]
        msg.timebase = min(timestamps) if timestamps else 0
        # Isaac's PointCloud2 header denotes the completed render product,
        # while this FAST-LIO fork treats CustomMsg.header as the first point
        # time and adds offset_time to obtain cloud_end_time.
        if msg.timebase:
            msg.header.stamp.sec = int(msg.timebase // 1_000_000_000)
            msg.header.stamp.nanosec = int(msg.timebase % 1_000_000_000)
        msg.point_num = len(points)
        msg.lidar_id = 1
        line_count = max(1, int(self.get_parameter("livox_line_count").value))
        for point_data in points:
            point = CustomPoint()
            point.offset_time = int(point_data.timestamp_ns - msg.timebase)
            point.x = point_data.x
            point.y = point_data.y
            point.z = point_data.z
            point.reflectivity = max(0, min(255, int(round(point_data.intensity))))
            point.tag = 0x10
            emitter_id = point_data.emitter_id if point_data.emitter_id is not None else 0
            point.line = int(emitter_id % line_count)
            msg.points.append(point)
        return msg

    def _scan_from_bins(self, cloud, angle_min, angle_max, angle_increment,
                        range_min, range_max, ranges, intensities):
        scan = LaserScan()
        scan.header = cloud.header
        scan.angle_min = angle_min
        scan.angle_max = angle_min + angle_increment * (len(ranges) - 1)
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = float(self.get_parameter("livox_scan_period").value)
        scan.range_min = range_min
        scan.range_max = range_max
        scan.ranges = ranges
        scan.intensities = intensities
        return scan

    def _publish_lidar_outputs(self, future) -> None:
        self._lidar_in_flight = False
        if not rclpy.ok():
            return
        try:
            livox, scan = future.result()
        except Exception as exc:
            self.get_logger().warn(f"LiDAR conversion worker failed: {exc}")
            return
        if livox is not None and self.livox_pub is not None and rclpy.ok():
            self.livox_pub.publish(livox)
        if scan is not None and rclpy.ok():
            for pub in self.scan_pubs:
                if pub.get_subscription_count() > 0:
                    pub.publish(scan)

    def _cloud_to_scan(self, cloud: PointCloud2) -> LaserScan:
        angle_min = float(self.get_parameter("scan_angle_min").value)
        angle_max = float(self.get_parameter("scan_angle_max").value)
        angle_increment = max(1e-6, float(self.get_parameter("scan_angle_increment").value))
        range_min = float(self.get_parameter("scan_range_min").value)
        range_max = float(self.get_parameter("scan_range_max").value)
        height_min = float(self.get_parameter("scan_height_min").value)
        height_max = float(self.get_parameter("scan_height_max").value)
        bin_count = max(1, int(math.ceil((angle_max - angle_min) / angle_increment)))
        ranges = [math.inf] * bin_count
        intensities = [0.0] * bin_count

        max_points = max(1, int(self.get_parameter("livox_max_points").value))
        points = _decode_lidar_points(cloud, require_livox_metadata=False, max_points=max_points)

        for point in points:
            if point.z < height_min or point.z > height_max:
                continue
            distance = math.hypot(point.x, point.y)
            if distance < range_min or distance > range_max:
                continue
            angle = math.atan2(point.y, point.x)
            if angle < angle_min or angle >= angle_max:
                continue
            bin_index = int((angle - angle_min) / angle_increment)
            if 0 <= bin_index < bin_count and distance < ranges[bin_index]:
                ranges[bin_index] = float(distance)
                intensities[bin_index] = point.intensity

        scan = LaserScan()
        scan.header = cloud.header
        scan.angle_min = angle_min
        scan.angle_max = angle_min + angle_increment * (bin_count - 1)
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = float(self.get_parameter("livox_scan_period").value)
        scan.range_min = range_min
        scan.range_max = range_max
        scan.ranges = ranges
        scan.intensities = intensities
        return scan

    def _on_scan(self, scan: LaserScan) -> None:
        for pub in self.scan_pubs:
            pub.publish(scan)

    def _on_odom(self, odom: Odometry) -> None:
        current_stamp_ns = odom.header.stamp.sec * 1_000_000_000 + odom.header.stamp.nanosec
        if current_stamp_ns <= 0:
            current_stamp_ns = time.monotonic_ns()

        current_linear_velocity = (
            float(odom.twist.twist.linear.x),
            float(odom.twist.twist.linear.y),
            float(odom.twist.twist.linear.z),
        )
        current_angular_velocity = (
            float(odom.twist.twist.angular.x),
            float(odom.twist.twist.angular.y),
            float(odom.twist.twist.angular.z),
        )
        if self._latest_odom_stamp_ns is not None:
            delta_ns = current_stamp_ns - self._latest_odom_stamp_ns
            if delta_ns > 0:
                dt_sec = delta_ns / 1_000_000_000.0
                self._latest_imu_linear_acceleration = tuple(
                    (current - previous) / dt_sec
                    for current, previous in zip(current_linear_velocity, self._latest_odom_linear_velocity)
                )
            else:
                self._latest_imu_linear_acceleration = (0.0, 0.0, 0.0)
        self._latest_odom_stamp_ns = current_stamp_ns
        self._latest_odom_linear_velocity = current_linear_velocity
        self._latest_odom_angular_velocity = current_angular_velocity

        for pub in self.odom_pubs:
            pub.publish(odom)

    def _on_joint_states(self, joint_state: JointState) -> None:
        try:
            index = joint_state.name.index("lift_joint")
        except ValueError:
            return
        if index < len(joint_state.position):
            self.latest_lift_position = float(joint_state.position[index])
            if abs(self._lift_jog_velocity) < 1e-9:
                self._lift_target_position = self.latest_lift_position
        if index < len(joint_state.velocity):
            self.latest_lift_velocity = float(joint_state.velocity[index])

    def _publish_lift_target(self, target: float) -> None:
        target = max(-0.650, min(0.300, float(target)))
        self._lift_target_position = target
        msg = Float64MultiArray()
        msg.data = [target]
        self._lift_position_pub.publish(msg)

    def _on_lift_step_command(self, command: Float64MultiArray) -> None:
        if not command.data:
            return
        step = float(command.data[0])
        self._lift_jog_velocity = 0.0
        self._publish_lift_target(self.latest_lift_position + step)

    def _on_lift_jog_command(self, command: Float64) -> None:
        self._lift_jog_velocity = max(-0.10, min(0.10, float(command.data)))
        if abs(self._lift_jog_velocity) < 1e-9:
            self._lift_target_position = self.latest_lift_position

    def _publish_lift_jog_target(self) -> None:
        if abs(self._lift_jog_velocity) < 1e-9:
            return
        self._publish_lift_target(
            self._lift_target_position + self._lift_jog_velocity * 0.1
        )

    def _publish_imu(self) -> None:
        if self.imu_pub is None:
            return
        msg = Imu()
        latest_odom_stamp_ns = getattr(self, "_latest_odom_stamp_ns", None)
        if latest_odom_stamp_ns is not None:
            stamp_ns = latest_odom_stamp_ns
        else:
            now = self.get_clock().now()
            stamp_ns = getattr(now, "nanoseconds", None)
            if stamp_ns is None:
                stamp_ns = int(getattr(now, "sec", 0)) * 1_000_000_000 + int(
                    getattr(now, "nanosec", 0)
                )
        # /odom can be delayed and advance several simulation ticks between
        # wall-clock IMU callbacks while Isaac is below real time.  Limit a
        # single timestamp step to one 100 Hz period so downstream motion
        # compensation never sees a large jump; repeated samples are allowed,
        # but backward jumps are not.
        last_imu_stamp_ns = getattr(self, "_last_imu_stamp_ns", None)
        if last_imu_stamp_ns is not None:
            stamp_ns = max(
                last_imu_stamp_ns,
                min(stamp_ns, last_imu_stamp_ns + 10_000_000),
            )
        self._last_imu_stamp_ns = stamp_ns
        msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        msg.header.frame_id = str(self.get_parameter("livox_frame_id").value)
        gravity_specific_force = (0.0, 0.0, -SIM_LIVOX_GRAVITY_MPS2)
        if latest_odom_stamp_ns is None:
            if not self._odom_imu_warning_emitted:
                self._odom_imu_warning_emitted = True
                self.get_logger().warn(
                    "No /odom has arrived yet; /livox/imu is publishing placeholder orientation until odom is available"
                )
            angular_velocity = (0.0, 0.0, 0.0)
            linear_acceleration = gravity_specific_force
            msg.angular_velocity_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            msg.linear_acceleration_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            angular_velocity = self._latest_odom_angular_velocity
            linear_acceleration = tuple(
                acceleration + gravity
                for acceleration, gravity in zip(self._latest_imu_linear_acceleration, gravity_specific_force)
            )
            msg.angular_velocity_covariance = [0.05, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.05]
            msg.linear_acceleration_covariance = [0.10, 0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.10]
        msg.orientation.w = 1.0
        msg.orientation_covariance = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        msg.angular_velocity.x = angular_velocity[0]
        msg.angular_velocity.y = angular_velocity[1]
        msg.angular_velocity.z = angular_velocity[2]
        msg.linear_acceleration.x = linear_acceleration[0]
        msg.linear_acceleration.y = linear_acceleration[1]
        msg.linear_acceleration.z = linear_acceleration[2]
        self.imu_pub.publish(msg)

    def _publish_battery_state(self) -> None:
        if self.battery_state_pub is None:
            return

        percentage = float(self.get_parameter("battery_percentage").value)
        msg = BatteryState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.percentage = max(0.0, min(1.0, percentage))
        msg.voltage = float(self.get_parameter("battery_voltage").value)
        msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        msg.power_supply_health = BatteryState.POWER_SUPPLY_HEALTH_GOOD
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
        msg.present = True
        msg.location = "sim"
        self.battery_state_pub.publish(msg)

    def _publish_lift_status(self) -> None:
        homing = String()
        homing.data = "COMPLETED:simulated"
        self.homing_state_pub.publish(homing)

        limits = String()
        limits.data = (
            "raw=0x00000000,upper=0,home=0,lower=0,b0_NOT=0,b1_POT=0,b2_HOME=0,"
            "b7_SI4=0,b8_SI5=0,b9_SI6=0,b19_DI4=0,b20_DI5=0,b21_DI6=0,"
            "b27_DI4=0,b28_DI5=0,b29_DI6=0"
        )
        self.limit_switch_state_pub.publish(limits)

        if self.motor_status_pub is None:
            return
        msg = MotorStatus()
        msg.stamp = self.get_clock().now().to_msg()
        msg.position_m = self.latest_lift_position
        msg.velocity_mps = self.latest_lift_velocity
        msg.physical_position_m = self.latest_lift_position
        msg.statusword = 0x1237
        msg.cia402_state = "OPERATION_ENABLED"
        msg.mode_of_operation = "position"
        msg.is_enabled = True
        msg.is_fault = False
        msg.is_target_reached = True
        msg.is_homing = False
        msg.digital_inputs_raw = 0
        msg.upper_limit_switch = False
        msg.home_switch = False
        msg.lower_limit_switch = False
        msg.limit_switch_valid = True
        msg.homing_complete = True
        msg.homing_state = "COMPLETED"
        msg.profile_speed_mps = 0.05
        msg.profile_accel_mps2 = 0.2
        msg.profile_decel_mps2 = 0.2
        self.motor_status_pub.publish(msg)

    def _create_lift_services(self) -> None:
        for name in (
            "/lift_slide_driver/start_homing",
            "/lift_slide_driver/return_home",
            "/lift_slide_driver/enable",
            "/lift_slide_driver/quick_stop",
            "/lift_slide_driver/hold_position",
        ):
            self.create_service(Trigger, name, self._trigger_ok)

    def _trigger_ok(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = "simulated"
        return response

    def _create_mapping_stub_services(self) -> None:
        if SaveMaps is not None:
            self.create_service(SaveMaps, "/pgo/save_maps", self._mapping_service_ok)
        if RefineMap is not None:
            self.create_service(RefineMap, "/hba_node/refine_map", self._mapping_service_ok)
        if SavePoses is not None:
            self.create_service(SavePoses, "/hba_node/save_poses", self._mapping_service_ok)

    def _mapping_service_ok(self, _request: object, response: object) -> object:
        response.success = True
        response.message = "simulated service stub; no map file was written"
        return response

    def destroy_node(self) -> bool:
        self._sensor_workers.shutdown(wait=True, cancel_futures=True)
        self._lidar_worker.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OpenFlexSimCompatBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        # ROS 2 Humble can surface this pybind conversion error when SIGINT
        # invalidates the context while a large PointCloud2 is being taken.
        if "Unable to convert call argument to Python object" not in str(exc):
            raise
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        except RuntimeError as exc:
            if "context is invalid" not in str(exc):
                raise
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
