#!/usr/bin/env python3
"""Accumulate an RTX LiDAR PointCloud2 into a small semantic voxel map.

The node deliberately consumes the raw Isaac PointCloud2 in parallel with the
CustomMsg/FAST-LIO bridge.  Isaac's object-id map is a ``std_msgs/String``
containing JSON and maps full 128-bit stable IDs to USD prim paths.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_ros
import yaml


_DTYPES = {
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
class SemanticClass:
    name: str
    class_id: int
    color: tuple[int, int, int]


@dataclass
class Voxel:
    xyz_sum: np.ndarray
    count: int
    class_votes: Counter[int]
    object_id_votes: Counter[int]


def _field_dtype(field: PointField, bigendian: bool) -> np.dtype:
    try:
        dtype = _DTYPES[int(field.datatype)]
    except KeyError as exc:
        raise ValueError(f"unsupported PointCloud2 datatype {field.datatype}") from exc
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if bigendian else "<")
    return dtype


def read_field(cloud: PointCloud2, field: PointField) -> np.ndarray:
    """Read a scalar or fixed-size array PointCloud2 field with row padding."""
    height, width = int(cloud.height), int(cloud.width)
    count = int(field.count)
    if height <= 0 or width <= 0 or count <= 0:
        return np.empty((0, count), dtype=_field_dtype(field, bool(cloud.is_bigendian)))
    point_step = int(cloud.point_step)
    row_step = int(cloud.row_step) or point_step * width
    dtype = _field_dtype(field, bool(cloud.is_bigendian))
    field_bytes = dtype.itemsize * count
    if point_step <= 0 or row_step < point_step * width:
        raise ValueError("invalid PointCloud2 point/row stride")
    if int(field.offset) < 0 or int(field.offset) + field_bytes > point_step:
        raise ValueError(f"field '{field.name}' exceeds point_step")
    data = bytes(cloud.data)
    end = (height - 1) * row_step + (width - 1) * point_step + int(field.offset) + field_bytes
    if end > len(data):
        raise ValueError(f"PointCloud2 data is truncated for field '{field.name}'")
    array = np.ndarray(
        shape=(height, width, count),
        dtype=dtype,
        buffer=data,
        offset=int(field.offset),
        strides=(row_step, point_step, dtype.itemsize),
    )
    return array.reshape(-1, count)


def decode_object_ids(cloud: PointCloud2) -> np.ndarray:
    """Return full 128-bit IDs as Python integers, preserving all four words."""
    fields = {field.name: field for field in cloud.fields}
    field = fields.get("object_id")
    if field is None or int(field.datatype) != PointField.UINT32 or int(field.count) != 4:
        raise ValueError("PointCloud2 must contain object_id as UINT32[4]")
    words = read_field(cloud, field).astype(np.uint64, copy=False)
    return np.asarray(
        [sum(int(word[index]) << (32 * index) for index in range(4)) for word in words],
        dtype=object,
    )


def decode_xyz(cloud: PointCloud2) -> np.ndarray:
    fields = {field.name: field for field in cloud.fields}
    missing = sorted({"x", "y", "z"} - fields.keys())
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {', '.join(missing)}")
    return np.column_stack(
        [read_field(cloud, fields[name])[:, 0] for name in ("x", "y", "z")]
    ).astype(np.float64, copy=False)


def transform_points(points: np.ndarray, transform: TransformStamped) -> np.ndarray:
    """Apply a geometry_msgs transform without depending on tf_transformations."""
    q = transform.transform.rotation
    tx, ty, tz = transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= 1.0e-12:
        raise ValueError("TF quaternion has zero norm")
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return points @ rotation.T + np.array([tx, ty, tz], dtype=np.float64)


def pack_cloud(points: np.ndarray, classes: np.ndarray, colors: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Create a compact XYZ/class/rgb PointCloud2 without Python point loops."""
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="class_id", offset=12, datatype=PointField.UINT16, count=1),
        PointField(name="rgb", offset=16, datatype=PointField.UINT32, count=1),
    ]
    point_step = 20
    payload = bytearray(point_step * len(points))
    xyz = np.asarray(points, dtype=np.float32)
    rgb = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | colors[:, 2].astype(np.uint32)
    )
    structured = np.zeros(len(points), dtype=np.dtype([("xyz", "<f4", (3,)), ("class_id", "<u2"), ("pad", "<u2"), ("rgb", "<u4")]))
    structured["xyz"] = xyz
    structured["class_id"] = classes.astype(np.uint16)
    structured["rgb"] = rgb
    payload[:] = structured.tobytes()
    message = PointCloud2()
    if stamp is not None:
        message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.height = 1
    message.width = len(points)
    message.fields = fields
    message.is_bigendian = False
    message.point_step = point_step
    message.row_step = point_step * len(points)
    message.is_dense = True
    message.data = bytes(payload)
    return message


class Mid360SemanticMapper(Node):
    def __init__(self) -> None:
        super().__init__("mid360_semantic_mapper")
        self.declare_parameter("config_path", "")
        config_path = str(self.get_parameter("config_path").value).strip()
        self.config = self._load_config(config_path)
        self._configure_from_config()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.map_pub = self.create_publisher(PointCloud2, self.semantic_topic, 1)
        self.report_pub = self.create_publisher(String, self.report_topic, 1)
        self.object_map_sub = self.create_subscription(String, self.object_map_topic, self._on_object_map, 10)
        self.cloud_sub = self.create_subscription(PointCloud2, self.pointcloud_topic, self._on_cloud, 1)
        self.save_service = self.create_service(Trigger, self.save_service_name, self._on_save)
        self.voxels: dict[tuple[int, int, int], Voxel] = {}
        self.object_id_to_path: dict[int, str] = {}
        self.frames = 0
        self.raw_points = 0
        self.finite_points = 0
        self.mapped_points = 0
        self.unknown_points = 0
        self.unresolved_ids: Counter[str] = Counter()
        self.class_point_counts: Counter[str] = Counter()
        self._last_stamp = None
        self.get_logger().info(
            f"Semantic mapper ready: cloud={self.pointcloud_topic}, object_map={self.object_map_topic}, "
            f"map_frame={self.map_frame}, voxel={self.voxel_size:g} m"
        )

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        if not config_path:
            return {}
        with Path(config_path).expanduser().open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("semantic mapper config must be a YAML mapping")
        return data

    def _configure_from_config(self) -> None:
        input_cfg = self.config.get("input", {})
        frame_cfg = self.config.get("frames", {})
        mapping_cfg = self.config.get("mapping", {})
        output_cfg = self.config.get("output", {})
        self.pointcloud_topic = str(input_cfg.get("pointcloud_topic", "/openflex/livox_frame/lidar"))
        self.object_map_topic = str(input_cfg.get("object_id_map_topic", self.pointcloud_topic.rstrip("/") + "/object_id_map"))
        self.map_frame = str(frame_cfg.get("map_frame", "livox_frame"))
        self.transform_timeout = float(frame_cfg.get("transform_timeout_sec", 0.05))
        self.allow_identity = bool(frame_cfg.get("allow_identity_when_same_frame", True))
        self.voxel_size = float(mapping_cfg.get("voxel_size", 0.10))
        self.min_range = float(mapping_cfg.get("min_range", 0.05))
        self.max_range = float(mapping_cfg.get("max_range", 70.0))
        self.max_voxels = int(mapping_cfg.get("max_voxels", 1_000_000))
        if self.voxel_size <= 0.0:
            raise ValueError("mapping.voxel_size must be positive")
        if self.min_range < 0.0 or self.max_range <= self.min_range:
            raise ValueError("mapping range must satisfy 0 <= min_range < max_range")
        if self.max_voxels <= 0:
            raise ValueError("mapping.max_voxels must be positive")
        self.semantic_topic = str(output_cfg.get("semantic_pointcloud_topic", "/openflex/semantic_map"))
        self.report_topic = str(output_cfg.get("report_topic", "/openflex/semantic_map/report"))
        self.save_service_name = str(output_cfg.get("save_service", "/openflex/semantic_map/save"))
        self.output_dir = Path(str(output_cfg.get("directory", "~/openflex_semantic_map"))).expanduser()
        self.classes = self._parse_classes(self.config.get("classes", {}))
        self.rules = []
        for item in self.config.get("rules", []):
            if not isinstance(item, dict) or "prim_path_regex" not in item:
                continue
            class_name = str(item.get("class", "unknown"))
            self.rules.append((re.compile(str(item["prim_path_regex"])), class_name))

    @staticmethod
    def _parse_classes(data: dict[str, Any]) -> dict[str, SemanticClass]:
        result: dict[str, SemanticClass] = {}
        for name, raw in data.items():
            if not isinstance(raw, dict):
                continue
            color = tuple(max(0, min(255, int(v))) for v in raw.get("color", [128, 128, 128]))
            if len(color) != 3:
                raise ValueError(f"class '{name}' color must have three values")
            result[str(name)] = SemanticClass(str(name), int(raw.get("id", 0)), color)
        if "unknown" not in result:
            result["unknown"] = SemanticClass("unknown", 0, (128, 128, 128))
        return result

    def _on_object_map(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            mapping = payload.get("id_to_labels", payload)
            if not isinstance(mapping, dict):
                raise ValueError("id_to_labels is not an object")
            parsed = {}
            for key, value in mapping.items():
                if isinstance(value, dict):
                    value = value.get("prim_path", value.get("path", value.get("class", "")))
                parsed[int(key)] = str(value)
            self.object_id_to_path = parsed
            # The helper may publish the first object table after the first
            # cloud. Rebuild existing voxel labels so startup ordering cannot
            # permanently turn an otherwise valid map into "unknown".
            self._reclassify_voxels()
            if self.voxels:
                self._publish_map(self._last_stamp)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring invalid Object ID map: {exc}")

    def _lookup_transform(self, cloud: PointCloud2) -> TransformStamped | None:
        source = cloud.header.frame_id.lstrip("/")
        target = self.map_frame.lstrip("/")
        if source == target and self.allow_identity:
            transform = TransformStamped()
            transform.transform.rotation.w = 1.0
            return transform
        try:
            return self.tf_buffer.lookup_transform(
                self.map_frame,
                cloud.header.frame_id,
                Time.from_msg(cloud.header.stamp),
                timeout=Duration(seconds=self.transform_timeout),
            )
        except Exception as exc:
            self.get_logger().warning(f"No TF {self.map_frame} <- {cloud.header.frame_id}: {exc}")
            return None

    def _class_for(self, object_id: int, *, record_unresolved: bool = True) -> SemanticClass:
        path = self.object_id_to_path.get(object_id)
        if path is None:
            if record_unresolved:
                self.unresolved_ids[str(object_id)] += 1
            return self.classes["unknown"]
        for pattern, class_name in self.rules:
            if pattern.search(path):
                return self.classes.get(class_name, self.classes["unknown"])
        return self.classes["unknown"]

    def _on_cloud(self, cloud: PointCloud2) -> None:
        try:
            points = decode_xyz(cloud)
            object_ids = decode_object_ids(cloud)
        except ValueError as exc:
            self.get_logger().error(f"Semantic input rejected: {exc}")
            return
        self.frames += 1
        self._last_stamp = cloud.header.stamp
        self.raw_points += len(points)
        finite = np.isfinite(points).all(axis=1)
        ranges = np.linalg.norm(points, axis=1)
        finite &= (ranges >= self.min_range) & (ranges <= self.max_range)
        self.finite_points += int(finite.sum())
        if not finite.any():
            return
        transform = self._lookup_transform(cloud)
        if transform is None:
            return
        selected = np.flatnonzero(finite)
        points = transform_points(points[selected], transform)
        ids = object_ids[selected]
        classes = [self._class_for(int(object_id)) for object_id in ids]
        self.mapped_points += len(points)
        self.unknown_points += sum(item.name == "unknown" for item in classes)
        self.class_point_counts.update(item.name for item in classes)
        keys = np.floor(points / self.voxel_size).astype(np.int64)
        for point, key, semantic_class, object_id in zip(points, keys, classes, ids):
            key_tuple = tuple(int(value) for value in key)
            voxel = self.voxels.get(key_tuple)
            if voxel is None:
                if len(self.voxels) >= self.max_voxels:
                    continue
                voxel = Voxel(np.zeros(3, dtype=np.float64), 0, Counter(), Counter())
                self.voxels[key_tuple] = voxel
            voxel.xyz_sum += point
            voxel.count += 1
            voxel.object_id_votes[int(object_id)] += 1
            voxel.class_votes[semantic_class.class_id] += 1
        self._publish_map(cloud.header.stamp)

    def _reclassify_voxels(self) -> None:
        """Recompute voxel class votes after an Object-ID map update."""
        self.class_point_counts.clear()
        self.unresolved_ids.clear()
        self.unknown_points = 0
        by_id = {semantic.class_id: semantic for semantic in self.classes.values()}
        for voxel in self.voxels.values():
            voxel.class_votes.clear()
            for object_id, count in voxel.object_id_votes.items():
                if object_id not in self.object_id_to_path:
                    self.unresolved_ids[str(object_id)] += count
                semantic_class = self._class_for(object_id, record_unresolved=False)
                voxel.class_votes[semantic_class.class_id] += count
            for class_id, count in voxel.class_votes.items():
                class_name = by_id.get(class_id, self.classes["unknown"]).name
                self.class_point_counts[class_name] += count
                if class_name == "unknown":
                    self.unknown_points += count

    def _publish_map(self, stamp) -> None:
        if not self.voxels:
            return
        keys = list(self.voxels)
        points = np.asarray([self.voxels[key].xyz_sum / self.voxels[key].count for key in keys])
        class_ids = np.asarray([max(self.voxels[key].class_votes, key=self.voxels[key].class_votes.get) for key in keys], dtype=np.uint16)
        by_id = {semantic.class_id: semantic for semantic in self.classes.values()}
        colors = np.asarray([by_id.get(int(class_id), self.classes["unknown"]).color for class_id in class_ids], dtype=np.uint8)
        self.map_pub.publish(pack_cloud(points, class_ids, colors, self.map_frame, stamp))
        report = String()
        report.data = json.dumps(self._report(), sort_keys=True)
        self.report_pub.publish(report)

    def _report(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "raw_points": self.raw_points,
            "finite_points": self.finite_points,
            "mapped_points": self.mapped_points,
            "unknown_points": self.unknown_points,
            "labeled_coverage": self.mapped_points and 1.0 - self.unknown_points / self.mapped_points or 0.0,
            "voxel_count": len(self.voxels),
            "unresolved_object_ids": len(self.unresolved_ids),
            "class_point_counts": dict(self.class_point_counts),
            "map_frame": self.map_frame,
            "voxel_size": self.voxel_size,
        }

    def _on_save(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._write_ply(self.output_dir / "semantic_map.ply")
            (self.output_dir / "legend.json").write_text(
                json.dumps({name: {"id": item.class_id, "color": item.color} for name, item in self.classes.items()}, indent=2),
                encoding="utf-8",
            )
            (self.output_dir / "report.json").write_text(json.dumps(self._report(), indent=2), encoding="utf-8")
            response.success = True
            response.message = str(self.output_dir)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _write_ply(self, path: Path) -> None:
        by_id = {semantic.class_id: semantic for semantic in self.classes.values()}
        with path.open("w", encoding="ascii") as stream:
            stream.write("ply\nformat ascii 1.0\n")
            stream.write(f"element vertex {len(self.voxels)}\n")
            stream.write("property float x\nproperty float y\nproperty float z\n")
            stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty ushort class_id\nend_header\n")
            for voxel in self.voxels.values():
                point = voxel.xyz_sum / voxel.count
                class_id = max(voxel.class_votes, key=voxel.class_votes.get)
                color = by_id.get(class_id, self.classes["unknown"]).color
                stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]} {class_id}\n")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = Mid360SemanticMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
