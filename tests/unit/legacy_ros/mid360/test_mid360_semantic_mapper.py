#!/usr/bin/env python3
"""Pure-function and configuration tests for the semantic mapper."""

from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path
import re
import struct
import sys
import unittest

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud2, PointField


REPO_DIR = Path(__file__).resolve().parents[4]
SCRIPT = REPO_DIR / "sim_runtime" / "components" / "sensors" / "mid360" / "mid360_semantic_mapper.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mid360_semantic_mapper_test_module", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_cloud(object_ids):
    cloud = PointCloud2()
    cloud.height = 1
    cloud.width = len(object_ids)
    cloud.point_step = 28
    cloud.row_step = cloud.point_step * cloud.width
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="object_id", offset=12, datatype=PointField.UINT32, count=4),
    ]
    data = bytearray(cloud.row_step)
    for index, object_id in enumerate(object_ids):
        offset = index * cloud.point_step
        words = [(object_id >> (32 * word)) & 0xFFFFFFFF for word in range(4)]
        struct.pack_into("<fffIIII", data, offset, float(index), 0.0, 1.0, *words)
    cloud.data = bytes(data)
    return cloud


class SemanticMapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_full_128_bit_object_ids_are_preserved(self):
        ids = [(1 << 96) + (7 << 64) + (3 << 32) + 9, 42]
        decoded = self.module.decode_object_ids(make_cloud(ids))
        self.assertEqual(decoded.tolist(), ids)

    def test_transform_points(self):
        transform = TransformStamped()
        transform.transform.translation.x = 2.0
        transform.transform.rotation.w = 1.0
        transformed = self.module.transform_points(np.array([[1.0, 2.0, 3.0]]), transform)
        np.testing.assert_allclose(transformed, [[3.0, 2.0, 3.0]])

    def test_pack_cloud_has_class_and_rgb_fields(self):
        message = self.module.pack_cloud(
            np.array([[1.0, 2.0, 3.0]]),
            np.array([2], dtype=np.uint16),
            np.array([[220, 70, 50]], dtype=np.uint8),
            "map",
            None,
        )
        self.assertEqual(message.header.frame_id, "map")
        self.assertEqual(message.width, 1)
        self.assertEqual([field.name for field in message.fields], ["x", "y", "z", "class_id", "rgb"])
        self.assertEqual(len(message.data), 20)

    def test_standalone_config_contains_labels_and_rules(self):
        config = (REPO_DIR / "sim_runtime" / "config" / "sensors" / "mid360" / "mid360_semantic_mapper.yaml").read_text(encoding="utf-8")
        stage = (REPO_DIR / "sim_runtime" / "assets" / "environments" / "mid360_empty_stage.usda").read_text(encoding="utf-8")
        self.assertIn("TargetBox[0-9]+", config)
        self.assertIn("SemanticsLabelsAPI:class", stage)
        self.assertIn('semantics:labels:class = ["floor"]', stage)

    def test_late_object_map_reclassifies_existing_voxels(self):
        mapper = object.__new__(self.module.Mid360SemanticMapper)
        mapper.classes = {
            "unknown": self.module.SemanticClass("unknown", 0, (128, 128, 128)),
            "obstacle": self.module.SemanticClass("obstacle", 2, (220, 70, 50)),
        }
        mapper.rules = [(re.compile(r"/World/Box[^/]*(?:/|$)"), "obstacle")]
        mapper.object_id_to_path = {}
        mapper.class_point_counts = Counter()
        mapper.unresolved_ids = Counter()
        mapper.unknown_points = 0
        mapper.voxels = {
            (0, 0, 0): self.module.Voxel(
                np.zeros(3), 1, Counter({0: 1}), Counter({123: 1})
            )
        }

        mapper._reclassify_voxels()
        self.assertEqual(mapper.voxels[(0, 0, 0)].class_votes, Counter({0: 1}))

        mapper.object_id_to_path = {123: "/World/Box01"}
        mapper._reclassify_voxels()
        self.assertEqual(mapper.voxels[(0, 0, 0)].class_votes, Counter({2: 1}))
        self.assertEqual(mapper.unknown_points, 0)


if __name__ == "__main__":
    unittest.main()
