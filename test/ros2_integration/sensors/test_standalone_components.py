#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = (
    REPO_ROOT
    / "ros2_pkgs"
    / "openflex_isaac_sim"
    / "openflex_isaac_sensors"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from openflex_isaac_sensors.diagnostics import SensorDiagnostics
from openflex_isaac_sensors.frame_packet import FramePacket
from openflex_isaac_sensors.integration import (
    load_mid360_config,
    load_realsense_config,
    resolve_robot_mount_path,
)
from openflex_isaac_sensors.mid360 import (
    create_standalone_mid360,
    robot_lidar_graph_path,
    robot_sensor_root_path,
)
from openflex_isaac_sensors.mount import LocalPose, resolve_mount_prim_path
from openflex_isaac_sensors.sinks import AsyncJsonlSink
from openflex_isaac_sensors.transport import BoundedFrameQueue


class StandaloneComponentsTest(unittest.TestCase):
    def test_standalone_mid360_factory_is_exported(self) -> None:
        self.assertTrue(callable(create_standalone_mid360))

    def test_mid360_helper_keeps_playback_graph_with_runtime_gate(self) -> None:
        source = (PACKAGE_ROOT / "openflex_isaac_sensors" / "mid360.py").read_text(encoding="utf-8")
        direct_source = source.split("\ndef _create_graph_owned_lidar_graph", 1)[0]
        self.assertIn(
            '("OnPlaybackTick", "omni.graph.action.OnPlaybackTick")',
            direct_source,
        )
        self.assertIn(
            '("OnPlaybackTick.outputs:tick", "LidarEnableGate.inputs:execIn")',
            direct_source,
        )
        self.assertIn(
            '("LidarEnableGate.outputs:execOut", "PointCloudPublish.inputs:execIn")',
            direct_source,
        )
        self.assertIn('("LidarEnableGate.inputs:step", 0)', direct_source)
        self.assertNotIn(
            '("SimulationFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame")',
            direct_source,
        )

    def test_standalone_default_transport_is_historical_direct_path(self) -> None:
        source = (PACKAGE_ROOT / "openflex_isaac_sensors" / "mid360.py").read_text(encoding="utf-8")
        self.assertIn('transport: str = "direct"', source)

    def test_mid360_performance_audit_uses_sensor_data_qos(self) -> None:
        source = (
            REPO_ROOT / "test" / "performance" / "mid360_standalone_performance.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from rclpy.qos import qos_profile_sensor_data", source)
        self.assertIn("RAW_TOPIC, self._raw, qos_profile_sensor_data", source)
        self.assertNotIn("reliability=ReliabilityPolicy.RELIABLE", source)

    def test_robot_lidar_graph_is_outside_referenced_robot_tree(self) -> None:
        graph_path = robot_lidar_graph_path("/World/OpenFlex")
        self.assertEqual(graph_path, "/World/OpenFlex_Sensors/MID360/Lidar_ROS2_Graph")
        self.assertFalse(graph_path.startswith("/World/OpenFlex/"))

    def test_robot_mid360_sensor_root_is_outside_referenced_robot_tree(self) -> None:
        root_path = robot_sensor_root_path("/World/OpenFlex")
        self.assertEqual(root_path, "/World/OpenFlex_Sensors/MID360")
        self.assertFalse(root_path.startswith("/World/OpenFlex/"))

    def test_robot_mount_path_tracks_loaded_robot_root(self) -> None:
        self.assertEqual(
            resolve_robot_mount_path(
                "/World/OpenFlex/Geometry/base_link/CameraMount",
                "/openarmx_integrated",
            ),
            "/openarmx_integrated/Geometry/base_link/CameraMount",
        )

    def test_branch_specific_config_loaders_do_not_require_the_other_sensor(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "realsense" / "config").mkdir(parents=True)
            (root / "realsense" / "config" / "realsense_robot_mounts.yaml").write_text(
                "schema_version: 1" + chr(10) + "cameras: {}" + chr(10), encoding="utf-8"
            )
            self.assertEqual(load_realsense_config(root)["schema_version"], 1)
            self.assertRaises(FileNotFoundError, load_mid360_config, root)

    def test_transition_tree_loads_both_released_configs(self) -> None:
        asset_root = REPO_ROOT
        self.assertEqual(load_realsense_config(asset_root)["schema_version"], 1)
        lidar = load_mid360_config(asset_root)
        self.assertEqual(lidar["schema_version"], 1)
        fallback = LocalPose.from_mapping(lidar["sensor"]["fallback_mount_pose"])
        self.assertEqual(fallback.translation_m, (0.3, 0.0, 0.12))

    def test_mount_contract_does_not_require_robot_names(self) -> None:
        self.assertEqual(
            resolve_mount_prim_path("/World/StandaloneSensors", "d435_standalone"),
            "/World/StandaloneSensors/d435_standalone_Mount",
        )
        self.assertEqual(
            resolve_mount_prim_path("/World/Robot/link", "d405", "/World/Robot/link/d405_mount"),
            "/World/Robot/link/d405_mount",
        )

    def test_pose_normalizes_quaternion(self) -> None:
        pose = LocalPose.from_mapping({"translation_m": [1, 2, 3], "quaternion_wxyz": [2, 0, 0, 0]})
        self.assertEqual(pose.translation_m, (1.0, 2.0, 3.0))
        self.assertEqual(pose.quaternion_wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_queue_drops_oldest_and_keeps_latest(self) -> None:
        queue = BoundedFrameQueue[int](2)
        for item in (1, 2, 3):
            queue.put(item)
        self.assertEqual(queue.drain(), [2, 3])
        self.assertEqual(queue.stats()["drop_count"], 1)

    def test_packet_owns_numpy_payload(self) -> None:
        import numpy as np

        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        packet = FramePacket.owned(
            episode_id=0,
            snapshot_id=None,
            camera_name="d405",
            frame_id=0,
            sample_sim_time_ns=1,
            capture_wall_time_ns=2,
            calibration_id="test",
            rgb=rgb,
            depth_m=np.ones((2, 2, 1), dtype=np.float32),
            depth_semantics="z_depth_ideal_aligned",
            source_state_seq=0,
            rgb_encoding="rgb8",
            depth_encoding="32FC1_m",
        )
        rgb[0, 0, 0] = 255
        self.assertEqual(int(packet.rgb[0, 0, 0]), 0)
        packet.validate()

    def test_async_sink_drains_owned_packet(self) -> None:
        import tempfile
        import time

        packet = FramePacket.owned(
            episode_id=0,
            snapshot_id=None,
            camera_name="d405",
            frame_id=0,
            sample_sim_time_ns=1,
            capture_wall_time_ns=2,
            calibration_id="test",
            rgb=None,
            depth_m=None,
            depth_semantics="z_depth_ideal_aligned",
            source_state_seq=0,
            rgb_encoding="rgb8",
            depth_encoding="32FC1_m",
        )
        with tempfile.TemporaryDirectory() as directory:
            sink = AsyncJsonlSink(Path(directory) / "frames.jsonl")
            sink.write(packet)
            sink.close()
            self.assertEqual(sink.stats()["put_count"], 1)
            self.assertEqual(len((Path(directory) / "frames.jsonl").read_text().splitlines()), 1)

    def test_diagnostics_reports_unique_frames_and_rate(self) -> None:
        diag = SensorDiagnostics("d435")
        for index in range(4):
            diag.record_physics(index / 90.0, 1_000_000_000 + index * 11_111_111)
        for index in range(3):
            diag.record_frame(index, index / 30.0, 2_000_000_000 + index * 33_333_333)
        summary = diag.summary()
        self.assertEqual(summary["frames"], 3)
        self.assertTrue(summary["unique_frame_ids"])
        self.assertAlmostEqual(summary["frame_sim_hz"], 30.0)


if __name__ == "__main__":
    unittest.main()
