from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call


_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "arena_benchmark"
    / "src"
    / "mrs_arena"
    / "sensors"
    / "mid360_lifecycle.py"
)
_SPEC = importlib.util.spec_from_file_location("mid360_lifecycle", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load MID360 lifecycle module: {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class Mid360MountResolutionTest(unittest.TestCase):
    def test_kit_args_enable_the_classic_rtx_sensor_render_settings(self) -> None:
        configured = _MODULE.configure_sensor_kit_args(
            "--verbose --/renderer/raytracingMotion/enabled=false --custom=value"
        )

        self.assertEqual(configured.count("--/renderer/raytracingMotion/enabled=true"), 1)
        self.assertIn("--/rtx/hydra/supportMultiTickRate=true", configured)
        self.assertIn("--/renderer/multiGpu/enabled=false", configured)
        self.assertIn("--/exts/omni.replicator.srtx/enabled=false", configured)
        self.assertIn("--verbose", configured)
        self.assertIn("--custom=value", configured)
        self.assertNotIn("--/renderer/raytracingMotion/enabled=false", configured)

    def test_maps_canonical_mount_suffix_onto_arena_robot_prim(self) -> None:
        self.assertEqual(
            _MODULE.resolve_robot_mount_path(
                "/World/envs/env_0/Robot",
                "/World/OpenFlex/Geometry/base_link/Mid360Mount",
            ),
            "/World/envs/env_0/Robot/Geometry/base_link/Mid360Mount",
        )

    def test_does_not_duplicate_base_link_when_given_the_articulation_root_path(self) -> None:
        self.assertEqual(
            _MODULE.resolve_robot_mount_path(
                "/World/envs/env_0/Robot/Geometry/base_link",
                "/World/OpenFlex/Geometry/base_link/Mid360Mount",
            ),
            "/World/envs/env_0/Robot/Geometry/base_link/Mid360Mount",
        )

    def test_rejects_mount_config_without_robot_geometry_anchor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Geometry/base_link/Mid360Mount"):
            _MODULE.resolve_robot_mount_path(
                "/World/envs/env_0/Robot", "/World/OpenFlex/Invalid/Mount"
            )

    def test_relay_source_uses_the_classic_mid360_pointcloud_topic(self) -> None:
        self.assertEqual(
            _MODULE.lidar_pointcloud_topic(
                {
                    "pointcloud_topic": "/openflex/livox_frame/lidar",
                    "output_topic": "/livox/lidar_points",
                }
            ),
            "/openflex/livox_frame/lidar",
        )

    def test_extension_manager_falls_back_to_the_kit_app_singleton(self) -> None:
        expected_manager = object()
        kit_app = SimpleNamespace(get_extension_manager=lambda: expected_manager)
        wrapper_app = SimpleNamespace()

        manager = _MODULE._resolve_extension_manager(
            wrapper_app, kit_app_getter=lambda: kit_app
        )

        self.assertIs(manager, expected_manager)


class Mid360StartupPreparationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = object.__new__(_MODULE.ArenaMid360Runtime)
        self.stage = object()
        self.root_path = "/World/envs/env_0/Robot/Geometry/base_link/Mid360_Runtime"
        self.graph_path = self.root_path + "/Lidar_ROS2_Graph"
        self.imu_graph_path = self.root_path + "/IMU_ROS2_Graph"
        self.lidar_handle = object()
        self.lidar_sensor = object()
        self.render_product = object()
        self.relay = SimpleNamespace(
            is_ready=Mock(return_value=True),
            has_data=Mock(return_value=True),
            is_healthy=Mock(return_value=True),
            stop=Mock(),
        )
        self.lidar_create = Mock(side_effect=self._record_lidar_create)
        self.imu_create = Mock()
        self.mid360 = SimpleNamespace(
            _LIDAR_SENSOR_HANDLES=[],
            _LIDAR_RENDER_PRODUCT_HANDLES=[],
            create_robot_mid360=self.lidar_create,
            create_physics_imu_graph=self.imu_create,
            release_physics_imu_sensor=Mock(),
        )
        self.runtime._app = SimpleNamespace(update=Mock())
        self.runtime._robot_prim_path = "/World/envs/env_0/Robot/Geometry/base_link"
        self.runtime._repository_root = Path("/repository")
        self.runtime._sensor_config = {
            "mount_prim_path": "/World/OpenFlex/Geometry/base_link/Mid360Mount",
            "pointcloud_topic": "/openflex/livox_frame/lidar",
            "livox_topic": "/livox/lidar",
            "imu_topic": "/livox/imu",
            "frame_id": "livox_frame",
            "default_profile": "MID360_APPROX",
            "performance_profile": "MID360_PERFORMANCE",
            "tick_rate_hz": 10.0,
        }
        self.runtime._extension_manager = SimpleNamespace(
            is_extension_enabled=Mock(return_value=True)
        )
        self.runtime._bridge_ready_updates = 0
        self.runtime._mid360_module = self.mid360
        self.runtime._root_path = ""
        self.runtime._mount_path = ""
        self.runtime._root_created = False
        self.runtime._active_sensors = set()
        self.runtime._prepared_lidar = None
        self.runtime._prepared_imu = None
        self.runtime._lidar_relay = None
        self.runtime._preparation_error = ""
        self.runtime._stage = lambda: self.stage
        self.runtime._ensure_sensor_root = lambda stage: self.root_path
        self.runtime._set_gate = Mock()
        self.runtime._start_livox_relay = Mock(return_value=self.relay)
        self.runtime._cleanup_lidar = Mock()
        self.runtime._cleanup_imu = Mock()
        self.runtime._release_sensor_root = Mock()

    def _record_lidar_create(self, *args, **kwargs) -> None:
        kwargs["resource_sink"].update(
            {
                "stage": self.stage,
                "root_path": self.root_path,
                "graph_path": self.graph_path,
                "lidar_path": self.root_path + "/Lidar/LiDAR",
                "lidar": self.lidar_handle,
                "lidar_sensor": self.lidar_sensor,
                "render_product": self.render_product,
                "render_product_path": "/Render/Products/MID360",
            }
        )

    def test_api_lidar_lifecycle_only_gates_the_startup_prepared_sensor(self) -> None:
        self.runtime.prepare_sensors()
        self.runtime.prepare_sensors()
        self.runtime._start_livox_relay.assert_not_called()

        resource = self.runtime.create_lidar()
        self.assertTrue(resource.ready())
        resource.destroy()

        self.assertEqual(self.lidar_create.call_count, 1)
        self.imu_create.assert_called_once()
        self.runtime._start_livox_relay.assert_called_once_with(
            "/openflex/livox_frame/lidar", "livox_frame"
        )
        self.assertEqual(
            self.runtime._set_gate.call_args_list,
            [
                call(self.graph_path + "/LidarEnableGate.inputs:step", False),
                call(self.imu_graph_path + "/IMUEnableGate.inputs:step", False),
                call(self.graph_path + "/LidarEnableGate.inputs:step", True),
                call(self.graph_path + "/LidarEnableGate.inputs:step", False),
            ],
        )
        self.relay.stop.assert_called_once()
        self.runtime._cleanup_lidar.assert_not_called()

    def test_api_imu_lifecycle_only_gates_the_startup_prepared_graph(self) -> None:
        self.runtime.prepare_sensors()

        resource = self.runtime.create_imu()
        self.assertTrue(resource.ready())
        resource.destroy()

        self.imu_create.assert_called_once()
        self.assertEqual(
            self.runtime._set_gate.call_args_list,
            [
                call(self.graph_path + "/LidarEnableGate.inputs:step", False),
                call(self.imu_graph_path + "/IMUEnableGate.inputs:step", False),
                call(self.imu_graph_path + "/IMUEnableGate.inputs:step", True),
                call(self.imu_graph_path + "/IMUEnableGate.inputs:step", False),
            ],
        )
        self.runtime._cleanup_imu.assert_not_called()

    def test_shutdown_removes_prepared_resources_after_closing_both_gates(self) -> None:
        self.runtime.prepare_sensors()
        prepared_lidar = self.runtime._prepared_lidar
        prepared_imu = self.runtime._prepared_imu

        self.runtime.shutdown()

        self.runtime._cleanup_imu.assert_called_once_with(prepared_imu)
        self.runtime._cleanup_lidar.assert_called_once_with(prepared_lidar)
        self.runtime._release_sensor_root.assert_called_once_with(self.stage)
        self.assertIsNone(self.runtime._prepared_lidar)
        self.assertIsNone(self.runtime._prepared_imu)
        self.assertEqual(
            self.runtime._set_gate.call_args_list[-2:],
            [
                call(self.graph_path + "/LidarEnableGate.inputs:step", False),
                call(self.imu_graph_path + "/IMUEnableGate.inputs:step", False),
            ],
        )

    def test_shutdown_preserves_resources_for_retry_after_cleanup_failures(self) -> None:
        self.runtime.prepare_sensors()
        prepared_lidar = self.runtime._prepared_lidar
        prepared_imu = self.runtime._prepared_imu
        self.runtime._lidar_relay = self.relay
        self.runtime._set_gate.side_effect = [
            RuntimeError("lidar gate failed"),
            None,
            None,
            None,
        ]
        self.relay.stop.side_effect = [RuntimeError("relay stop failed"), None]
        self.runtime._cleanup_imu.side_effect = [RuntimeError("imu cleanup failed"), None]
        self.runtime._cleanup_lidar.side_effect = [RuntimeError("lidar cleanup failed"), None]
        self.runtime._release_sensor_root.side_effect = [RuntimeError("root cleanup failed"), None]

        with self.assertRaisesRegex(RuntimeError, "sensor shutdown cleanup failed") as raised:
            self.runtime.shutdown()

        for message in (
            "lidar gate failed",
            "relay stop failed",
            "imu cleanup failed",
            "lidar cleanup failed",
            "root cleanup failed",
        ):
            self.assertIn(message, str(raised.exception))
        self.assertIs(self.runtime._prepared_lidar, prepared_lidar)
        self.assertIs(self.runtime._prepared_imu, prepared_imu)
        self.assertIs(self.runtime._prepared_stage, self.stage)
        self.assertIs(self.runtime._lidar_relay, self.relay)
        self.runtime._set_gate.assert_has_calls(
            [
                call(self.graph_path + "/LidarEnableGate.inputs:step", False),
                call(self.imu_graph_path + "/IMUEnableGate.inputs:step", False),
            ]
        )
        self.runtime._cleanup_imu.assert_called_once_with(prepared_imu)
        self.runtime._cleanup_lidar.assert_called_once_with(prepared_lidar)
        self.runtime._release_sensor_root.assert_called_once_with(self.stage)

        self.runtime.shutdown()

        self.assertIsNone(self.runtime._prepared_lidar)
        self.assertIsNone(self.runtime._prepared_imu)
        self.assertIsNone(self.runtime._prepared_stage)
        self.assertIsNone(self.runtime._lidar_relay)
        self.assertEqual(self.runtime._cleanup_imu.call_count, 2)
        self.assertEqual(self.runtime._cleanup_lidar.call_count, 2)
        self.assertEqual(self.runtime._release_sensor_root.call_count, 2)


if __name__ == "__main__":
    unittest.main()
