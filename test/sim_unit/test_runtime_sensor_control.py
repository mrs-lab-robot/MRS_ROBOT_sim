from __future__ import annotations

import json
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

import openflex_isaac_sensors.runtime_control as runtime_control
import openflex_isaac_sensors.runtime_sensors as runtime_sensors
import openflex_isaac_sensors.mid360 as mid360
import openflex_isaac_sensors.rig as rig_module
from openflex_isaac_sensors.runtime_control import (
    RuntimeSensorControlServer,
    RuntimeSensorManager,
    RuntimeSensorResource,
)
from openflex_isaac_sensors.runtime_sensors import (
    RobotSensorRuntime,
    _CameraPublisherBootstrap,
)
from openflex_isaac_sensors.rig import RealSenseRig
from openflex_isaac_sensors.integration import (
    load_realsense_config,
    resolve_robot_mount_path,
)


class RuntimeSensorControlTest(unittest.TestCase):
    def setUp(self):
        self.owner_thread = threading.get_ident()
        self.created_on = []
        self.destroyed_on = []
        self.manager = RuntimeSensorManager(request_timeout_s=2.0)

        def create_camera():
            self.created_on.append(threading.get_ident())

            def destroy_camera():
                self.destroyed_on.append(threading.get_ident())

            return destroy_camera

        self.manager.register(
            "camera_head",
            label="头部相机",
            topic="/cam_head/color/image",
            create=create_camera,
        )

        self.readiness_polls = []

        def create_delayed_sensor():
            def ready():
                self.readiness_polls.append(threading.get_ident())
                return len(self.readiness_polls) >= 3

            return RuntimeSensorResource(destroy=lambda: None, ready=ready)

        self.manager.register(
            "camera_delayed",
            label="延迟就绪相机",
            topic="/cam_delayed/color/image",
            create=create_delayed_sensor,
        )

        self.retry_destroy_attempts = []

        def create_retry_sensor():
            def destroy():
                self.retry_destroy_attempts.append(threading.get_ident())
                if len(self.retry_destroy_attempts) == 1:
                    raise RuntimeError("temporary cleanup error")

            return destroy

        self.manager.register(
            "camera_retry",
            label="可重试相机",
            topic="/cam_retry/color/image",
            create=create_retry_sensor,
        )
        self.manager.mark_ready()
        self.server = RuntimeSensorControlServer(self.manager, port=0)
        self.server.start()

    def tearDown(self):
        self.server.stop()

    def _post_on_worker_while_isaac_updates(self, action, sensor_id="camera_head"):
        outcomes = []

        def send_request():
            request = Request(
                f"{self.server.base_url}/v1/sensors/{sensor_id}",
                data=json.dumps({"action": action}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=3.0) as response:
                    outcomes.append((response.status, json.loads(response.read())))
            except HTTPError as error:
                outcomes.append((error.code, json.loads(error.read())))

        worker = threading.Thread(target=send_request)
        worker.start()
        deadline = time.monotonic() + 2.0
        while worker.is_alive() and time.monotonic() < deadline:
            self.manager.process_pending()
            time.sleep(0.005)
        worker.join(timeout=0.1)
        self.assertFalse(worker.is_alive(), "sensor lifecycle request did not complete")
        self.assertEqual(len(outcomes), 1)
        return outcomes[0]

    def test_http_create_and_destroy_manage_sensor_resources_on_isaac_thread(self):
        with urlopen(f"{self.server.base_url}/v1/sensors", timeout=2.0) as response:
            initial = json.loads(response.read())
        self.assertEqual(initial["sensors"]["camera_head"]["state"], "inactive")
        self.assertFalse(initial["sensors"]["camera_head"]["active"])

        status, created = self._post_on_worker_while_isaac_updates("create")
        self.assertEqual(status, 200)
        self.assertEqual(created["sensors"]["camera_head"]["state"], "active")
        self.assertEqual(self.created_on, [self.owner_thread])
        self.assertEqual(self.destroyed_on, [])

        status, destroyed = self._post_on_worker_while_isaac_updates("destroy")
        self.assertEqual(status, 200)
        self.assertEqual(destroyed["sensors"]["camera_head"]["state"], "inactive")
        self.assertEqual(self.destroyed_on, [self.owner_thread])

    def test_repeated_create_is_idempotent_and_does_not_duplicate_resources(self):
        self.assertEqual(
            self._post_on_worker_while_isaac_updates("create", "camera_retry")[0], 200
        )
        self.assertEqual(self._post_on_worker_while_isaac_updates("create")[0], 200)
        self.assertEqual(self.created_on, [self.owner_thread])

    def test_create_waits_until_runtime_resource_reports_ready(self):
        status, response = self._post_on_worker_while_isaac_updates(
            "create", "camera_delayed"
        )

        self.assertEqual(status, 200)
        self.assertEqual(response["sensors"]["camera_delayed"]["state"], "active")
        self.assertGreaterEqual(len(self.readiness_polls), 3)
        self.assertTrue(all(
            thread_id == self.owner_thread for thread_id in self.readiness_polls
        ))

    def test_unknown_sensor_and_invalid_action_are_rejected(self):
        request = Request(
            f"{self.server.base_url}/v1/sensors/camera_missing",
            data=b'{"action":"create"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=2.0)
        self.assertEqual(error.exception.code, 404)

        status, payload = self._post_on_worker_while_isaac_updates("toggle")
        self.assertEqual(status, 400)
        self.assertIn("action", payload["error"])

        malformed = Request(
            f"{self.server.base_url}/v1/sensors/camera_head",
            data=b'{"action":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as malformed_error:
            urlopen(malformed, timeout=2.0)
        self.assertEqual(malformed_error.exception.code, 400)

    def test_destroy_failure_keeps_resource_for_retry(self):
        self.assertEqual(
            self._post_on_worker_while_isaac_updates("create", "camera_retry")[0], 200
        )
        self.assertEqual(self._post_on_worker_while_isaac_updates("destroy", "camera_retry")[0], 409)
        failed = self.manager.snapshot()["sensors"]["camera_retry"]
        self.assertEqual(failed["state"], "error")
        self.assertTrue(failed["resource_present"])
        self.assertIn("temporary cleanup error", failed["error"])
        self.assertEqual(
            self._post_on_worker_while_isaac_updates("create", "camera_retry")[0], 409
        )
        self.assertEqual(len(self.retry_destroy_attempts), 1)

        status, destroyed = self._post_on_worker_while_isaac_updates("destroy", "camera_retry")
        self.assertEqual(status, 200)
        self.assertEqual(destroyed["sensors"]["camera_retry"]["state"], "inactive")
        self.assertEqual(
            self.retry_destroy_attempts, [self.owner_thread, self.owner_thread]
        )


class Isaac51SensorApiCompatibilityTest(unittest.TestCase):
    def test_bridge_extension_loader_uses_stable_core_utils_api(self):
        from openflex_isaac_sensors.sinks import _enable_isaac_extension

        calls = []
        isaacsim = ModuleType("isaacsim")
        isaacsim.__path__ = []
        core = ModuleType("isaacsim.core")
        core.__path__ = []
        utils = ModuleType("isaacsim.core.utils")
        utils.__path__ = []
        extensions = ModuleType("isaacsim.core.utils.extensions")
        extensions.enable_extension = lambda name: calls.append(name) or True

        with patch.dict(
            sys.modules,
            {
                "isaacsim": isaacsim,
                "isaacsim.core": core,
                "isaacsim.core.utils": utils,
                "isaacsim.core.utils.extensions": extensions,
            },
        ):
            self.assertTrue(_enable_isaac_extension("isaacsim.ros2.bridge"))

        self.assertEqual(calls, ["isaacsim.ros2.bridge"])

    def test_realsense_rig_authors_usd_camera_without_experimental_sensor_api(self):
        class Attribute:
            def __init__(self):
                self.value = None

            def IsValid(self):
                return True

            def Set(self, value):
                self.value = value

        class Prim:
            def __init__(self, path, parent=None):
                self.path = path
                self.parent = parent
                self.attributes = {}

            def __bool__(self):
                return True

            def IsValid(self):
                return True

            def IsInstanceable(self):
                return False

            def IsInstanceProxy(self):
                return False

            def GetParent(self):
                return self.parent or InvalidPrim()

            def GetAttribute(self, name):
                return self.attributes.get(name, InvalidAttribute())

            def CreateAttribute(self, name, _type, custom=True):
                attribute = Attribute()
                self.attributes[name] = attribute
                return attribute

        class InvalidPrim:
            def __bool__(self):
                return False

            def IsValid(self):
                return False

        class InvalidAttribute:
            def __bool__(self):
                return False

            def IsValid(self):
                return False

        class CameraSchema:
            def __init__(self, prim):
                self.prim = prim
                self.focal_length = Attribute()
                self.clipping_range = Attribute()
                self.horizontal_aperture = Attribute()
                self.vertical_aperture = Attribute()

            def GetPrim(self):
                return self.prim

            def GetFocalLengthAttr(self):
                return self.focal_length

            def GetClippingRangeAttr(self):
                return self.clipping_range

            def GetHorizontalApertureAttr(self):
                return self.horizontal_aperture

            def GetVerticalApertureAttr(self):
                return self.vertical_aperture

        class Stage:
            def __init__(self):
                parent = Prim("/openflex/link")
                self.prims = {"/openflex/link": parent}
                self.cameras = {}

            def GetPrimAtPath(self, path):
                return self.prims.get(str(path), InvalidPrim())

            def DefinePrim(self, path, _type):
                prim = Prim(str(path))
                self.prims[str(path)] = prim
                return prim

        stage = Stage()

        class CameraType:
            @staticmethod
            def Define(stage_arg, path):
                prim = Prim(str(path), stage_arg.GetPrimAtPath("/openflex/link"))
                stage_arg.prims[str(path)] = prim
                schema = CameraSchema(prim)
                stage_arg.cameras[str(path)] = schema
                return schema

        pxr = ModuleType("pxr")
        pxr.Gf = SimpleNamespace(Vec2f=lambda near, far: (near, far))
        pxr.Sdf = SimpleNamespace(
            Path=lambda value: str(value),
            ValueTypeNames=SimpleNamespace(Float=object()),
        )
        pxr.UsdGeom = SimpleNamespace(Camera=CameraType)

        rig = RealSenseRig("camera_head")
        with patch.dict(sys.modules, {"pxr": pxr}), patch.object(
            rig_module, "apply_local_pose", lambda *_args: None
        ), patch.object(
            rig,
            "create_python_render_product",
            side_effect=lambda: setattr(rig, "_render_product_path", "/Render/camera_head")
            or "/Render/camera_head",
        ):
            rig.create(
                stage,
                "/openflex/link",
                None,
                {
                    "focal_length_mm": 3.7,
                    "near_m": 0.08,
                    "far_m": 8.0,
                    "width": 640,
                    "height": 480,
                    "calibration": {
                        "width": 640,
                        "height": 480,
                        "K": [415.0, 0.0, 320.0, 0.0, 415.0, 240.0, 0.0, 0.0, 1.0],
                    },
                },
                create_render_product=True,
            )

        camera = stage.cameras["/openflex/link/camera_head_Mount/camera_head_Camera"]
        self.assertEqual(camera.focal_length.value, 3.7)
        self.assertEqual(camera.clipping_range.value, (0.08, 8.0))
        self.assertAlmostEqual(camera.horizontal_aperture.value, 3.7 * 640.0 / 415.0)
        self.assertAlmostEqual(camera.vertical_aperture.value, 3.7 * 480.0 / 415.0)
        self.assertEqual(rig._sensor.__class__.__name__, "_DirectCameraProduct")
        self.assertEqual(rig.render_product_path, "/Render/camera_head")
        self.assertEqual(
            stage.prims["/openflex/link/camera_head_Mount/camera_head_Camera"]
            .attributes["omni:sensor:tickRate"].value,
            30.0,
        )

    def test_lidar_helper_reuses_isaac_51_sensor_render_product(self):
        events = []

        class StableLidarRtx:
            _render_product = SimpleNamespace(path="/Render/StableLidar")

            def get_render_product_path(self):
                return self._render_product.path

            def attach_annotator(self, name):
                events.append(("annotator", name))

        lidar = StableLidarRtx()
        with patch.object(mid360, "_LIDAR_SENSOR_HANDLES", []), patch.object(
            mid360, "_LIDAR_RENDER_PRODUCT_HANDLES", []
        ) as render_handles:
            product_path, sensor = mid360._create_lidar_render_product(lidar, "livox_frame")

        self.assertEqual(product_path, "/Render/StableLidar")
        self.assertIs(sensor, lidar)
        self.assertEqual(events, [("annotator", "GenericModelOutput")])
        self.assertIs(render_handles[0], lidar._render_product)

    def test_lidar_factory_falls_back_to_isaac_51_lidarrtx(self):
        calls = []

        class StableLidarRtx:
            def __init__(self, *, prim_path, name, config_file_name, **kwargs):
                calls.append((prim_path, name, config_file_name, kwargs))
                self.prim_path = prim_path

        isaacsim = ModuleType("isaacsim")
        isaacsim.__path__ = []
        sensors = ModuleType("isaacsim.sensors")
        sensors.__path__ = []
        rtx = ModuleType("isaacsim.sensors.rtx")
        rtx.LidarRtx = StableLidarRtx
        modules = {
            "isaacsim": isaacsim,
            "isaacsim.sensors": sensors,
            "isaacsim.sensors.rtx": rtx,
        }

        class Prim:
            def IsValid(self):
                return True

        class Stage:
            def GetPrimAtPath(self, _path):
                return Prim()

        with patch.dict(sys.modules, modules):
            lidar = mid360._create_lidar(Stage(), "/World/Lidar", tick_rate_hz=10.0)

        self.assertEqual(lidar.prim_path, "/World/Lidar")
        self.assertEqual(calls[0][:2], ("/World/Lidar", "openflex_mid360"))
        self.assertTrue(calls[0][2])

    def test_mid360_remote_isaac_profile_resolution_uses_configured_mid360_variant(self):
        config, variant = mid360._resolve_mid360_config("MID360_APPROX")

        self.assertEqual(config, "multiScan100")
        self.assertEqual(
            variant,
            {"Product": "multiScan165", "Profile": "Profile01_20Hz_0p5deg"},
        )

    def test_mid360_performance_profile_preserves_variant_scan_rate_and_geometry(self):
        attributes = {
            "omni:sensor:Core:scanType": "MULTI_SCAN",
            "omni:sensor:Core:patternFiringRateHz": 20,
            "omni:sensor:Core:scanRateBaseHz": 20,
            "omni:sensor:Core:validStartAzimuthDeg": -180.0,
            "omni:sensor:Core:validEndAzimuthDeg": 180.0,
            "omni:sensor:Core:emitterState:s001:elevationDeg": [-10.0, 0.0, 10.0],
        }

        class Attribute:
            def __init__(self, name):
                self.name = name

            def IsValid(self):
                return True

            def Get(self):
                return attributes.get(self.name)

            def Set(self, value):
                attributes[self.name] = value

        class Prim:
            def IsValid(self):
                return True

            def HasAPI(self, _api):
                return True

            def GetAttribute(self, name):
                return Attribute(name)

        class Stage:
            def GetPrimAtPath(self, _path):
                return Prim()

        mid360._configure_mid360_performance_profile(Stage(), "/World/Lidar")

        self.assertEqual(attributes["omni:sensor:Core:scanType"], "MULTI_SCAN")
        self.assertEqual(attributes["omni:sensor:Core:scanRateBaseHz"], 20)
        self.assertEqual(attributes["omni:sensor:Core:validStartAzimuthDeg"], -180.0)
        self.assertEqual(attributes["omni:sensor:Core:validEndAzimuthDeg"], 180.0)
        self.assertEqual(
            attributes["omni:sensor:Core:emitterState:s001:elevationDeg"],
            [-10.0, 0.0, 10.0],
        )
        self.assertEqual(attributes["omni:sensor:Core:patternFiringRateHz"], 20)
        self.assertEqual(attributes["omni:sensor:Core:maxReturns"], 1)
        self.assertEqual(attributes["omni:sensor:Core:nearRangeM"], 0.1)
        self.assertEqual(attributes["omni:sensor:Core:farRangeM"], 40.0)

    def test_mid360_full_profile_preserves_variant_scan_rate(self):
        attributes = {"omni:sensor:Core:patternFiringRateHz": 20}

        class Attribute:
            def __init__(self, name):
                self.name = name

            def IsValid(self):
                return True

            def Set(self, value):
                attributes[self.name] = value

        class Prim:
            def IsValid(self):
                return True

            def HasAPI(self, _api):
                return True

            def GetAttribute(self, name):
                return Attribute(name)

        class Stage:
            def GetPrimAtPath(self, _path):
                return Prim()

        mid360._configure_mid360_full_profile(Stage(), "/World/Lidar")

        self.assertEqual(attributes["omni:sensor:Core:patternFiringRateHz"], 20)
        self.assertEqual(attributes["omni:sensor:Core:nearRangeM"], 0.1)
        self.assertEqual(attributes["omni:sensor:Core:farRangeM"], 40.0)

    def test_imu_factory_falls_back_to_isaac_51_physics_sensor(self):
        calls = []

        class StableImuSensor:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        isaacsim = ModuleType("isaacsim")
        isaacsim.__path__ = []
        sensors = ModuleType("isaacsim.sensors")
        sensors.__path__ = []
        physics = ModuleType("isaacsim.sensors.physics")
        physics.IMUSensor = StableImuSensor

        with patch.dict(
            sys.modules,
            {
                "isaacsim": isaacsim,
                "isaacsim.sensors": sensors,
                "isaacsim.sensors.physics": physics,
            },
        ), patch.object(mid360, "_IMU_SENSOR_HANDLES", {}):
            sensor = mid360._make_imu_sensor("/World/IMU", frequency_hz=160.0)

        self.assertIsInstance(sensor, StableImuSensor)
        self.assertEqual(calls[0]["prim_path"], "/World/IMU")
        self.assertEqual(calls[0]["frequency"], 160)
        self.assertEqual(calls[0]["name"], "openflex_imu")



class RobotSensorCatalogTest(unittest.TestCase):
    def test_realsense_loader_resolves_the_referenced_nominal_calibrations(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        camera_config = load_realsense_config(repo_root)

        cameras = camera_config["cameras"]
        self.assertEqual(
            cameras["base_d435"]["calibration"]["calibration_id"],
            "d435_nominal_640x480_rectified_v1",
        )
        self.assertEqual(
            cameras["left_wrist_d405"]["calibration"]["calibration_id"],
            "d405_nominal_640x480_rectified_v1",
        )
        self.assertEqual(cameras["base_d435"]["calibration"]["K"][0], 415.0)

    def test_head_camera_mount_targets_the_imported_head_yaw_link(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        camera_config = load_realsense_config(repo_root)
        head_camera = camera_config["cameras"]["head_d435"]

        parent_path = resolve_robot_mount_path(
            head_camera["parent_prim"], "/openflex"
        )
        mount_path = resolve_robot_mount_path(
            head_camera["mount_prim_path"], "/openflex"
        )

        expected_parent_path = (
            "/openflex/Geometry/base_link/lift_carriage_link/"
            "head_pitch_link/head_yaw_link"
        )
        self.assertEqual(parent_path, expected_parent_path)
        self.assertEqual(mount_path, expected_parent_path + "/HeadCameraMount")

        robot_usd = repo_root / "sim_runtime" / "assets" / "robots" / "openflex_robot.usda"
        asset_parent_path = (
            "/openarmx_integrated/Geometry/base_link/lift_carriage_link/"
            "head_pitch_link/head_yaw_link"
        )
        asset_text = robot_usd.read_text(encoding="utf-8")
        self.assertTrue(
            f"<{asset_parent_path}>" in asset_text,
            "the configured head camera parent must exist in the canonical robot USD hierarchy",
        )

    def test_robot_sensor_catalog_registers_inactive_factories_without_touching_stage(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        stage_accesses = []
        manager = RuntimeSensorManager()
        runtime = RobotSensorRuntime(
            stage_getter=lambda: stage_accesses.append("stage requested"),
            robot_prim_path="/openflex",
            realsense_asset_dir=repo_root,
            mid360_asset_dir=repo_root,
        )

        runtime.register(manager)
        manager.mark_ready()
        snapshot = manager.snapshot()

        self.assertEqual(
            set(snapshot["sensors"]),
            {"camera_base", "camera_head", "camera_left", "camera_right", "lidar", "imu"},
        )
        self.assertEqual(snapshot["sensors"]["camera_head"]["topic"], "/cam_head/color/image")
        self.assertEqual(snapshot["sensors"]["lidar"]["topic"], "/openflex/livox_frame/lidar")
        self.assertTrue(all(
            sensor["state"] == "inactive" and not sensor["active"]
            for sensor in snapshot["sensors"].values()
        ))
        self.assertEqual(stage_accesses, [])

    def test_lidar_runtime_defaults_are_loaded_from_the_isaac_sensor_yaml(self):
        fake_lidar_config = {
            "sensor": {
                "default_profile": "REMOTE_BASE_MODEL",
                "performance_profile": "REMOTE_PERFORMANCE_MODE",
                "tick_rate_hz": 17.0,
            }
        }
        with patch.object(
            runtime_sensors, "load_realsense_config", return_value={"cameras": {}}
        ), patch.object(runtime_sensors, "load_mid360_config", return_value=fake_lidar_config):
            runtime = RobotSensorRuntime(
                stage_getter=lambda: object(),
                robot_prim_path="/openflex",
                realsense_asset_dir="/unused",
                mid360_asset_dir="/unused",
            )

        self.assertEqual(runtime._lidar_model_config, "REMOTE_BASE_MODEL")
        self.assertEqual(runtime._lidar_profile, "REMOTE_PERFORMANCE_MODE")
        self.assertEqual(runtime._lidar_tick_rate_hz, 17.0)

    def test_lidar_runtime_passes_the_yaml_base_profile_into_sensor_creation(self):
        fake_lidar_config = {
            "sensor": {
                "default_profile": "REMOTE_BASE_MODEL",
                "performance_profile": "REMOTE_PERFORMANCE_MODE",
                "tick_rate_hz": 17.0,
            }
        }
        with patch.object(
            runtime_sensors, "load_realsense_config", return_value={"cameras": {}}
        ), patch.object(runtime_sensors, "load_mid360_config", return_value=fake_lidar_config):
            runtime = RobotSensorRuntime(
                stage_getter=lambda: object(),
                robot_prim_path="/openflex",
                realsense_asset_dir="/unused",
                mid360_asset_dir="/unused",
            )

        runtime._mount_path = "/openflex/BaseCameraMount"
        with patch.object(
            runtime,
            "_ensure_mid360_mount_and_root",
            return_value="/World/OpenFlex/MID360",
        ), patch.object(
            mid360,
            "create_robot_mid360",
            return_value="/World/OpenFlex/MID360/Lidar",
        ) as create_lidar:
            runtime._create_lidar()

        self.assertEqual(create_lidar.call_args.kwargs["sensor_config"], "REMOTE_BASE_MODEL")
        self.assertEqual(create_lidar.call_args.kwargs["tick_rate_hz"], 17.0)

    def test_sensor_configuration_error_is_reported_without_faking_a_catalog(self):
        manager = RuntimeSensorManager()
        manager.mark_ready(error="missing sensor asset config")

        snapshot = manager.snapshot()

        self.assertTrue(snapshot["ready"])
        self.assertEqual(snapshot["error"], "missing sensor asset config")
        self.assertEqual(snapshot["sensors"], {})


class MainLoopSensorPumpTest(unittest.TestCase):
    def test_rest_runner_loop_polls_delayed_sensor_until_it_is_active(self):
        install_pump = getattr(
            runtime_control, "install_main_loop_sensor_pump", None
        )
        self.assertTrue(
            callable(install_pump),
            "the Kit main-loop command pump must own sensor lifecycle polling",
        )

        owner_thread = threading.get_ident()
        manager = RuntimeSensorManager(request_timeout_s=2.0)
        readiness_threads = []

        def create_sensor():
            def ready():
                readiness_threads.append(threading.get_ident())
                return len(readiness_threads) >= 3

            return RuntimeSensorResource(destroy=lambda: None, ready=ready)

        manager.register(
            "camera_pump",
            label="主循环轮询相机",
            topic="/cam_pump/color/image",
            create=create_sensor,
        )
        manager.mark_ready()

        class SimulationLoop:
            def __init__(self):
                self.rest_command_polls = 0

            def process_commands(self):
                self.rest_command_polls += 1

        simulation_loop = SimulationLoop()
        install_pump(simulation_loop, manager)
        outcomes = []

        def request_create():
            outcomes.append(manager.request("camera_pump", "create"))

        worker = threading.Thread(target=request_create)
        worker.start()
        deadline = time.monotonic() + 1.5
        while worker.is_alive() and time.monotonic() < deadline:
            simulation_loop.process_commands()
            time.sleep(0.005)
        worker.join(timeout=0.1)

        self.assertFalse(worker.is_alive(), "main-loop sensor request did not finish")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["sensors"]["camera_pump"]["state"], "active")
        self.assertGreaterEqual(len(readiness_threads), 3)
        self.assertTrue(all(thread_id == owner_thread for thread_id in readiness_threads))
        self.assertGreaterEqual(simulation_loop.rest_command_polls, 3)


class RuntimeSensorDeferredDestroyTest(unittest.TestCase):
    def test_destroy_stays_pending_until_runtime_reports_cleanup_complete(self):
        owner_thread = threading.get_ident()
        manager = RuntimeSensorManager(request_timeout_s=2.0)
        destroy_threads = []
        readiness_threads = []
        readiness_states = []

        def create_sensor():
            def destroy():
                destroy_threads.append(threading.get_ident())

            def destroy_ready():
                readiness_threads.append(threading.get_ident())
                readiness_states.append(manager.snapshot()["sensors"]["camera_delayed_destroy"]["state"])
                return len(readiness_threads) >= 3

            return RuntimeSensorResource(
                destroy=destroy,
                destroy_ready=destroy_ready,
            )

        manager.register(
            "camera_delayed_destroy",
            label="延迟销毁相机",
            topic="/cam_delayed_destroy/color/image",
            create=create_sensor,
        )
        manager.mark_ready()

        def request_while_processing(action):
            outcomes = []

            def request():
                outcomes.append(manager.request("camera_delayed_destroy", action))

            worker = threading.Thread(target=request)
            worker.start()
            deadline = time.monotonic() + 1.0
            while worker.is_alive() and time.monotonic() < deadline:
                manager.process_pending()
                time.sleep(0.002)
            worker.join(timeout=0.1)
            self.assertFalse(worker.is_alive(), f"{action} request did not finish")
            self.assertEqual(len(outcomes), 1)
            return outcomes[0]

        self.assertEqual(
            request_while_processing("create")["sensors"]["camera_delayed_destroy"]["state"],
            "active",
        )
        result = request_while_processing("destroy")

        self.assertEqual(result["sensors"]["camera_delayed_destroy"]["state"], "inactive")
        self.assertEqual(destroy_threads, [owner_thread])
        self.assertEqual(readiness_threads, [owner_thread] * 3)
        self.assertEqual(readiness_states, ["destroying"] * 3)

    def test_failed_camera_bootstrap_keeps_deferred_cleanup_on_the_update_thread(self):
        manager = RuntimeSensorManager(request_timeout_s=2.0)
        cleanup_updates = []
        destroyed = []

        def create_sensor():
            def ready():
                raise RuntimeError("publisher gate setup failed")

            def destroy_ready():
                cleanup_updates.append(threading.get_ident())
                return len(cleanup_updates) >= 3

            return RuntimeSensorResource(
                destroy=lambda: destroyed.append(threading.get_ident()),
                ready=ready,
                destroy_ready=destroy_ready,
            )

        manager.register(
            "camera_bootstrap_error",
            label="初始化失败的相机",
            topic="/cam_bootstrap_error/color/image",
            create=create_sensor,
        )
        manager.mark_ready()
        request_errors = []

        def request_create():
            try:
                manager.request("camera_bootstrap_error", "create")
            except RuntimeError as exc:
                request_errors.append(str(exc))

        worker = threading.Thread(target=request_create)
        worker.start()
        manager.process_pending()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive(), "failed create request did not return")
        self.assertEqual(len(request_errors), 1)
        self.assertIn("publisher gate setup failed", request_errors[0])
        self.assertEqual(destroyed, [threading.get_ident()])
        self.assertTrue(manager.snapshot()["sensors"]["camera_bootstrap_error"]["resource_present"])

        manager.process_pending()
        self.assertTrue(manager.snapshot()["sensors"]["camera_bootstrap_error"]["resource_present"])
        manager.process_pending()

        sensor = manager.snapshot()["sensors"]["camera_bootstrap_error"]
        self.assertEqual(sensor["state"], "error")
        self.assertFalse(sensor["resource_present"])
        self.assertEqual(cleanup_updates, [threading.get_ident()] * 3)


class CameraResourceDeferredTeardownTest(unittest.TestCase):
    def test_render_product_is_not_removed_until_quiescence_updates_complete(self):
        make_resource = getattr(runtime_sensors, "_make_camera_runtime_resource", None)
        self.assertTrue(
            callable(make_resource),
            "camera teardown must defer RenderProduct removal across Kit updates",
        )

        events = []
        live_paths = {"/camera", "/render_product"}

        class Attribute:
            def IsValid(self):
                return True

            def Set(self, value):
                events.append(("tick_rate", value))

        class Prim:
            def IsValid(self):
                return True

            def __bool__(self):
                return True

            def GetAttribute(self, _name):
                return Attribute()

        class Stage:
            def GetPrimAtPath(self, path):
                return Prim() if path in live_paths else None

        class Bridge:
            def set_camera_enabled(self, name, enabled):
                events.append(("publishers", name, enabled))

            def destroy(self, _stage):
                events.append(("bridge_destroy",))

        class Rig:
            def destroy(self):
                events.append(("rig_destroy",))
                live_paths.clear()

        resource = make_resource(
            bridge=Bridge(),
            camera_key="camera_head",
            rig=Rig(),
            stage=Stage(),
            camera_prim_path="/camera",
            render_product_path="/render_product",
            settle_updates=3,
        )

        resource.destroy()

        self.assertNotIn(("rig_destroy",), events)
        self.assertFalse(resource.destroy_ready())
        self.assertFalse(resource.destroy_ready())
        self.assertNotIn(("rig_destroy",), events)
        self.assertTrue(resource.destroy_ready())
        self.assertIn(("rig_destroy",), events)
        self.assertEqual(events[0], ("publishers", "camera_head", False))
        self.assertEqual(events[1], ("tick_rate", 0.0))

    def test_rig_releases_python_render_product_through_its_owner_api(self):
        events = []
        live_paths = {"/camera", "/render_product"}

        class Prim:
            def IsValid(self):
                return True

            def __bool__(self):
                return True

        class Stage:
            def GetPrimAtPath(self, path):
                return Prim() if str(path) in live_paths else None

            def RemovePrim(self, path):
                path = str(path)
                events.append(("stage_remove", path))
                live_paths.discard(path)

        class RenderProduct:
            def destroy(self):
                events.append(("render_product_destroy",))
                live_paths.discard("/render_product")

        rig = RealSenseRig("camera_head")
        rig._stage = Stage()
        rig._camera_path = "/camera"
        rig._camera_prim_path = "/camera"
        rig._render_product_path = "/render_product"
        rig._render_product_handle = RenderProduct()

        class FakeSdfPath(str):
            pass

        pxr_module = ModuleType("pxr")
        pxr_module.Sdf = SimpleNamespace(Path=FakeSdfPath)
        with patch.dict(sys.modules, {"pxr": pxr_module}):
            rig.destroy()

        self.assertEqual(events[0], ("render_product_destroy",))
        self.assertNotIn(("stage_remove", "/render_product"), events)
        self.assertIn(("stage_remove", "/camera"), events)


class CameraPublisherBootstrapTest(unittest.TestCase):
    def test_restores_configured_tick_rate_after_first_render_product_tick(self):
        events = []

        class Bridge:
            def set_camera_enabled(self, _name, _enabled):
                pass

            def configure_camera_gates(self, _cameras):
                pass

        bootstrap = _CameraPublisherBootstrap(
            Bridge(),
            "camera_base",
            [{"camera_key": "camera_base"}],
            set_target_tick_rate=lambda: events.append("target-rate"),
            warmup_updates=3,
        )

        self.assertFalse(bootstrap.ready())
        self.assertEqual(events, [])
        self.assertFalse(bootstrap.ready())
        self.assertEqual(events, ["target-rate"])
        self.assertFalse(bootstrap.ready())
        self.assertFalse(bootstrap.ready())
        self.assertFalse(bootstrap.ready())
        self.assertTrue(bootstrap.ready())
        self.assertEqual(events, ["target-rate"])

    def test_enable_camera_graph_before_resolving_synthetic_data_gates(self):
        events = []

        class Bridge:
            def set_camera_enabled(self, name, enabled):
                events.append(("enabled", name, enabled))

            def configure_camera_gates(self, cameras):
                events.append(("configure", cameras))

        record = {"camera_key": "camera_head", "render_product_path": "/Render/RP"}
        bootstrap = _CameraPublisherBootstrap(
            Bridge(), "camera_head", [record], warmup_updates=3
        )

        self.assertFalse(bootstrap.ready())
        self.assertFalse(bootstrap.ready())
        self.assertFalse(bootstrap.ready())
        self.assertEqual(events, [("enabled", "camera_head", True)])

        self.assertFalse(bootstrap.ready())
        self.assertFalse(bootstrap.ready())
        self.assertTrue(bootstrap.ready())
        self.assertEqual(events[0], ("enabled", "camera_head", True))
        self.assertEqual(events[1], ("configure", [record]))
        self.assertTrue(bootstrap.ready())
        self.assertEqual(len(events), 2)


class RealSenseRenderProductIdentityTest(unittest.TestCase):
    def test_recreating_same_camera_uses_a_fresh_render_product_path(self):
        paths = []

        def create_render_product(camera_path, resolution, *, name):
            path = f"/Render/OmniverseKit/HydraTextures/{name}"
            paths.append((camera_path, resolution, path))
            return SimpleNamespace(path=path)

        omni_module = ModuleType("omni")
        omni_module.__path__ = []
        replicator_module = ModuleType("omni.replicator")
        replicator_module.__path__ = []
        core_module = ModuleType("omni.replicator.core")
        core_module.create = SimpleNamespace(render_product=create_render_product)

        with patch.dict(
            sys.modules,
            {
                "omni": omni_module,
                "omni.replicator": replicator_module,
                "omni.replicator.core": core_module,
            },
        ):
            for _ in range(2):
                rig = RealSenseRig("base_d435")
                rig._camera_path = "/openflex/base_d435_Camera"
                rig.create_python_render_product()

        self.assertEqual(paths[0][0], "/openflex/base_d435_Camera")
        self.assertEqual(paths[0][1], (640, 480))
        self.assertNotEqual(paths[0][2], paths[1][2])


if __name__ == "__main__":
    unittest.main()
