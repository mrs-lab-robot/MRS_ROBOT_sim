from __future__ import annotations

import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from mrs_robot_lab.recorders.arena_sensor_api import (
    ArenaSensorControlServer,
    ArenaSensorManager,
    SensorResource,
)


class ArenaSensorControlServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = ArenaSensorManager(request_timeout_s=2.0)
        self.owner_thread = threading.get_ident()
        self.effects: list[tuple[str, int]] = []
        self.healthy = True
        self.manager.register(
            "lidar",
            label="MID360",
            topic="/livox/lidar",
            create=lambda: SensorResource(destroy=lambda: None),
        )
        self.manager.register(
            "imu",
            label="IMU",
            topic="/livox/imu",
            create=lambda: self._create_imu_resource(),
        )
        self.manager.register_readonly(
            "camera_base",
            label="底盘相机",
            topic="/cam_base/color/image",
            active=True,
        )
        self.manager.mark_ready()
        try:
            self.server = ArenaSensorControlServer(self.manager, port=0)
        except PermissionError as error:
            self.skipTest(f"sandbox does not permit loopback sockets: {error}")
        self.server.start()

    def _create_imu_resource(self) -> SensorResource:
        self.effects.append(("create", threading.get_ident()))
        return SensorResource(
            destroy=lambda: self.effects.append(("destroy", threading.get_ident())),
            health=lambda: self.healthy,
        )

    def tearDown(self) -> None:
        self.server.stop()

    def _request_and_pump(self, path: str, payload: dict | None = None):
        result: list[tuple[int, dict]] = []
        failure: list[BaseException] = []
        done = threading.Event()

        def request() -> None:
            try:
                data = None if payload is None else json.dumps(payload).encode("utf-8")
                request_obj = Request(
                    f"{self.server.base_url}{path}",
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="GET" if payload is None else "POST",
                )
                with urlopen(request_obj, timeout=3.0) as response:
                    result.append((response.status, json.loads(response.read())))
            except HTTPError as error:
                result.append((error.code, json.loads(error.read())))
            except BaseException as error:  # surfaced on the owner thread below
                failure.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=request, daemon=True)
        thread.start()
        deadline = time.monotonic() + 3.0
        while not done.is_set() and time.monotonic() < deadline:
            self.manager.process_pending()
            done.wait(0.001)
        thread.join(timeout=0.1)
        if failure:
            raise failure[0]
        self.assertTrue(done.is_set(), "sensor request did not complete")
        return result[0]

    def test_get_reports_dynamic_lidar_and_existing_arena_camera_state(self) -> None:
        status, payload = self._request_and_pump("/v1/sensors")

        self.assertEqual(status, 200)
        self.assertTrue(payload["ready"])
        self.assertEqual(
            payload["sensors"]["lidar"],
            {
                "label": "MID360",
                "topic": "/livox/lidar",
                "state": "inactive",
                "active": False,
                "resource_present": False,
                "error": "",
            },
        )
        self.assertEqual(payload["sensors"]["imu"]["state"], "inactive")
        self.assertFalse(payload["sensors"]["imu"]["active"])
        self.assertEqual(payload["sensors"]["camera_base"]["state"], "active")
        self.assertTrue(payload["sensors"]["camera_base"]["resource_present"])

    def test_create_and_destroy_are_applied_by_the_owner_thread(self) -> None:
        status, created = self._request_and_pump("/v1/sensors/imu", {"action": "create"})
        self.assertEqual(status, 200)
        self.assertEqual(created["sensors"]["imu"]["state"], "active")

        status, destroyed = self._request_and_pump(
            "/v1/sensors/imu", {"action": "destroy"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(destroyed["sensors"]["imu"]["state"], "inactive")
        self.assertEqual(
            self.effects,
            [("create", self.owner_thread), ("destroy", self.owner_thread)],
        )

    def test_api_rejects_unknown_sensors_and_extra_payload_fields(self) -> None:
        status, unknown = self._request_and_pump(
            "/v1/sensors/not_registered", {"action": "create"}
        )
        self.assertEqual(status, 404)
        self.assertIn("unknown sensor", unknown["error"])

        status, invalid = self._request_and_pump(
            "/v1/sensors/lidar", {"action": "create", "force": True}
        )
        self.assertEqual(status, 400)
        self.assertIn("only the action field", invalid["error"])

    def test_readonly_camera_lifecycle_returns_conflict(self) -> None:
        status, payload = self._request_and_pump(
            "/v1/sensors/camera_base", {"action": "destroy"}
        )
        self.assertEqual(status, 409)
        self.assertIn("session configuration", payload["error"])

    def test_unhealthy_active_relay_is_reported_as_error_but_remains_destroyable(self) -> None:
        status, created = self._request_and_pump("/v1/sensors/imu", {"action": "create"})
        self.assertEqual(status, 200)
        self.assertEqual(created["sensors"]["imu"]["state"], "active")

        self.healthy = False
        self.manager.process_pending()
        state = self.manager.snapshot()["sensors"]["imu"]
        self.assertEqual(state["state"], "error")
        self.assertTrue(state["resource_present"])

        status, destroyed = self._request_and_pump(
            "/v1/sensors/imu", {"action": "destroy"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(destroyed["sensors"]["imu"]["state"], "inactive")

    def test_shutdown_keeps_failed_cleanup_resource_owned_and_releases_others(self) -> None:
        manager = ArenaSensorManager(request_timeout_s=2.0)

        def fail_cleanup() -> None:
            raise RuntimeError("relay stop failed")

        manager.register(
            "lidar",
            label="MID360",
            topic="/livox/lidar",
            create=lambda: SensorResource(destroy=fail_cleanup),
        )
        released: list[bool] = []
        manager.register(
            "imu",
            label="IMU",
            topic="/livox/imu",
            create=lambda: SensorResource(destroy=lambda: released.append(True)),
        )
        manager.mark_ready()

        for sensor_id in ("lidar", "imu"):
            done = threading.Event()
            failure: list[BaseException] = []

            def create_sensor(target: str = sensor_id) -> None:
                try:
                    manager.request(target, "create")
                except BaseException as error:
                    failure.append(error)
                finally:
                    done.set()

            thread = threading.Thread(target=create_sensor, daemon=True)
            thread.start()
            while not done.is_set():
                manager.process_pending()
                done.wait(0.001)
            thread.join(timeout=0.1)
            self.assertFalse(failure)

        manager.shutdown()
        sensors = manager.snapshot()["sensors"]
        self.assertEqual(sensors["lidar"]["state"], "error")
        self.assertTrue(sensors["lidar"]["resource_present"])
        self.assertEqual(sensors["imu"]["state"], "inactive")
        self.assertFalse(sensors["imu"]["resource_present"])
        self.assertEqual(released, [True])


if __name__ == "__main__":
    unittest.main()
