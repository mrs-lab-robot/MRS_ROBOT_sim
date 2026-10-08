"""Tests for the Isaac Lab worker CLI without binding a real socket or booting Kit."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from openflex_isaac_contract.session_config import (
    FrequencyConfig,
    SensorConfig,
    SessionConfig,
    WorkerState,
)
from mrs_robot_lab.recorders.capture_api import CaptureRequest
from mrs_robot_lab import worker_cli
from mrs_robot_lab.worker_cli import _update_api_status, preflight_result, serve_worker
from mrs_robot_lab.workers import WorkerStatus


def make_config() -> SessionConfig:
    return SessionConfig(
        session_id="cli-test",
        embodiment_id="openflex",
        scene_id="flat_navigation",
        task_id="navigation_to_goal",
        frequency=FrequencyConfig(physics_hz=120, control_hz=30, render_hz=30),
    )


class FakeWorker:
    def __init__(self, _config):
        self.state = WorkerState.IDLE
        self.listeners = []
        self.commands = []
        self.pumps = 0
        self.shutdown_calls = 0

    def configure(self):
        self.state = WorkerState.PREFLIGHT

    def launch(self):
        self.state = WorkerState.ENV_READY
        self.notify()

    def get_status(self):
        return WorkerStatus(
            self.state,
            "test status",
            metadata={"session_id": "cli-test", "config_hash": "abc", "steps": 0},
        )

    def add_status_listener(self, listener):
        self.listeners.append(listener)
        listener(self.get_status())

    def notify(self):
        for listener in self.listeners:
            listener(self.get_status())

    def handle_command(self, command, payload):
        self.commands.append((command, payload))
        self.state = WorkerState.STOPPING
        self.notify()

    def pump_once(self):
        self.pumps += 1

    def shutdown(self):
        self.shutdown_calls += 1
        self.state = WorkerState.STOPPING


class FakeServer:
    def __init__(self):
        self.status_updates = []
        self.started = False
        self.closed = False
        self.requests = [CaptureRequest("stop", "/api/v1/session/stop", {})]

    def start(self):
        self.started = True

    def take_requests(self):
        requests, self.requests = self.requests, []
        return requests

    def update_status(self, **values):
        self.status_updates.append(values)

    def close(self):
        self.closed = True


class WorkerCliTest(unittest.TestCase):
    def test_serve_worker_attaches_the_started_api_before_launching_kit(self):
        order = []

        class TransportWorker(FakeWorker):
            def attach_teleop_transport(self, transport):
                order.append(("attach", transport))

            def launch(self):
                order.append(("launch",))
                super().launch()

        worker = TransportWorker(make_config())
        server = FakeServer()

        serve_worker(worker, server, max_iterations=1)

        self.assertEqual(order, [("attach", server), ("launch",)])

    def test_api_status_includes_quarantined_episode_path(self):
        server = FakeServer()
        status = WorkerStatus(
            WorkerState.ENV_READY,
            metadata={
                "session_id": "cli-test",
                "config_hash": "abc",
                "last_quarantined_path": "/episodes/quarantine/ep-1.hdf5",
                "last_qc_failure_reasons": ["缺少 HDF5 元数据"],
                "last_export_error": "LeRobot 环境暂不可用",
            },
        )

        _update_api_status(server, status)

        self.assertEqual(
            server.status_updates[-1]["last_quarantined_path"],
            "/episodes/quarantine/ep-1.hdf5",
        )
        self.assertEqual(
            server.status_updates[-1]["last_qc_failure_reasons"],
            ["缺少 HDF5 元数据"],
        )
        self.assertEqual(
            server.status_updates[-1]["last_export_error"],
            "LeRobot 环境暂不可用",
        )

    def test_api_status_exposes_vr_sidecar_readiness(self):
        server = FakeServer()
        _update_api_status(
            server,
            WorkerStatus(
                WorkerState.ENV_READY,
                metadata={
                    "session_id": "cli-test",
                    "teleop_mode": "vr",
                    "teleop": {"ready": True, "message": "Pico VR connected"},
                },
            ),
        )

        self.assertTrue(server.status_updates[-1]["teleop_ready"])
        self.assertEqual(server.status_updates[-1]["teleop_mode"], "vr")
        self.assertEqual(server.status_updates[-1]["teleop_message"], "Pico VR connected")

    def test_preflight_constructs_worker_but_does_not_launch_kit(self):
        created = []

        class PreflightWorker(FakeWorker):
            def __init__(self, config):
                super().__init__(config)
                created.append(self)

        result, code = preflight_result(make_config(), worker_factory=PreflightWorker)

        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertIn("尚未启动", result["message"])
        self.assertEqual(created[0].state, WorkerState.PREFLIGHT)

    def test_preflight_failure_is_reported_as_a_blocking_result(self):
        class FailingWorker(FakeWorker):
            def configure(self):
                raise ValueError("相机适配器未实现")

        result, code = preflight_result(make_config(), worker_factory=FailingWorker)

        self.assertEqual(code, 2)
        self.assertFalse(result["passed"])
        self.assertIn("相机适配器未实现", result["message"])

    def test_real_task_preflight_accepts_camera_and_keyboard_without_sidecars(self):
        simulation_root = Path(__file__).resolve().parents[2]
        config = SessionConfig(
            session_id="preflight-only-test",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(physics_hz=120, control_hz=30, render_hz=30),
            sensors=(
                SensorConfig("base_camera", "camera", frequency_hz=30),
                SensorConfig("odom", "localization", frequency_hz=30),
            ),
            simulation_repo_root=str(simulation_root),
            isaac_lab_python="/opt/isaaclab/bin/python",
            lerobot_python="/opt/lerobot/bin/python",
            episode_root="/tmp/mrs-episodes",
            dataset_root="/tmp/mrs-datasets",
            dataset_id="openflex_sim/navigation_to_goal",
            teleop_mode="keyboard",
        )

        result, code = preflight_result(config)

        self.assertEqual(code, 0)
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["metadata"]["preflight"]["teleop"],
            {"mode": "keyboard", "sidecar_required": False},
        )

    def test_real_task_preflight_accepts_existing_vr_ros_sidecar_configuration(self):
        simulation_root = Path(__file__).resolve().parents[2]
        with __import__("tempfile").TemporaryDirectory() as directory:
            root = Path(directory)
            ros_setup = root / "ros" / "setup.bash"
            underlay_setup = root / "openflex" / "setup.bash"
            workspace_root = root / "openflex_ws"
            ros_setup.parent.mkdir(parents=True)
            underlay_setup.parent.mkdir(parents=True)
            workspace_root.mkdir()
            ros_setup.write_text("# ROS setup fixture\n", encoding="utf-8")
            underlay_setup.write_text("# underlay fixture\n", encoding="utf-8")
            config = SessionConfig(
                session_id="vr-preflight-test",
                embodiment_id="openflex",
                scene_id="flat_navigation",
                task_id="navigation_to_goal",
                frequency=FrequencyConfig(physics_hz=120, control_hz=30, render_hz=30),
                sensors=(
                    SensorConfig("base_camera", "camera", frequency_hz=30),
                    SensorConfig("odom", "localization", frequency_hz=30),
                ),
                teleop_mode="vr",
                simulation_repo_root=str(simulation_root),
                metadata={
                    "ros2_runtime": {
                        "ros_setup": str(ros_setup),
                        "underlay_setup": str(underlay_setup),
                        "workspace_root": str(workspace_root),
                        "domain_id": 49,
                    }
                },
            )

            result, code = preflight_result(config)

        self.assertEqual(code, 0, result)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metadata"]["preflight"]["teleop"]["mode"], "vr")
        self.assertEqual(result["metadata"]["preflight"]["teleop"]["ros_domain_id"], 49)

    def test_api_stop_command_is_dispatched_on_worker_thread_and_closes_server(self):
        worker = FakeWorker(make_config())
        server = FakeServer()

        code = serve_worker(worker, server, sleep_fn=lambda _seconds: None)

        self.assertEqual(code, 0)
        self.assertTrue(server.started)
        self.assertTrue(server.closed)
        self.assertEqual(worker.commands, [("stop", {})])
        self.assertEqual(worker.shutdown_calls, 0)
        self.assertTrue(any(update.get("ready") is True for update in server.status_updates))
        self.assertEqual(server.status_updates[-1]["state"], WorkerState.STOPPING.value)

    def test_cli_wires_the_lerobot_export_callback_into_the_worker(self):
        config = make_config()
        callback = object()
        worker = FakeWorker(config)
        server = FakeServer()
        with (
            mock.patch.object(worker_cli, "_load_config", return_value=config),
            mock.patch.object(worker_cli, "create_lerobot_export_callback", return_value=callback),
            mock.patch.object(worker_cli, "IsaacLabWorker", return_value=worker) as worker_factory,
            mock.patch.object(worker_cli, "CaptureControlServer", return_value=server),
            mock.patch.object(worker_cli, "serve_worker", return_value=0),
            mock.patch.object(worker_cli.signal, "signal", return_value=None),
        ):
            result = worker_cli.main(["--config-json", "{}"])

        self.assertEqual(result, 0)
        worker_factory.assert_called_once_with(config, export_callback=callback)


if __name__ == "__main__":
    unittest.main()
