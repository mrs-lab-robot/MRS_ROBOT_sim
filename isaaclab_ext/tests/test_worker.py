"""Tests for IsaacLabWorker state machine and lifecycle.

TDD: Tests written first, implementation follows.
"""

import unittest
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from openflex_isaac_contract.session_config import (
    FrequencyConfig,
    SensorConfig,
    SessionConfig,
    WorkerState,
)

from mrs_robot_lab.recorders.capture_api import CaptureControlServer
from mrs_robot_lab.workers import IsaacLabWorker, WorkerStatus, _IsaacLabRuntime


class TestKeyboardRuntime(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[2]
        self.config = SessionConfig(
            session_id="keyboard-runtime",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("base_camera", "camera", frequency_hz=30.0),
                SensorConfig("odom", "localization", frequency_hz=30.0),
            ),
            simulation_repo_root=str(self.repo_root),
            teleop_mode="keyboard",
        )

    def test_keyboard_preflight_does_not_require_vr_sidecar_configuration(self):
        runtime = _IsaacLabRuntime()

        result = runtime.preflight(self.config)

        self.assertEqual(
            result["teleop"],
            {"mode": "keyboard", "sidecar_required": False},
        )

    def test_keyboard_sidecar_start_is_ready_without_spawning_a_process(self):
        process_factory = MagicMock()
        runtime = _IsaacLabRuntime(process_factory=process_factory)

        runtime.start_sidecars(self.config, object())

        self.assertFalse(process_factory.called)
        self.assertEqual(
            runtime.teleop_state(),
            {"ready": True, "message": "GUI keyboard input channel ready"},
        )

    def test_worker_marks_loopback_keyboard_commands_ready_after_launch(self):
        worker = IsaacLabWorker(self.config, runtime=FakeWorkerRuntime())
        server = CaptureControlServer(port=0)
        try:
            worker.attach_teleop_transport(server)
            worker.configure()
            worker.launch()

            self.assertTrue(server.teleop_state()["ready"])
            command = {
                "kind": "base_twist",
                "values": [0.1, 0.0, 0.0],
                "active": False,
            }
            server.submit_teleop_command(command)
            received = server.take_latest_teleop_command()
            self.assertEqual(
                {key: received[key] for key in command},
                command,
            )
            self.assertIsInstance(received["received_monotonic_ns"], int)
        finally:
            worker.shutdown()
            server.close()


class TestWorkerInitialization(unittest.TestCase):
    """Test worker initialization."""

    def setUp(self):
        """Create minimal valid config."""
        self.config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(
                physics_hz=120.0,
                control_hz=30.0,
                render_hz=30.0,
            ),
        )

    def test_worker_starts_in_idle(self):
        """Worker应该从IDLE状态开始"""
        worker = IsaacLabWorker(self.config)
        self.assertEqual(worker.state, WorkerState.IDLE)

    def test_worker_stores_config(self):
        """Worker应该存储配置"""
        worker = IsaacLabWorker(self.config)
        self.assertEqual(worker.config.session_id, "test_001")
        self.assertEqual(worker.config.embodiment_id, "openflex")

    def test_get_status_returns_snapshot(self):
        """get_status应该返回状态快照"""
        worker = IsaacLabWorker(self.config)
        status = worker.get_status()
        self.assertIsInstance(status, WorkerStatus)
        self.assertEqual(status.state, WorkerState.IDLE)
        self.assertIsInstance(status.message, str)
        self.assertGreaterEqual(status.progress, 0.0)
        self.assertLessEqual(status.progress, 1.0)


class TestWorkerConfiguration(unittest.TestCase):
    """Test worker configuration phase."""

    def setUp(self):
        """Create minimal valid config."""
        self.config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(
                physics_hz=120.0,
                control_hz=30.0,
                render_hz=30.0,
            ),
        )

    def test_configure_validates_and_transitions_to_preflight(self):
        """configure应该验证配置并转到PREFLIGHT"""
        worker = IsaacLabWorker(self.config)
        self.assertEqual(worker.state, WorkerState.IDLE)

        worker.configure()
        self.assertEqual(worker.state, WorkerState.PREFLIGHT)

    def test_configure_rejects_invalid_config(self):
        """configure应该拒绝无效配置"""
        bad_config = SessionConfig(
            session_id="",  # Invalid: empty
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
        )
        worker = IsaacLabWorker(bad_config)

        with self.assertRaises(ValueError):
            worker.configure()

        # Should still be in IDLE
        self.assertEqual(worker.state, WorkerState.IDLE)

    def test_cannot_configure_twice(self):
        """不能重复configure"""
        worker = IsaacLabWorker(self.config)
        worker.configure()
        self.assertEqual(worker.state, WorkerState.PREFLIGHT)

        with self.assertRaises(RuntimeError) as ctx:
            worker.configure()
        self.assertIn("cannot configure", str(ctx.exception).lower())


class TestWorkerStateTransitions(unittest.TestCase):
    """Test worker state machine transitions."""

    def setUp(self):
        """Create minimal valid config."""
        self.config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(
                physics_hz=120.0,
                control_hz=30.0,
                render_hz=30.0,
            ),
        )

    def test_cannot_launch_from_idle(self):
        """从IDLE状态不能直接launch"""
        worker = IsaacLabWorker(self.config)
        self.assertEqual(worker.state, WorkerState.IDLE)

        with self.assertRaises(RuntimeError) as ctx:
            worker.launch()
        self.assertIn("current state=idle", str(ctx.exception).lower())

    def test_can_launch_from_preflight(self):
        """从PREFLIGHT状态可以launch"""
        worker = IsaacLabWorker(self.config)

        # Configure first
        worker.configure()
        self.assertEqual(worker.state, WorkerState.PREFLIGHT)

        # Mock internal methods to avoid NotImplementedError
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()

        worker.launch()
        self.assertEqual(worker.state, WorkerState.ENV_READY)

    def test_cannot_launch_twice(self):
        """不能重复启动Worker"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock internal methods
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()

        # First launch should succeed
        worker.launch()
        self.assertEqual(worker.state, WorkerState.ENV_READY)

        # Second launch should fail
        with self.assertRaises(RuntimeError) as ctx:
            worker.launch()
        self.assertIn("current state=env_ready", str(ctx.exception).lower())

    def test_illegal_internal_state_transition_is_rejected(self):
        worker = IsaacLabWorker(self.config)

        with self.assertRaises(RuntimeError) as ctx:
            worker._transition_to(WorkerState.ENV_READY)

        self.assertIn("invalid worker state transition", str(ctx.exception).lower())

    def test_launch_failure_exposes_failure_reason_in_status(self):
        worker = IsaacLabWorker(self.config)
        worker.configure()
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock(side_effect=RuntimeError("missing task adapter"))

        with self.assertRaises(RuntimeError):
            worker.launch()

        status = worker.get_status()
        self.assertEqual(status.state, WorkerState.FAILED)
        self.assertIn("missing task adapter", status.metadata["failure_reason"])
        self.assertIn("missing task adapter", status.message)

    def test_cannot_record_before_ready(self):
        """ENV_READY之前不能开始录制"""
        worker = IsaacLabWorker(self.config)
        worker.configure()
        self.assertEqual(worker.state, WorkerState.PREFLIGHT)

        with self.assertRaises(RuntimeError) as ctx:
            worker.start_recording()
        self.assertIn("cannot start recording", str(ctx.exception).lower())

    def test_can_record_after_ready(self):
        """ENV_READY之后可以开始录制"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock launch sequence
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()

        worker.launch()
        self.assertEqual(worker.state, WorkerState.ENV_READY)

        worker.start_recording()
        self.assertEqual(worker.state, WorkerState.RECORDING)

    def test_stop_recording_returns_to_ready(self):
        """停止录制应该返回ENV_READY"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock and launch
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()
        worker.launch()

        worker.start_recording()
        self.assertEqual(worker.state, WorkerState.RECORDING)

        worker.stop_recording()
        self.assertEqual(worker.state, WorkerState.ENV_READY)

    def test_discard_episode_returns_to_ready(self):
        """丢弃Episode应该返回ENV_READY"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock and launch
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()
        worker.launch()

        worker.start_recording()
        worker.discard_episode()
        self.assertEqual(worker.state, WorkerState.ENV_READY)

    def test_cannot_save_when_not_recording(self):
        """非RECORDING状态不能保存"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock and launch
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()
        worker.launch()

        self.assertEqual(worker.state, WorkerState.ENV_READY)

        with self.assertRaises(RuntimeError) as ctx:
            worker.save_episode()
        self.assertIn("cannot save", str(ctx.exception).lower())


class TestWorkerLaunchSequence(unittest.TestCase):
    """Test worker launch state sequence."""

    def setUp(self):
        """Create minimal valid config."""
        self.config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(
                physics_hz=120.0,
                control_hz=30.0,
                render_hz=30.0,
            ),
        )

    def test_launch_sequence_calls_all_stages(self):
        """launch应该按顺序调用所有阶段"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock all internal methods
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()

        worker.launch()

        # Verify all stages were called
        worker._launch_app.assert_called_once()
        worker._build_environment.assert_called_once()
        worker._reset_environment.assert_called_once()
        worker._warmup.assert_called_once()
        worker._start_sidecars.assert_called_once()
        worker._health_check.assert_called_once()

        # Final state should be ENV_READY
        self.assertEqual(worker.state, WorkerState.ENV_READY)

    def test_launch_failure_transitions_to_failed(self):
        """启动失败应该转到FAILED状态"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock methods, one fails
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock(side_effect=RuntimeError("Build failed"))

        with self.assertRaises(RuntimeError):
            worker.launch()

        # State should be FAILED
        self.assertEqual(worker.state, WorkerState.FAILED)


class TestWorkerShutdown(unittest.TestCase):
    """Test worker shutdown."""

    def setUp(self):
        """Create minimal valid config."""
        self.config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(
                physics_hz=120.0,
                control_hz=30.0,
                render_hz=30.0,
            ),
        )

    def test_shutdown_from_idle(self):
        """可以从IDLE状态关闭"""
        worker = IsaacLabWorker(self.config)
        self.assertEqual(worker.state, WorkerState.IDLE)

        worker.shutdown()
        self.assertEqual(worker.state, WorkerState.STOPPING)


class FakeWorkerRuntime:
    """Small runtime double that records the single Kit lifecycle."""

    def __init__(self):
        self.calls = []
        self.environment = object()

    def launch_app(self, config):
        self.calls.append(("launch_app", config.session_id))
        return object()

    def build_environment(self, config, app):
        self.calls.append(("build_environment", config.task_id))
        return self.environment

    def reset_environment(self, environment, seed):
        self.calls.append(("reset_environment", seed))

    def warmup(self, environment, steps):
        self.calls.append(("warmup", steps))

    def start_sidecars(self, config, environment):
        self.calls.append(("start_sidecars", config.teleop_mode))

    def health_check(self, config, environment):
        self.calls.append(("health_check", config.config_hash))

    def close(self, environment, app):
        self.calls.append(("close", environment is self.environment))


class FakeEpisodeController:
    def __init__(self):
        self.num_steps = 0
        self.calls = []
        self.metadata = {}

    def start(self, metadata):
        self.metadata = dict(metadata)
        self.calls.append(("start", dict(metadata)))

    def annotate(self, metadata):
        self.metadata.update(metadata)
        self.calls.append(("annotate", dict(metadata)))

    def append(self, **sample):
        self.calls.append(("append", sample))
        self.num_steps += 1

    def save(self, path):
        self.calls.append(("save", Path(path)))
        return Path(path)

    def discard(self):
        self.calls.append(("discard",))


class FakeTeleopTransport:
    def __init__(self, command, *, ready=True):
        self.command = command
        self.states = []
        self.ready = ready

    def take_latest_teleop_command(self):
        command, self.command = self.command, None
        return command

    def update_teleop_state(self, **state):
        self.states.append(state)

    def teleop_state(self):
        return {
            "ready": self.ready,
            "message": "VR control connected" if self.ready else "VR control disconnected",
            "joint_state": {},
        }


class TestWorkerTeleopPump(unittest.TestCase):
    def test_episode_start_uses_runtime_sidecar_health_over_stale_api_cache(self):
        config = SessionConfig(
            session_id="vr-sidecar-exited",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            teleop_mode="vr",
        )

        class DeadSidecarRuntime(FakeWorkerRuntime):
            def teleop_state(self):
                return {"ready": False, "message": "VR ROS 2 sidecar exited with code 7"}

        worker = IsaacLabWorker(config, runtime=DeadSidecarRuntime())
        worker.configure()
        worker.launch()
        worker.attach_teleop_transport(FakeTeleopTransport(None, ready=True))

        with self.assertRaisesRegex(RuntimeError, "sidecar exited with code 7"):
            worker.start_recording()

    def test_episode_start_is_blocked_when_vr_sidecar_is_not_ready(self):
        config = SessionConfig(
            session_id="vr-not-ready",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            teleop_mode="vr",
        )
        worker = IsaacLabWorker(config, runtime=FakeWorkerRuntime())
        worker.configure()
        worker.launch()
        worker.attach_teleop_transport(FakeTeleopTransport(None, ready=False))

        with self.assertRaisesRegex(RuntimeError, "VR control disconnected"):
            worker.start_recording()

    def test_latest_vr_command_steps_the_environment_and_is_recorded_on_kit_thread(self):
        config = SessionConfig(
            session_id="vr-pump",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            dataset_id="openflex_sim/navigation_vr",
        )
        recorder = FakeEpisodeController()

        class TeleopRuntime(FakeWorkerRuntime):
            def step_once(self, _config, _environment, command, *, step_index, recording):
                self.calls.append(("step_once", command, step_index, recording))
                return {
                    "sample": {"step_index": step_index, "action": [0.0]},
                    "joint_state": {"names": ["joint_a"], "position": [0.25], "velocity": [0.0]},
                }

        runtime = TeleopRuntime()
        worker = IsaacLabWorker(
            config,
            runtime=runtime,
            recorder_factory=lambda: recorder,
        )
        worker.configure()
        worker.launch()
        transport = FakeTeleopTransport(
            {"kind": "base_twist", "values": [0.2, 0.0, 0.0]}
        )
        worker.attach_teleop_transport(transport)
        self.assertTrue(worker.get_status().metadata["teleop"]["ready"])
        worker.start_recording()
        worker.pump_once()

        self.assertEqual(
            [call for call in runtime.calls if call[0] == "step_once"],
            [("step_once", {"kind": "base_twist", "values": [0.2, 0.0, 0.0]}, 0, True)],
        )
        append = next(call for call in recorder.calls if call[0] == "append")
        self.assertEqual(append[1], {"step_index": 0, "action": [0.0]})
        self.assertEqual(transport.states[-1]["joint_state"]["position"], [0.25])

    def test_vr_disconnect_during_recording_quarantines_episode(self):
        config = SessionConfig(
            session_id="vr-disconnect",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            teleop_mode="vr",
        )
        recorder = FakeEpisodeController()

        class TeleopRuntime(FakeWorkerRuntime):
            def step_once(self, _config, _environment, command, *, step_index, recording):
                return {
                    "sample": {"step_index": step_index, "action": [0.0]},
                    "joint_state": {},
                }

        worker = IsaacLabWorker(
            config,
            runtime=TeleopRuntime(),
            recorder_factory=lambda: recorder,
        )
        worker.configure()
        worker.launch()
        transport = FakeTeleopTransport(None, ready=True)
        worker.attach_teleop_transport(transport)
        worker.start_recording({"episode_id": "ep-vr-disconnect"})

        transport.ready = False
        worker.pump_once()

        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "episode.hdf5"
            original.write_bytes(b"episode")
            quarantined = Path(directory) / "quarantine" / original.name
            with patch.object(worker, "_save_hdf5", return_value=original), patch.object(
                worker, "_quality_check", return_value=True
            ), patch.object(worker, "_quarantine_hdf5", return_value=quarantined):
                result = worker.save_episode()

        self.assertEqual(result, quarantined)
        self.assertEqual(worker.state, WorkerState.ENV_READY)
        self.assertEqual(
            worker.get_status().metadata["last_qc_failure_reasons"],
            ["VR 遥操作在 Episode 录制期间断开：VR control disconnected"],
        )
        self.assertEqual(
            recorder.metadata["recording_faults"],
            ["VR 遥操作在 Episode 录制期间断开：VR control disconnected"],
        )


class TestWorkerInjectedRuntime(unittest.TestCase):
    def setUp(self):
        self.config = SessionConfig(
            session_id="test_runtime",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            dataset_id="openflex_sim/navigation_v1",
        )

    def test_launch_uses_one_runtime_in_the_documented_order(self):
        runtime = FakeWorkerRuntime()
        worker = IsaacLabWorker(self.config, runtime=runtime, warmup_steps=3)

        worker.configure()
        worker.launch()

        self.assertEqual(
            [call[0] for call in runtime.calls],
            [
                "launch_app",
                "build_environment",
                "reset_environment",
                "warmup",
                "start_sidecars",
                "health_check",
            ],
        )
        self.assertEqual(runtime.calls[3], ("warmup", 3))
        self.assertEqual(worker.get_status().metadata["environment_id"], "navigation_to_goal")

    def test_recording_forwards_metadata_samples_and_save_to_episode_controller(self):
        runtime = FakeWorkerRuntime()
        recorder = FakeEpisodeController()
        worker = IsaacLabWorker(
            self.config,
            runtime=runtime,
            recorder_factory=lambda: recorder,
        )
        worker.configure()
        worker.launch()
        worker.start_recording({"episode_id": "ep-1"})
        worker.append_sample(step_index=0, action=[0.0])

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, "_quality_check", return_value=True), patch.object(
                worker, "_export_lerobot"
            ):
                output = worker.save_episode(Path(directory) / "episode.h5")

        self.assertEqual(output.name, "episode.h5")
        self.assertEqual([call[0] for call in recorder.calls], ["start", "append", "save"])
        self.assertEqual(recorder.calls[0][1]["dataset_id"], "openflex_sim/navigation_v1")
        self.assertEqual(worker.state, WorkerState.ENV_READY)

    def test_non_vr_worker_episode_passes_qc_and_reaches_real_lerobot_exporter(self):
        import os
        from io import BytesIO

        from PIL import Image

        from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
        from mrs_robot_lab.recorders.episode_controller import EpisodeController
        from mrs_robot_lab.worker_cli import create_lerobot_export_callback

        lerobot_python = os.environ.get("MRS_ROBOT_TEST_LEROBOT_PYTHON", "").strip()
        if not lerobot_python or not Path(lerobot_python).is_file():
            self.skipTest("MRS_ROBOT_TEST_LEROBOT_PYTHON is not configured")

        image = BytesIO()
        Image.new("RGB", (64, 64), (20, 40, 60)).save(image, format="JPEG")
        simulation_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = SessionConfig(
                session_id="non-vr-worker-export",
                embodiment_id="openflex",
                scene_id="flat_navigation",
                task_id="navigation_to_goal",
                frequency=FrequencyConfig(120.0, 30.0, 30.0, output_fps=30.0),
                sensors=(SensorConfig("base_camera", "camera", frequency_hz=30.0),),
                episode_root=str(root / "episodes"),
                dataset_root=str(root / "datasets"),
                dataset_id="openflex_sim/non_vr_worker_smoke",
                simulation_repo_root=str(simulation_root),
                lerobot_python=lerobot_python,
            )
            worker = IsaacLabWorker(
                config,
                runtime=FakeWorkerRuntime(),
                recorder_factory=EpisodeController,
                export_callback=create_lerobot_export_callback(),
            )
            worker.configure()
            worker.launch()
            worker.start_recording({"input_source": "keyboard"})
            worker.append_sample(
                sim_time_ns=0,
                step_index=0,
                command_seq=1,
                operator_command=[0.0] * ACTION_DIMENSION,
                applied_target=[0.0] * ACTION_DIMENSION,
                joint_position=[0.0] * len(JOINT_STATE_NAMES),
                joint_velocity=[0.0] * len(JOINT_STATE_NAMES),
                next_joint_position=[0.01] * len(JOINT_STATE_NAMES),
                camera_frames={"base_d435": image.getvalue()},
            )

            saved_path = worker.save_episode()
            dataset_path = root / "datasets" / "openflex_sim" / "non_vr_worker_smoke"

            self.assertTrue(saved_path.is_file())
            self.assertTrue(dataset_path.is_dir())
            self.assertEqual(worker.state, WorkerState.ENV_READY)
            self.assertEqual(worker.get_status().metadata["last_export_error"], "")
            worker.shutdown()

    def test_quality_failed_episode_is_moved_to_quarantine_and_reported(self):
        runtime = FakeWorkerRuntime()
        recorder = FakeEpisodeController()
        worker = IsaacLabWorker(
            self.config,
            runtime=runtime,
            recorder_factory=lambda: recorder,
        )
        worker.configure()
        worker.launch()
        worker.start_recording({"episode_id": "ep-qc-failed"})
        worker.append_sample(step_index=0, action=[0.0])

        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "episode.hdf5"
            original.write_bytes(b"not trainable")
            with patch.object(worker, "_save_hdf5", return_value=original), patch.object(
                worker, "_quality_check", return_value=False
            ):
                quarantined = worker.save_episode()

            self.assertFalse(original.exists())
            self.assertTrue(quarantined.is_file())
            self.assertEqual(quarantined.parent.name, "quarantine")
            self.assertEqual(
                worker.get_status().metadata["last_quarantined_path"],
                str(quarantined),
            )
            self.assertEqual(
                worker.get_status().metadata["last_qc_failure_reasons"],
                ["Episode 质量检查未通过"],
            )
        self.assertEqual(worker.state, WorkerState.ENV_READY)

    def test_quarantine_does_not_overwrite_an_existing_rejected_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quarantine = root / "quarantine"
            quarantine.mkdir()
            existing = quarantine / "episode.hdf5"
            existing.write_bytes(b"earlier rejected episode")
            source = root / "episode.hdf5"
            source.write_bytes(b"new rejected episode")

            quarantined = IsaacLabWorker._quarantine_hdf5(source)

            self.assertNotEqual(quarantined, existing)
            self.assertEqual(existing.read_bytes(), b"earlier rejected episode")
            self.assertEqual(quarantined.read_bytes(), b"new rejected episode")
            self.assertFalse(source.exists())

    def test_quality_failure_records_actionable_reasons(self):
        worker = IsaacLabWorker(self.config)
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "incomplete.hdf5"
            import h5py

            with h5py.File(episode, "w"):
                pass

            self.assertFalse(worker._quality_check(episode))
            reasons = worker.get_status().metadata["last_qc_failure_reasons"]

        self.assertTrue(reasons)
        self.assertTrue(any("元数据" in reason or "sim_time_ns" in reason for reason in reasons))

    def test_quality_check_requires_every_enabled_camera_stream(self):
        import h5py
        import numpy as np
        from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES

        camera_ids = (
            "base_camera",
            "head_camera",
            "left_wrist_camera",
            "right_wrist_camera",
        )
        config = SessionConfig(
            session_id="four-camera-qc",
            embodiment_id="openflex",
            scene_id="dual_arm_box_tabletop",
            task_id="dual_arm_box_transport",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=tuple(SensorConfig(sensor_id, "camera", frequency_hz=30.0) for sensor_id in camera_ids),
        )
        worker = IsaacLabWorker(config)
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "one-camera.hdf5"
            with h5py.File(episode, "w") as dataset:
                dataset.attrs["metadata_json"] = (
                    '{"episode_id":"ep-1","config_hash":"cfg",'
                    '"contract_hash":"contract","random_seed":1,'
                    '"input_source":"keyboard","quality_status":"passed",'
                    '"embodiment_id":"openflex","scene_id":"dual_arm_box_tabletop",'
                    '"task_id":"dual_arm_box_transport",'
                    '"dataset_id":"openflex_sim/dual_arm_box_transport",'
                    '"teleop_mode":"keyboard"}'
                )
                dataset.attrs["format"] = "mrs_robot_capture_v1"
                dataset.create_dataset("sim_time_ns", data=[1])
                dataset.create_dataset("step_index", data=[0])
                dataset.create_dataset("command_seq", data=[1])
                action = np.zeros((1, ACTION_DIMENSION), dtype=np.float32)
                joint_state = np.zeros((1, len(JOINT_STATE_NAMES)), dtype=np.float32)
                dataset.create_dataset("operator_command", data=action)
                dataset.create_dataset("applied_target", data=action)
                dataset.create_dataset("action", data=action)
                dataset.create_dataset("joint_position", data=joint_state)
                dataset.create_dataset("joint_velocity", data=joint_state)
                dataset.create_dataset("next_joint_position", data=joint_state)
                cameras = dataset.create_group("camera_frames")
                camera = cameras.create_group("base_d435")
                camera.create_dataset("sim_time_ns", data=[1])
                camera.create_dataset("jpeg", data=[b"jpeg-frame"])

            self.assertFalse(worker._quality_check(episode))

        reasons = worker.get_status().metadata["last_qc_failure_reasons"]
        self.assertTrue(any("head_d435" in reason for reason in reasons))
        self.assertTrue(any("left_wrist_d405" in reason for reason in reasons))
        self.assertTrue(any("right_wrist_d405" in reason for reason in reasons))
        self.assertFalse(any("格式标识" in reason for reason in reasons))

    def test_simulated_vr_episode_is_not_approved_for_training_export(self):
        import h5py
        import numpy as np
        from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES

        worker = IsaacLabWorker(self.config)
        with tempfile.TemporaryDirectory() as directory:
            episode = Path(directory) / "simulated-vr.hdf5"
            with h5py.File(episode, "w") as dataset:
                dataset.attrs["format"] = "mrs_robot_capture_v1"
                dataset.attrs["metadata_json"] = (
                    '{"episode_id":"ep-sim","config_hash":"cfg",'
                    '"contract_hash":"contract","random_seed":1,'
                    '"input_source":"simulated_vr","quality_status":"passed",'
                    '"embodiment_id":"openflex","scene_id":"flat_navigation",'
                    '"task_id":"navigation_to_goal",'
                    '"dataset_id":"openflex_sim/navigation_to_goal",'
                    '"teleop_mode":"vr"}'
                )
                dataset.create_dataset("sim_time_ns", data=[1])
                dataset.create_dataset("step_index", data=[0])
                dataset.create_dataset("command_seq", data=[0])
                action = np.zeros((1, ACTION_DIMENSION), dtype=np.float32)
                joint = np.zeros((1, len(JOINT_STATE_NAMES)), dtype=np.float32)
                for name, values in (
                    ("operator_command", action),
                    ("applied_target", action),
                    ("action", action),
                    ("joint_position", joint),
                    ("joint_velocity", joint),
                    ("next_joint_position", joint),
                ):
                    dataset.create_dataset(name, data=values)
                camera = dataset.create_group("camera_frames/base_d435")
                camera.create_dataset("sim_time_ns", data=[1])
                camera.create_dataset("jpeg", data=[b"jpeg-frame"])

            self.assertFalse(worker._quality_check(episode))

        reasons = worker.get_status().metadata["last_qc_failure_reasons"]
        self.assertTrue(any("诊断输入" in reason for reason in reasons))

    def test_export_failure_keeps_hdf5_and_can_be_retried_without_re_recording(self):
        recorder = FakeEpisodeController()
        worker = IsaacLabWorker(
            self.config,
            runtime=FakeWorkerRuntime(),
            recorder_factory=lambda: recorder,
        )
        worker.configure()
        worker.launch()
        worker.start_recording({"episode_id": "ep-export-retry"})
        worker.append_sample(step_index=0, action=[0.0])

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(worker, "_quality_check", return_value=True), patch.object(
                worker,
                "_export_lerobot",
                side_effect=[RuntimeError("LeRobot 环境暂不可用"), None],
            ) as export:
                saved_path = worker.save_episode(Path(directory) / "episode.hdf5")

                self.assertEqual(worker.state, WorkerState.ENV_READY)
                self.assertEqual(worker.get_status().metadata["last_saved_path"], str(saved_path))
                self.assertIn("LeRobot 环境暂不可用", worker.get_status().metadata["last_export_error"])

                retried_path = worker.retry_export()

        self.assertEqual(retried_path, saved_path)
        self.assertEqual(export.call_count, 2)
        self.assertEqual(worker.state, WorkerState.ENV_READY)
        self.assertEqual(worker.get_status().metadata["last_export_error"], "")

    def test_shutdown_closes_the_single_runtime(self):
        runtime = FakeWorkerRuntime()
        worker = IsaacLabWorker(self.config, runtime=runtime)
        worker.configure()
        worker.launch()
        worker.shutdown()

        self.assertEqual(runtime.calls[-1], ("close", True))

    def test_command_dispatch_keeps_all_environment_operations_on_the_worker(self):
        runtime = FakeWorkerRuntime()
        worker = IsaacLabWorker(self.config, runtime=runtime)
        worker.configure()
        worker.launch()

        worker.handle_command("start", {"metadata": {"episode_id": "cmd-episode"}})
        self.assertEqual(worker.state, WorkerState.RECORDING)
        worker.handle_command("discard")
        worker.handle_command("reset")

        self.assertEqual(worker.state, WorkerState.ENV_READY)
        self.assertEqual(
            [call[0] for call in runtime.calls].count("reset_environment"),
            2,
        )
        self.assertEqual([call[0] for call in runtime.calls].count("health_check"), 2)

    def test_shutdown_from_preflight(self):
        """可以从PREFLIGHT状态关闭"""
        worker = IsaacLabWorker(self.config)
        worker.configure()
        self.assertEqual(worker.state, WorkerState.PREFLIGHT)

        worker.shutdown()
        self.assertEqual(worker.state, WorkerState.STOPPING)

    def test_shutdown_after_launch(self):
        """启动后可以关闭"""
        worker = IsaacLabWorker(self.config)
        worker.configure()

        # Mock launch
        worker._launch_app = MagicMock()
        worker._build_environment = MagicMock()
        worker._reset_environment = MagicMock()
        worker._warmup = MagicMock()
        worker._start_sidecars = MagicMock()
        worker._health_check = MagicMock()
        worker.launch()

        worker.shutdown()
        self.assertEqual(worker.state, WorkerState.STOPPING)


if __name__ == "__main__":
    unittest.main()
