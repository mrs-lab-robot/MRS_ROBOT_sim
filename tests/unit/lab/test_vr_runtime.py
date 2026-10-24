from __future__ import annotations

import tempfile
import signal
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from unittest.mock import patch

from openflex_isaac_contract.session_config import FrequencyConfig, SensorConfig, SessionConfig
from mrs_robot_lab.workers import _capture_camera_frames, _extract_joint_state
from mrs_robot_lab.workers import _IsaacLabRuntime
from mrs_robot_lab.assets.robot_interface import JOINT_STATE_NAMES


class _OwnedProcess:
    def __init__(self, return_code: int | None = None) -> None:
        self.return_code = return_code
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        return self.return_code

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        if self.return_code is None:
            self.return_code = 0
        return self.return_code


class _FakeEnvironment:
    def __init__(self, *, arm_task: bool = False) -> None:
        self.device = "cpu"
        self.scene = SimpleNamespace(sensors={})
        self._robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_names=list(JOINT_STATE_NAMES),
                joint_pos=torch.zeros((1, len(JOINT_STATE_NAMES))),
                joint_vel=torch.zeros((1, len(JOINT_STATE_NAMES))),
            )
        )
        if arm_task:
            self._joint_targets = torch.zeros((1, 16))
        self.actions = []

    def step(self, action):
        self.actions.append(action.detach().cpu().clone())
        self._robot.data.joint_pos.add_(0.01)
        return None


class VrRuntimeSidecarTest(unittest.TestCase):
    def _config(self, root: Path) -> SessionConfig:
        ros_setup = root / "ros setup.bash"
        underlay = root / "openflex setup.bash"
        urdf = root / "robot control.urdf"
        bridge = root / "isaaclab_ext/scripts/worker_ros2_bridge.py"
        for path in (ros_setup, underlay, urdf, bridge):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test fixture\n", encoding="utf-8")
        return SessionConfig(
            session_id="vr-runtime",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            teleop_mode="vr",
            simulation_repo_root=str(root),
            metadata={
                "ros2_runtime": {
                    "ros_setup": str(ros_setup),
                    "underlay_setup": str(underlay),
                    "workspace_root": str(root),
                    "domain_id": 49,
                }
            },
        )

    def test_vr_sidecar_starts_only_after_worker_api_reports_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            child = _OwnedProcess()
            launched: list[tuple[list[str], dict]] = []
            api_calls: list[tuple[str, dict | None]] = []

            def process_factory(command, **kwargs):
                launched.append((command, kwargs))
                return child

            def api_request(url, payload=None, timeout=0.4):
                api_calls.append((url, payload))
                return {"ready": True, "message": "VR ROS 2 control nodes are ready", "joint_state": {}}

            runtime = _IsaacLabRuntime(
                process_factory=process_factory,
                api_request=api_request,
                sleep_fn=lambda _seconds: None,
            )
            runtime.configure_teleop_api("http://127.0.0.1:24105")
            runtime.start_sidecars(config, object())

        self.assertEqual(len(launched), 1)
        self.assertEqual(launched[0][0][:2], ["bash", "-c"])
        self.assertTrue(launched[0][1]["start_new_session"])
        self.assertIn("vr_teleop.launch.py", launched[0][0][2])
        self.assertTrue(any(url.endswith("/api/v1/teleop/state") for url, _ in api_calls))

    def test_sidecar_process_exit_is_reported_instead_of_marking_environment_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            runtime = _IsaacLabRuntime(
                process_factory=lambda *_args, **_kwargs: _OwnedProcess(return_code=3),
                api_request=lambda *_args, **_kwargs: {"ready": False, "message": "not ready"},
                sleep_fn=lambda _seconds: None,
            )
            runtime.configure_teleop_api("http://127.0.0.1:24105")

            with self.assertRaisesRegex(RuntimeError, "sidecar exited with code 3"):
                runtime.start_sidecars(config, object())

    def test_teleop_state_reports_owned_sidecar_exit_after_startup(self) -> None:
        process = _OwnedProcess(return_code=7)
        runtime = _IsaacLabRuntime(
            api_request=lambda *_args, **_kwargs: {"ready": True},
        )
        runtime._control_api_url = "http://127.0.0.1:8765"
        runtime._sidecar_process = process
        runtime._sidecar_ready = True
        runtime._sidecar_message = "Pico connected"

        state = runtime.teleop_state()

        self.assertFalse(state["ready"])
        self.assertIn("exited with code 7", state["message"])

    def test_navigation_command_is_clipped_and_recorded_as_physical_input_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            environment = _FakeEnvironment()
            runtime = _IsaacLabRuntime()
            command = {"kind": "base_twist", "values": [0.4, -0.4, 0.58], "source_seq": 8}
            result = runtime.step_once(
                config, environment, command, step_index=1, recording=True
            )

        self.assertEqual(environment.actions[0].shape, (1, 3))
        for actual, expected in zip(
            environment.actions[0][0].tolist(), [0.5, -0.5, 0.25], strict=True
        ):
            self.assertAlmostEqual(actual, expected, places=5)
        sample = result["sample"]
        self.assertEqual(sample["operator_command"][:3], [0.4, -0.4, 0.58])
        self.assertEqual(sample["applied_target"][:3], [0.4, -0.4, 0.58])
        self.assertEqual(sample["sim_time_ns"], 33_333_333)
        self.assertEqual(sample["joint_position"], [0.0] * len(JOINT_STATE_NAMES))
        self.assertTrue(
            all(abs(value - 0.01) < 1e-6 for value in sample["next_joint_position"])
        )
        self.assertEqual(sample["raw_command"], command)

    def test_dual_arm_command_holds_an_uncommanded_arm_across_worker_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            config = SessionConfig(
                session_id=config.session_id,
                embodiment_id=config.embodiment_id,
                scene_id="dual_arm_box_tabletop",
                task_id="dual_arm_box_transport",
                frequency=config.frequency,
                teleop_mode="vr",
            )
            environment = _FakeEnvironment(arm_task=True)
            runtime = _IsaacLabRuntime()
            first = runtime.step_once(
                config,
                environment,
                {"kind": "dual_arm", "left": [0.2] * 8, "right": [-0.2] * 8},
                step_index=0,
                recording=True,
            )
            held_left = environment.actions[-1][0, :7].tolist()
            second = runtime.step_once(
                config,
                environment,
                {"kind": "dual_arm", "left": None, "right": [0.1] * 8},
                step_index=1,
                recording=False,
            )

        self.assertEqual(first["joint_state"]["names"], list(JOINT_STATE_NAMES))
        first_sample = first["sample"]
        self.assertEqual(first_sample["operator_command"][3:10], [0.2] * 7)
        self.assertEqual(first_sample["operator_command"][10:17], [-0.2] * 7)
        self.assertEqual(first_sample["operator_command"][20:22], [0.2, -0.2])
        self.assertAlmostEqual(first_sample["applied_target"][20], 0.044)
        self.assertAlmostEqual(first_sample["applied_target"][21], 0.0)
        self.assertEqual(environment.actions[-1][0, :7].tolist(), held_left)
        self.assertEqual(environment.actions[-1].shape, (1, 16))

    def test_joint_feedback_is_reordered_to_the_contract_even_if_usd_order_differs(self) -> None:
        environment = _FakeEnvironment()
        environment._robot.data.joint_names = list(reversed(JOINT_STATE_NAMES))
        environment._robot.data.joint_pos = torch.arange(len(JOINT_STATE_NAMES)).flip(0).reshape(1, -1)
        environment._robot.data.joint_vel = torch.arange(len(JOINT_STATE_NAMES)).flip(0).reshape(1, -1)

        state = _extract_joint_state(environment, JOINT_STATE_NAMES)

        self.assertEqual(state["names"], list(JOINT_STATE_NAMES))
        self.assertEqual(state["position"], [float(index) for index in range(len(JOINT_STATE_NAMES))])
        self.assertEqual(state["velocity"], [float(index) for index in range(len(JOINT_STATE_NAMES))])

    def test_camera_capture_obeys_its_own_rate_and_encodes_rgb_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            config = replace(
                config,
                sensors=(SensorConfig("base_camera", "camera", frequency_hz=15.0),),
            )
            environment = _FakeEnvironment()
            environment.scene.sensors["base_d435"] = SimpleNamespace(
                data=SimpleNamespace(
                    output={"rgb": torch.full((1, 2, 2, 3), 255, dtype=torch.uint8)}
                )
            )

            first = _capture_camera_frames(config, environment, 0)
            skipped = _capture_camera_frames(config, environment, 1)

        self.assertEqual(set(first), {"base_d435"})
        self.assertTrue(first["base_d435"].startswith(b"\xff\xd8"))
        self.assertEqual(skipped, {})

    def test_environment_reset_discards_cached_vr_arm_targets(self) -> None:
        environment = SimpleNamespace(reset=lambda seed: {"seed": seed})
        runtime = _IsaacLabRuntime()
        runtime._last_arm_action = (0.5,) * 16

        result = runtime.reset_environment(environment, 17)

        self.assertEqual(result, {"seed": 17})
        self.assertIsNone(runtime._last_arm_action)

    def test_shutdown_interrupts_only_the_worker_owned_vr_process_group(self) -> None:
        runtime = _IsaacLabRuntime()
        child = _OwnedProcess()
        child.pid = 43210
        runtime._sidecar_process = child

        with patch("mrs_robot_lab.workers.os.killpg") as kill_group:
            runtime.close(None, None)

        kill_group.assert_called_once_with(43210, signal.SIGINT)
        self.assertEqual(child.wait_calls, [8.0])
        self.assertIsNone(runtime._sidecar_process)


if __name__ == "__main__":
    unittest.main()
