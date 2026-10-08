"""Behavior tests for bridging Isaac Lab's HDF5 and the target LeRobot Python."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from openflex_isaac_contract.session_config import FrequencyConfig, SessionConfig
from mrs_robot_lab.worker_cli import create_lerobot_export_callback


SIM_ROOT = Path("/opt/mrs-sim")


def make_config() -> SessionConfig:
    return SessionConfig(
        session_id="export-test",
        embodiment_id="openflex",
        scene_id="flat_navigation",
        task_id="navigation_to_goal",
        frequency=FrequencyConfig(physics_hz=120, control_hz=30, render_hz=30),
        simulation_repo_root=str(SIM_ROOT),
        lerobot_python="/opt/lerobot/bin/python",
        episode_root="/var/tmp/mrs-episodes",
        dataset_root="/var/tmp/mrs-datasets",
        dataset_id="openflex_sim/navigation_to_goal",
    )


class LeRobotWorkerExportTest(unittest.TestCase):
    def test_worker_prepares_episode_and_invokes_selected_lerobot_python(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "episode.hdf5"
            source.touch()
            prepared_directories = []
            subprocess_calls = []

            def prepare_episode(path, prepared_directory, *, fps, task, source_reference):
                prepared = Path(prepared_directory)
                prepared.mkdir()
                (prepared / "manifest.json").write_text("{}", encoding="utf-8")
                prepared_directories.append(prepared)
                self.assertEqual(Path(path), source)
                self.assertEqual(fps, 30)
                self.assertEqual(task, "navigation_to_goal")
                self.assertEqual(source_reference, "episode.hdf5")
                return prepared

            def run_subprocess(command, **kwargs):
                subprocess_calls.append((command, kwargs))
                return mock.Mock(returncode=0, stdout='{"dataset_path":"ok"}', stderr="")

            callback = create_lerobot_export_callback(
                subprocess_runner=run_subprocess,
                prepare_episode=prepare_episode,
            )
            with mock.patch.dict(
                os.environ,
                {"PYTHONPATH": "/opt/isaaclab/site-packages:/tmp/other-python"},
                clear=False,
            ):
                callback(source, make_config(), {"episode_id": "episode-42"})

        self.assertEqual(len(subprocess_calls), 1)
        command, options = subprocess_calls[0]
        self.assertEqual(
            command,
            [
                "/opt/lerobot/bin/python",
                "-m",
                "mrs_robot_lab.recorders.lerobot_export_cli",
                "--prepared-directory",
                str(prepared_directories[0]),
                "--dataset-root",
                "/var/tmp/mrs-datasets",
                "--repo-id",
                "openflex_sim/navigation_to_goal",
                "--fps",
                "30",
                "--task",
                "navigation_to_goal",
            ],
        )
        self.assertFalse(prepared_directories[0].exists())
        self.assertEqual(options["timeout"], 1200)
        self.assertTrue(options["capture_output"])
        self.assertFalse(options["check"])
        self.assertEqual(options["env"]["MRS_ROBOT_SIM_ROOT"], str(SIM_ROOT))
        pythonpath = options["env"]["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(
            pythonpath,
            [
                str(SIM_ROOT / "isaaclab_ext" / "src"),
                str(
                    SIM_ROOT
                    / "sim_runtime"
                    / "ros2"
                    / "openflex_isaac_sim"
                    / "openflex_isaac_contract"
                ),
            ],
        )

    def test_worker_keeps_export_error_context_when_lerobot_process_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "episode.hdf5"
            source.touch()

            def prepare_episode(_path, prepared_directory, *, fps, task, source_reference):
                prepared = Path(prepared_directory)
                prepared.mkdir()
                return prepared

            def run_subprocess(_command, **_kwargs):
                return mock.Mock(returncode=17, stdout="", stderr="dataset schema mismatch")

            callback = create_lerobot_export_callback(
                subprocess_runner=run_subprocess,
                prepare_episode=prepare_episode,
            )

            with self.assertRaisesRegex(RuntimeError, "dataset schema mismatch"):
                callback(source, make_config(), {"episode_id": "episode-42"})

    def test_worker_turns_lerobot_timeout_into_retryable_export_error(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "episode.hdf5"
            source.touch()

            def prepare_episode(_path, prepared_directory, *, fps, task, source_reference):
                prepared = Path(prepared_directory)
                prepared.mkdir()
                return prepared

            def run_subprocess(command, **kwargs):
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])

            callback = create_lerobot_export_callback(
                subprocess_runner=run_subprocess,
                prepare_episode=prepare_episode,
            )

            with self.assertRaisesRegex(RuntimeError, "超过 20 分钟"):
                callback(source, make_config(), {"episode_id": "episode-42"})

    def test_lerobot_cli_appends_prepared_episode_and_prints_dataset_path(self):
        from mrs_robot_lab.recorders import lerobot_export_cli

        with tempfile.TemporaryDirectory() as temp:
            prepared = Path(temp) / "prepared"
            dataset_root = Path(temp) / "datasets"
            prepared.mkdir()
            stdout = io.StringIO()
            with mock.patch.object(
                lerobot_export_cli,
                "append_prepared_episode_to_lerobot",
                return_value=dataset_root / "openflex_sim" / "navigation_to_goal",
            ) as append_episode, contextlib.redirect_stdout(stdout):
                result = lerobot_export_cli.main(
                    [
                        "--prepared-directory",
                        str(prepared),
                        "--dataset-root",
                        str(dataset_root),
                        "--repo-id",
                        "openflex_sim/navigation_to_goal",
                        "--fps",
                        "30",
                        "--task",
                        "navigation_to_goal",
                    ]
                )

        self.assertEqual(result, 0)
        append_episode.assert_called_once_with(
            str(prepared),
            str(dataset_root),
            "openflex_sim/navigation_to_goal",
            fps=30,
            task="navigation_to_goal",
        )
        self.assertEqual(
            json.loads(stdout.getvalue()),
            {"passed": True, "dataset_path": str(dataset_root / "openflex_sim" / "navigation_to_goal")},
        )

    def test_lerobot_cli_reports_append_failure_without_a_success_result(self):
        from mrs_robot_lab.recorders import lerobot_export_cli

        stderr = io.StringIO()
        with mock.patch.object(
            lerobot_export_cli,
            "append_prepared_episode_to_lerobot",
            side_effect=ValueError("invalid staging data"),
        ), contextlib.redirect_stderr(stderr):
            result = lerobot_export_cli.main(
                [
                    "--prepared-directory",
                    "/tmp/prepared",
                    "--dataset-root",
                    "/tmp/dataset",
                    "--repo-id",
                    "openflex_sim/navigation_to_goal",
                    "--fps",
                    "30",
                    "--task",
                    "navigation_to_goal",
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(
            json.loads(stderr.getvalue()),
            {"passed": False, "error": "invalid staging data"},
        )


if __name__ == "__main__":
    unittest.main()
