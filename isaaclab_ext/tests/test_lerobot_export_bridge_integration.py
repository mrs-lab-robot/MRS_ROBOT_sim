"""End-to-end test for HDF5 preparation and the configured LeRobot interpreter."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from io import BytesIO
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from openflex_isaac_contract.session_config import FrequencyConfig, SessionConfig
from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.worker_cli import create_lerobot_export_callback


LEROBOT_PYTHON = os.environ.get("MRS_ROBOT_TEST_LEROBOT_PYTHON", "").strip()
SIM_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    LEROBOT_PYTHON and Path(LEROBOT_PYTHON).is_file() and importlib.util.find_spec("h5py"),
    "set MRS_ROBOT_TEST_LEROBOT_PYTHON to run the two-interpreter LeRobot bridge test",
)
class LeRobotExportBridgeIntegrationTest(unittest.TestCase):
    def test_hdf5_is_prepared_in_worker_python_and_reopened_by_lerobot_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episodes" / "navigation" / "bridge-episode.hdf5"
            source.parent.mkdir(parents=True)
            image_buffer = BytesIO()
            Image.new("RGB", (64, 64), (20, 40, 60)).save(image_buffer, format="JPEG")
            jpeg_bytes = image_buffer.getvalue()
            state = np.zeros((1, len(JOINT_STATE_NAMES)), dtype=np.float32)
            applied_target = np.zeros((1, ACTION_DIMENSION), dtype=np.float32)
            applied_target[0, :3] = (0.2, -0.1, 0.05)

            with h5py.File(source, "w") as episode:
                episode.attrs["format"] = "mrs_robot_capture_v1"
                episode.attrs["metadata_json"] = json.dumps(
                    {
                        "episode_id": "bridge-episode-001",
                        "config_hash": "bridge-config-hash",
                        "contract_hash": "bridge-contract-hash",
                        "random_seed": 7,
                        "input_source": "keyboard",
                        "task_id": "navigation_to_goal",
                        "joint_state_order": list(JOINT_STATE_NAMES),
                    }
                )
                episode.create_dataset("sim_time_ns", data=np.asarray([0], dtype=np.uint64))
                episode.create_dataset("action", data=applied_target)
                episode.create_dataset("applied_target", data=applied_target)
                episode.create_dataset("joint_position", data=state)
                episode.create_dataset("next_joint_position", data=state)
                camera = episode.create_group("camera_frames/head_d435")
                camera.create_dataset("sim_time_ns", data=np.asarray([0], dtype=np.uint64))
                jpeg = camera.create_dataset(
                    "jpeg", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                jpeg[0] = np.frombuffer(jpeg_bytes, dtype=np.uint8)

            dataset_root = root / "datasets"
            config = SessionConfig(
                session_id="lerobot-bridge-integration",
                embodiment_id="openflex",
                scene_id="flat_navigation",
                task_id="navigation_to_goal",
                frequency=FrequencyConfig(
                    physics_hz=120,
                    control_hz=30,
                    render_hz=30,
                    output_fps=30,
                ),
                simulation_repo_root=str(SIM_ROOT),
                lerobot_python=LEROBOT_PYTHON,
                episode_root=str(source.parent.parent),
                dataset_root=str(dataset_root),
                dataset_id="openflex_sim/bridge_integration",
            )
            hf_home = root / "hf-home"
            hf_datasets = root / "hf-datasets"
            with mock.patch.dict(
                os.environ,
                {"HF_HOME": str(hf_home), "HF_DATASETS_CACHE": str(hf_datasets)},
                clear=False,
            ):
                dataset_path = create_lerobot_export_callback()(
                    source,
                    config,
                    {"episode_id": "bridge-episode-001"},
                )
                portable_manifest = json.loads(
                    (dataset_path / "meta" / "mrs_episode_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                verify_script = (
                    "import json, sys; "
                    "from lerobot.datasets.lerobot_dataset import LeRobotDataset; "
                    "ds = LeRobotDataset(sys.argv[2], root=sys.argv[1], download_videos=False); "
                    "frame = ds[0]; "
                    "print(json.dumps({'episodes': ds.meta.total_episodes, 'frames': len(ds), "
                    "'has_action': 'action' in frame, 'has_state': 'observation.state' in frame, "
                    "'has_camera': 'observation.images.head' in frame}))"
                )
                reopen_environment = os.environ.copy()
                reopen_environment.pop("PYTHONPATH", None)
                reopen_environment.pop("PYTHONHOME", None)
                reopened = subprocess.run(
                    [LEROBOT_PYTHON, "-c", verify_script, str(dataset_path), config.dataset_id],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                    env=reopen_environment,
                )

        self.assertEqual(reopened.returncode, 0, reopened.stderr)
        result = json.loads(reopened.stdout.strip().splitlines()[-1])
        self.assertEqual(result, {"episodes": 1, "frames": 1, "has_action": True, "has_state": True, "has_camera": True})
        entry = portable_manifest["episodes"]["bridge-episode-001"]
        self.assertEqual(entry["provenance"]["config_hash"], "bridge-config-hash")
        self.assertEqual(entry["source_reference"], "navigation/bridge-episode.hdf5")
        self.assertEqual(len(entry["source_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
