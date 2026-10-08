from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.recorders.lerobot_export import (
    append_hdf5_episode_to_lerobot,
    append_prepared_episode_to_lerobot,
    prepare_arena_hdf5_for_lerobot,
)


class FakeLeRobotDataset:
    stores: dict[str, dict] = {}

    def __init__(self, repo_id, root, **_kwargs):
        self.repo_id = repo_id
        self.root = Path(root)
        self.store = self.stores[str(self.root)]
        self.fps = self.store["fps"]
        self.features = self.store["features"]
        self.meta = SimpleNamespace(total_episodes=self.store["episode_count"])
        self.pending_frames = []

    @classmethod
    def create(cls, repo_id, fps, *, features, root, **_kwargs):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
        cls.stores[str(root)] = {
            "fps": fps,
            "features": features,
            "episode_count": 0,
            "frames": [],
        }
        return cls(repo_id, root)

    def add_frame(self, frame):
        self.pending_frames.append(frame)

    def save_episode(self):
        self.store["frames"].append(list(self.pending_frames))
        self.store["episode_count"] += 1
        self.meta.total_episodes = self.store["episode_count"]


class LeRobotAppendTest(unittest.TestCase):
    def setUp(self):
        FakeLeRobotDataset.stores.clear()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.dataset_root = self.root / "datasets"
        self.episode_root = self.root / "episodes"
        self.dataset_root.mkdir()
        self.episode_root.mkdir()

    def tearDown(self):
        self.tempdir.cleanup()

    def _write_episode(self, episode_id: str, *, filename: str | None = None, color=(15, 40, 90)) -> Path:
        path = self.episode_root / f"{filename or episode_id}.hdf5"
        state = np.zeros((1, len(JOINT_STATE_NAMES)), dtype=np.float32)
        action = np.zeros((1, ACTION_DIMENSION), dtype=np.float32)
        from io import BytesIO

        encoded = BytesIO()
        Image.new("RGB", (4, 4), color).save(encoded, format="JPEG")
        with h5py.File(path, "w") as episode:
            episode.attrs["format"] = "mrs_robot_capture_v1"
            episode.attrs["metadata_json"] = json.dumps(
                {
                    "episode_id": episode_id,
                    "joint_state_order": list(JOINT_STATE_NAMES),
                    "config_hash": "config-hash-for-test",
                    "contract_hash": "contract-hash-for-test",
                    "random_seed": 42,
                    "input_source": "keyboard",
                    "task_id": "navigation_to_goal",
                    "host_config_path": "/private/machine/config.yaml",
                }
            )
            episode.create_dataset("sim_time_ns", data=[0])
            episode.create_dataset("action", data=action)
            episode.create_dataset("applied_target", data=action)
            episode.create_dataset("joint_position", data=state)
            episode.create_dataset("next_joint_position", data=state)
            camera = episode.create_group("camera_frames/head_d435")
            camera.create_dataset("sim_time_ns", data=[0])
            jpeg = camera.create_dataset(
                "jpeg", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
            )
            jpeg[0] = np.frombuffer(encoded.getvalue(), dtype=np.uint8)
        return path

    def _export(self, episode_id: str) -> Path:
        source_path = self.episode_root / f"{episode_id}.hdf5"
        if not source_path.exists():
            self._write_episode(episode_id)
        return append_hdf5_episode_to_lerobot(
            source_path,
            self.dataset_root,
            "openflex_sim/navigation_v1",
            fps=30,
            task="Navigate to the goal",
            dataset_class=FakeLeRobotDataset,
        )

    def test_export_creates_then_appends_and_duplicate_episode_is_idempotent(self):
        destination = self._export("episode-001")
        self.assertEqual(FakeLeRobotDataset.stores[str(destination)]["episode_count"], 1)

        portable_manifest = destination / "meta" / "mrs_episode_manifest.json"
        portable_manifest.unlink()
        self._export("episode-001")
        self.assertEqual(FakeLeRobotDataset.stores[str(destination)]["episode_count"], 1)
        self.assertTrue(portable_manifest.is_file())

        self._export("episode-002")
        self.assertEqual(FakeLeRobotDataset.stores[str(destination)]["episode_count"], 2)
        self.assertEqual(len(FakeLeRobotDataset.stores[str(destination)]["frames"]), 2)
        manifest = json.loads(
            (destination / "meta" / "mrs_episode_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(
            [manifest["episodes"][episode_id]["episode_index"] for episode_id in ("episode-001", "episode-002")],
            [0, 1],
        )
        provenance = manifest["episodes"]["episode-001"]["provenance"]
        self.assertEqual(provenance["config_hash"], "config-hash-for-test")
        self.assertEqual(provenance["input_source"], "keyboard")
        self.assertEqual(provenance["episode_id"], "episode-001")
        self.assertNotIn("joint_state_order", provenance)
        self.assertNotIn("host_config_path", provenance)
        self.assertEqual(len(manifest["episodes"]["episode-001"]["source_sha256"]), 64)
        self.assertEqual(
            manifest["episodes"]["episode-001"]["source_reference"],
            "episode-001.hdf5",
        )

    def test_export_rejects_reusing_episode_id_for_different_hdf5_content(self):
        destination = self._export("episode-001")
        changed_source = self._write_episode(
            "episode-001",
            filename="different-content",
            color=(220, 10, 30),
        )

        with self.assertRaisesRegex(ValueError, "different HDF5 content"):
            append_hdf5_episode_to_lerobot(
                changed_source,
                self.dataset_root,
                "openflex_sim/navigation_v1",
                fps=30,
                task="Navigate to the goal",
                dataset_class=FakeLeRobotDataset,
            )

        self.assertEqual(FakeLeRobotDataset.stores[str(destination)]["episode_count"], 1)

    def test_export_refuses_to_append_when_feature_schema_differs(self):
        destination = self._export("episode-001")
        FakeLeRobotDataset.stores[str(destination)]["fps"] = 20

        with self.assertRaisesRegex(ValueError, "FPS"):
            self._export("episode-002")

    def test_prepared_export_appends_without_reading_hdf5_in_writer_environment(self):
        source = self._write_episode("episode-prepared")
        prepared = prepare_arena_hdf5_for_lerobot(
            source,
            self.root / "prepared",
            fps=30,
            task="Navigate to the goal",
        )

        destination = append_prepared_episode_to_lerobot(
            prepared,
            self.dataset_root,
            "openflex_sim/navigation_v1",
            fps=30,
            task="Navigate to the goal",
            dataset_class=FakeLeRobotDataset,
        )

        self.assertEqual(FakeLeRobotDataset.stores[str(destination)]["episode_count"], 1)
        self.assertEqual(len(FakeLeRobotDataset.stores[str(destination)]["frames"][0]), 1)
        manifest = json.loads(
            (destination / "meta" / "mrs_episode_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["episodes"]["episode-prepared"]["episode_index"], 0)
        self.assertEqual(
            manifest["episodes"]["episode-prepared"]["provenance"]["config_hash"],
            "config-hash-for-test",
        )
        self.assertEqual(
            manifest["episodes"]["episode-prepared"]["source_reference"],
            "episode-prepared.hdf5",
        )

    def test_preparation_drops_absolute_windows_source_paths(self):
        source = self._write_episode("episode-private-path")
        prepared = prepare_arena_hdf5_for_lerobot(
            source,
            self.root / "prepared-private-path",
            fps=30,
            task="Navigate to the goal",
            source_reference=r"C:\Users\operator\episodes\episode-private-path.hdf5",
        )

        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_reference"], "")

    @unittest.skipUnless(importlib.util.find_spec("lerobot"), "LeRobot is not installed")
    def test_real_lerobot_writer_creates_a_dataset_that_can_be_reopened(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        source = self._write_episode("episode-real")
        destination = append_hdf5_episode_to_lerobot(
            source,
            self.dataset_root,
            "openflex_sim/navigation_v1",
            fps=30,
            task="Navigate to the goal",
        )
        dataset = LeRobotDataset(
            "openflex_sim/navigation_v1",
            root=destination,
            download_videos=False,
        )

        self.assertEqual(dataset.meta.total_episodes, 1)
        self.assertEqual(len(dataset), 1)
        self.assertIn("action", dataset[0])
        self.assertIn("observation.state", dataset[0])
        self.assertIn("observation.images.head", dataset[0])


if __name__ == "__main__":
    unittest.main()
