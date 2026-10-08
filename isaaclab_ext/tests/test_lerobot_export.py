from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from mrs_robot_lab.assets.robot_interface import JOINT_STATE_NAMES, LEFT_ARM_JOINTS
from mrs_robot_lab.recorders.lerobot_export import (
    load_arena_hdf5_frames,
    prepare_arena_hdf5_for_lerobot,
)


@unittest.skipUnless(
    importlib.util.find_spec("h5py") and importlib.util.find_spec("PIL"),
    "h5py and Pillow are required for Arena dataset export tests",
)
class ArenaLeRobotExportTest(unittest.TestCase):
    def test_export_mapping_preserves_robot_state_targets_base_action_and_camera_time(self) -> None:
        import h5py
        import numpy as np
        from PIL import Image
        from io import BytesIO

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            state = np.arange(19, dtype=np.float32).reshape(1, 19)
            next_state = state + 100.0
            actions = np.zeros((1, 22), dtype=np.float32)
            actions[0, :3] = (0.25, -0.5, 1.25)
            image_bytes = BytesIO()
            Image.new("RGB", (3, 2), (20, 40, 60)).save(image_bytes, format="JPEG")
            jpeg = image_bytes.getvalue()

            with h5py.File(path, "w") as episode:
                episode.attrs["format"] = "mrs_robot_capture_v1"
                episode.attrs["metadata_json"] = json.dumps(
                    {"joint_state_order": list(JOINT_STATE_NAMES)}
                )
                episode.create_dataset("sim_time_ns", data=np.asarray([0], dtype="uint64"))
                episode.create_dataset("action", data=actions)
                applied_targets = np.full_like(actions, 0.37)
                applied_targets[0, :3] = [0.25, -0.5, 1.25]
                episode.create_dataset("applied_target", data=applied_targets)
                episode.create_dataset("joint_position", data=state)
                episode.create_dataset("next_joint_position", data=next_state)
                camera = episode.create_group("camera_frames/head_d435")
                camera.create_dataset("sim_time_ns", data=np.asarray([0], dtype="uint64"))
                encoded = camera.create_dataset(
                    "jpeg", (1,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                encoded[0] = np.frombuffer(jpeg, dtype=np.uint8)

            frames, features = load_arena_hdf5_frames(path, fps=10, task="Pick the cube")
            prepared = prepare_arena_hdf5_for_lerobot(
                path,
                Path(directory) / "prepared",
                fps=10,
                task="Pick the cube",
            )
            manifest = json.loads((prepared / "manifest.json").read_text())
            self.assertEqual(manifest["frame_count"], 1)
            self.assertEqual(manifest["cameras"], ["head"])
            self.assertTrue((prepared / "images" / "head" / "00000000.jpg").is_file())

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["action"].shape, (22,))
        self.assertEqual(frames[0]["observation.state"].shape, (19,))
        self.assertAlmostEqual(float(frames[0]["action"][0]), 0.37)
        self.assertEqual(frames[0]["action"][-3:].tolist(), [0.25, -0.5, 1.25])
        self.assertAlmostEqual(
            float(frames[0]["observation.state"][0]),
            float(state[0, JOINT_STATE_NAMES.index(LEFT_ARM_JOINTS[0])]),
        )
        self.assertIn("observation.images.head", features)
        self.assertEqual(frames[0]["observation.images.head"].shape, (2, 3, 3))
        self.assertEqual(frames[0]["task"], "Pick the cube")

    def test_resampling_never_assigns_a_future_state_or_camera_frame(self) -> None:
        import h5py
        import numpy as np
        from PIL import Image
        from io import BytesIO

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "causal-episode.hdf5"
            actions = np.zeros((2, 22), dtype=np.float32)
            states = np.zeros((2, 19), dtype=np.float32)
            states[1, 0] = 99.0
            with h5py.File(path, "w") as episode:
                episode.attrs["format"] = "mrs_robot_arena_teleop_v1"
                episode.attrs["metadata_json"] = json.dumps(
                    {"joint_state_order": list(JOINT_STATE_NAMES)}
                )
                episode.create_dataset("sim_time_ns", data=np.asarray([0, 80_000_000]))
                episode.create_dataset("action", data=actions)
                episode.create_dataset("joint_position", data=states)
                episode.create_dataset("next_joint_position", data=states + 99.0)
                camera = episode.create_group("camera_frames/head_d435")
                camera.create_dataset("sim_time_ns", data=np.asarray([0, 60_000_000]))
                encoded = camera.create_dataset(
                    "jpeg", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                for index, color in enumerate(((255, 0, 0), (0, 0, 255))):
                    image_bytes = BytesIO()
                    Image.new("RGB", (8, 8), color).save(image_bytes, format="JPEG", quality=100)
                    encoded[index] = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)

            frames, _features = load_arena_hdf5_frames(path, fps=20, task="causal alignment")

        self.assertEqual(len(frames), 2)
        self.assertEqual(float(frames[1]["observation.state"][0]), 0.0)
        self.assertEqual(float(frames[1]["action"][0]), 0.0)
        image = frames[1]["observation.images.head"]
        self.assertGreater(float(image[..., 0].mean()), float(image[..., 2].mean()))

    @unittest.skipUnless(
        importlib.util.find_spec("lerobot"),
        "LeRobot is required for the dataset writer integration test",
    )
    def test_export_creates_a_training_dataset_with_video_features(self) -> None:
        import h5py
        import numpy as np
        from PIL import Image
        from io import BytesIO

        from mrs_robot_lab.recorders.lerobot_export import export_arena_hdf5_to_lerobot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "episode.hdf5"
            state = np.zeros((2, 19), dtype=np.float32)
            next_state = np.ones((2, 19), dtype=np.float32)
            actions = np.zeros((2, 22), dtype=np.float32)
            actions[:, :3] = (0.25, -0.5, 1.25)
            with h5py.File(source, "w") as episode:
                episode.attrs["format"] = "mrs_robot_arena_teleop_v1"
                episode.attrs["metadata_json"] = json.dumps(
                    {"joint_state_order": list(JOINT_STATE_NAMES)}
                )
                episode.create_dataset("sim_time_ns", data=np.asarray([0, 100_000_000]))
                episode.create_dataset("action", data=actions)
                episode.create_dataset("applied_target", data=actions)
                episode.create_dataset("joint_position", data=state)
                episode.create_dataset("next_joint_position", data=next_state)
                camera = episode.create_group("camera_frames/head_d435")
                camera.create_dataset("sim_time_ns", data=np.asarray([0, 100_000_000]))
                encoded = camera.create_dataset(
                    "jpeg", (2,), dtype=h5py.vlen_dtype(np.dtype("uint8"))
                )
                for index, color in enumerate(((20, 40, 60), (60, 40, 20))):
                    image_bytes = BytesIO()
                    Image.new("RGB", (16, 12), color).save(image_bytes, format="JPEG")
                    encoded[index] = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)

            destination = export_arena_hdf5_to_lerobot(
                source,
                root / "datasets",
                "test/arena_smoke",
                fps=10,
                task="Pick the cube",
            )
            info = json.loads((destination / "meta" / "info.json").read_text())

        self.assertEqual(info["total_episodes"], 1)
        self.assertEqual(info["fps"], 10)
        self.assertIn("action", info["features"])
        self.assertIn("observation.state", info["features"])
        self.assertIn("observation.images.head", info["features"])


@unittest.skipUnless(
    importlib.util.find_spec("lerobot")
    and importlib.util.find_spec("numpy")
    and importlib.util.find_spec("PIL"),
    "LeRobot, NumPy and Pillow are required for the prepared-dataset writer test",
)
class PreparedArenaLeRobotWriterTest(unittest.TestCase):
    def test_prepared_episode_writes_training_metadata_and_video_features(self) -> None:
        import numpy as np
        from PIL import Image

        from mrs_robot_lab.recorders.lerobot_export import export_prepared_episode_to_lerobot

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            head_frames = prepared / "images" / "head"
            head_frames.mkdir(parents=True)
            np.save(prepared / "action.npy", np.zeros((2, 22), dtype=np.float32))
            np.save(prepared / "observation_state.npy", np.zeros((2, 19), dtype=np.float32))
            np.save(prepared / "timestamp.npy", np.asarray((0.0, 0.1), dtype=np.float64))
            for index, color in enumerate(((20, 40, 60), (60, 40, 20))):
                Image.new("RGB", (64, 64), color).save(head_frames / f"{index:08d}.jpg")
            (prepared / "manifest.json").write_text(
                json.dumps({
                    "format": "mrs_robot_arena_lerobot_staging_v1",
                    "fps": 10,
                    "task": "Pick the cube",
                    "frame_count": 2,
                    "cameras": ["head"],
                }),
                encoding="utf-8",
            )

            destination = export_prepared_episode_to_lerobot(
                prepared,
                root / "datasets",
                "test/arena_smoke",
                fps=10,
                task="Pick the cube",
            )
            info = json.loads((destination / "meta" / "info.json").read_text())

        self.assertEqual(info["total_episodes"], 1)
        self.assertEqual(info["fps"], 10)
        self.assertIn("action", info["features"])
        self.assertIn("observation.state", info["features"])
        self.assertIn("observation.images.head", info["features"])


if __name__ == "__main__":
    unittest.main()
