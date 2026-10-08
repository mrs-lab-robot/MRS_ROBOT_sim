from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mrs_robot_lab.recorders.lerobot_export import append_prepared_episode_to_lerobot


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "LeRobot is not installed")
class PreparedLeRobotWriterIntegrationTest(unittest.TestCase):
    def test_prepared_episode_creates_dataset_reopenable_by_training_reader(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            images = prepared / "images" / "head"
            images.mkdir(parents=True)
            np.save(prepared / "action.npy", np.zeros((1, 22), dtype=np.float32))
            np.save(prepared / "observation_state.npy", np.zeros((1, 19), dtype=np.float32))
            np.save(prepared / "timestamp.npy", np.asarray([0.0], dtype=np.float64))
            # libsvtav1 rejects tiny test-only dimensions on some CPU builds;
            # use a realistic camera frame so this checks the LeRobot path,
            # not codec edge cases unrelated to the episode contract.
            Image.new("RGB", (64, 64), (20, 40, 60)).save(images / "00000000.jpg", format="JPEG")
            (prepared / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "mrs_robot_capture_lerobot_staging_v1",
                        "episode_id": "integration-episode-001",
                        "source_sha256": "a" * 64,
                        "fps": 30,
                        "task": "Navigate to the goal",
                        "frame_count": 1,
                        "cameras": ["head"],
                    }
                ),
                encoding="utf-8",
            )

            destination = append_prepared_episode_to_lerobot(
                prepared,
                root / "datasets",
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
            frame = dataset[0]
            self.assertIn("action", frame)
            self.assertIn("observation.state", frame)
            self.assertIn("observation.images.head", frame)


if __name__ == "__main__":
    unittest.main()
