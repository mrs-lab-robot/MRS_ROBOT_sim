"""Tests for provenance-gated, atomic episode persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.recorders.episode_controller import (
    EpisodeController,
    EpisodeQualityError,
)


class EpisodeControllerTest(unittest.TestCase):
    def _append_one_sample(self, controller: EpisodeController) -> None:
        width = len(JOINT_STATE_NAMES)
        controller.append(
            sim_time_ns=1_000_000,
            step_index=0,
            operator_command=[0.0] * ACTION_DIMENSION,
            applied_target=[0.1] * ACTION_DIMENSION,
            joint_position=[0.0] * width,
            joint_velocity=[0.0] * width,
            next_joint_position=[0.1] * width,
            command_seq=1,
        )

    def test_incomplete_provenance_is_rejected_before_writing_hdf5(self):
        controller = EpisodeController()
        controller.start(
            {
                "episode_id": "episode-001",
                "config_hash": "config",
                "input_source": "keyboard",
            }
        )
        self._append_one_sample(controller)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(EpisodeQualityError) as context:
                controller.save(Path(temp_dir) / "episode.h5")

        self.assertIn("contract_hash", context.exception.report.reasons)
        self.assertFalse(controller.output_path)

    def test_valid_episode_is_atomically_committed_and_marked_trainable(self):
        controller = EpisodeController()
        controller.start(
            {
                "episode_id": "episode-001",
                "config_hash": "config",
                "contract_hash": "contract",
                "random_seed": 5,
                "input_source": "keyboard",
            }
        )
        self._append_one_sample(controller)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = controller.save(Path(temp_dir) / "episode.h5")

            self.assertEqual(output.name, "episode.h5")
            self.assertTrue(output.is_file())
            self.assertEqual(list(output.parent.glob("*.tmp")), [])
            import h5py

            with h5py.File(output, "r") as dataset:
                metadata = json.loads(dataset.attrs["metadata_json"])
                self.assertEqual(metadata["quality_status"], "passed")
                self.assertEqual(metadata["episode_schema_version"], "1.0")

    def test_recording_fault_annotation_is_persisted_and_not_marked_trainable(self):
        controller = EpisodeController()
        controller.start(
            {
                "episode_id": "episode-vr-fault",
                "config_hash": "config",
                "contract_hash": "contract",
                "random_seed": 5,
                "input_source": "vr",
            }
        )
        self._append_one_sample(controller)
        controller.annotate(
            {
                "recording_faults": ["VR sidecar disconnected"],
                "teleop_ready_at_end": False,
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output = controller.save(Path(temp_dir) / "episode.h5")
            import h5py

            with h5py.File(output, "r") as dataset:
                metadata = json.loads(dataset.attrs["metadata_json"])

        self.assertEqual(metadata["recording_faults"], ["VR sidecar disconnected"])
        self.assertFalse(metadata["teleop_ready_at_end"])
        self.assertEqual(metadata["quality_status"], "failed")

    def test_failed_hdf5_write_removes_temporary_file(self):
        controller = EpisodeController()
        controller.start(
            {
                "episode_id": "episode-write-failure",
                "config_hash": "config",
                "contract_hash": "contract",
                "random_seed": 5,
                "input_source": "keyboard",
                "not_json_serializable": object(),
            }
        )
        self._append_one_sample(controller)

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(TypeError):
                controller.save(Path(temp_dir) / "episode.h5")
            self.assertEqual(list(Path(temp_dir).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
