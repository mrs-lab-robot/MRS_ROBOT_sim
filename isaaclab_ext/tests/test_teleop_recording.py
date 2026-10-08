from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from mrs_robot_lab.recorders.teleop_episode import EpisodeState, TeleopEpisodeRecorder


class TeleopEpisodeRecorderTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_start_save_and_discard_are_explicit_episode_transitions(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        recorder.append(
            sim_time_ns=10,
            action=[0.1] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.1] * 19,
            command_seq=7,
        )
        self.assertEqual(recorder.state, EpisodeState.RECORDING)

        with tempfile.TemporaryDirectory() as directory:
            output = recorder.save(Path(directory) / "demo_000001.hdf5", metadata={"scene": "openflex"})
            self.assertTrue(output.is_file())
            import h5py

            with h5py.File(output, "r") as episode:
                self.assertEqual(episode.attrs["format"], "mrs_robot_capture_v1")
                self.assertEqual(tuple(episode["action"].shape), (1, 22))
                self.assertEqual(tuple(episode["joint_position"].shape), (1, 19))
                self.assertAlmostEqual(float(episode["next_joint_position"][0, 0]), 0.1)
            self.assertEqual(recorder.state, EpisodeState.IDLE)

        recorder.start()
        recorder.append(sim_time_ns=20, action=[0.0] * 22, joint_position=[0.0] * 19,
                        next_joint_position=[0.0] * 19,
                        joint_velocity=[0.0] * 19, command_seq=8)
        recorder.discard()
        self.assertEqual(recorder.state, EpisodeState.IDLE)
        self.assertEqual(recorder.num_steps, 0)

    def test_rejects_samples_with_wrong_contract_dimensions(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        with self.assertRaisesRegex(ValueError, "22"):
            recorder.append(sim_time_ns=0, action=[0.0] * 21, joint_position=[0.0] * 19,
                            next_joint_position=[0.0] * 19,
                            joint_velocity=[0.0] * 19, command_seq=1)

    def test_save_never_overwrites_an_existing_episode_file(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        recorder.append(sim_time_ns=1, action=[0.0] * 22, joint_position=[0.0] * 19,
                        next_joint_position=[0.0] * 19,
                        joint_velocity=[0.0] * 19, command_seq=1)
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "kept.hdf5"
            existing.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                recorder.save(existing, metadata={})
            self.assertEqual(existing.read_bytes(), b"preserve")

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_saved_episode_keeps_timestamped_camera_frames_for_dataset_export(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        jpeg = bytes((0xFF, 0xD8, 0xFF, 0xD9))
        recorder.append(
            sim_time_ns=123_000_000,
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.1] * 19,
            command_seq=4,
            camera_frames={"head_d435": jpeg},
        )

        with tempfile.TemporaryDirectory() as directory:
            output = recorder.save(Path(directory) / "episode.hdf5", metadata={})
            import h5py

            with h5py.File(output, "r") as episode:
                camera = episode["camera_frames/head_d435"]
                self.assertEqual(camera["sim_time_ns"][:].tolist(), [123_000_000])
                self.assertEqual(camera["jpeg"][0].tobytes(), jpeg)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_independently_scheduled_camera_streams_can_have_different_frame_counts(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        jpeg = bytes((0xFF, 0xD8, 0xFF, 0xD9))
        for step in range(4):
            frames = {"base_d435": jpeg}
            if step % 2 == 0:
                frames["head_d435"] = jpeg
            recorder.append(
                sim_time_ns=step * 10,
                step_index=step,
                operator_command=[0.0] * 22,
                applied_target=[0.0] * 22,
                joint_position=[0.0] * 19,
                joint_velocity=[0.0] * 19,
                next_joint_position=[0.1] * 19,
                command_seq=step,
                camera_frames=frames,
            )

        with tempfile.TemporaryDirectory() as directory:
            output = recorder.save(Path(directory) / "multi-rate.hdf5", metadata={})
            import h5py

            with h5py.File(output, "r") as episode:
                self.assertEqual(len(episode["camera_frames/base_d435/jpeg"]), 4)
                self.assertEqual(len(episode["camera_frames/head_d435/jpeg"]), 2)
                self.assertEqual(
                    episode["camera_frames/head_d435/step_index"][:].tolist(), [0, 2]
                )


if __name__ == "__main__":
    unittest.main()
