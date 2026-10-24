"""Round-trip and conversion-transparency tests for the mimic_annotation group."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from mrs_robot_lab.recorders.teleop_episode import (
    EpisodeState,
    TeleopEpisodeRecorder,
)


def _pose_matrix(x: float, y: float, z: float) -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _annotation_step(*, step: int = 0, phase: int = 0, object_names=("transport_box_rigid_body",), labels=("approach_and_grasp", "lift", "carry_and_place")):
    drift = 0.01 * step
    return {
        "object_names": object_names,
        "subtask_labels": labels,
        "object_pose": [_pose_matrix(0.5 + drift, 0.1, 0.3) for _ in object_names],
        "eef_pose": [_pose_matrix(0.2, -0.3, 0.9), _pose_matrix(0.2, 0.3, 0.9)],
        "target_eef_pose": [_pose_matrix(0.21, -0.31, 0.91), _pose_matrix(0.21, 0.31, 0.91)],
        "subtask_signal": phase,
        "subtask_term_flags": [1 if phase >= index + 1 else 0 for index in range(len(labels))],
    }


def _append_rows(recorder: TeleopEpisodeRecorder, *, steps: int, with_mimic: bool) -> None:
    for index in range(steps):
        recorder.append(
            sim_time_ns=1_000_000 * (index + 1),
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=index,
            mimic_annotation=_annotation_step(step=index, phase=index % 3) if with_mimic else None,
        )


class MimicAnnotationRecordingTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_round_trip_writes_mimic_annotation_group(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        _append_rows(recorder, steps=5, with_mimic=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            saved = recorder.save(
                path,
                metadata={"robot_id": "openflex", "action_dimension": 22},
            )
            self.assertEqual(saved, path)
            import h5py
            import numpy as np

            with h5py.File(path, "r") as episode:
                self.assertIn("mimic_annotation", episode)
                group = episode["mimic_annotation"]
                self.assertEqual(int(group.attrs["version"]), 1)
                names = [name.decode() if isinstance(name, bytes) else str(name) for name in group["object_names"][:]]
                labels = [label.decode() if isinstance(label, bytes) else str(label) for label in group["subtask_labels"][:]]
                self.assertEqual(names, ["transport_box_rigid_body"])
                self.assertEqual(labels, ["approach_and_grasp", "lift", "carry_and_place"])
                self.assertEqual(group["object_pose"].shape, (5, 1, 4, 4))
                self.assertEqual(group["eef_pose"].shape, (5, 2, 4, 4))
                self.assertEqual(group["target_eef_pose"].shape, (5, 2, 4, 4))
                self.assertEqual(group["subtask_signal"].shape, (5,))
                self.assertEqual(group["subtask_term_flags"].shape, (5, 3))
                # object_pose 数值随采样变化（仿真里物体位姿要动）
                self.assertGreater(
                    float(np.abs(np.diff(group["object_pose"][:], axis=0)).max()), 0.0
                )
                # 子任务信号随相位切换
                self.assertEqual(list(group["subtask_signal"][:]), [0, 1, 2, 0, 1])
                self.assertEqual(int(group["subtask_term_flags"][1, 0]), 1)
                self.assertEqual(int(group["subtask_term_flags"][0, 0]), 0)
                metadata = json.loads(episode.attrs["metadata_json"])
                self.assertTrue(metadata["mimic_annotation"]["enabled"])
                self.assertEqual(metadata["mimic_annotation"]["object_names"], names)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_group_is_sibling_of_lio_fields(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        for index in range(2):
            recorder.append(
                sim_time_ns=1_000_000 * (index + 1),
                action=[0.0] * 22,
                joint_position=[0.0] * 19,
                joint_velocity=[0.0] * 19,
                next_joint_position=[0.0] * 19,
                command_seq=index,
                lio_pose=[0.0, 0.0, 0.0],
                lio_velocity=[0.0, 0.0, 0.0],
                mimic_annotation=_annotation_step(),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            recorder.save(path, metadata={"robot_id": "openflex", "action_dimension": 22})
            import h5py

            with h5py.File(path, "r") as episode:
                top_level = set(episode.keys())
                self.assertIn("mimic_annotation", top_level)
                self.assertIn("lio_pose", top_level)
                self.assertIn("lio_velocity", top_level)
                self.assertIn("action", top_level)
                # 与 lio_pose 同级：都是顶层条目，不是嵌套组
                self.assertEqual(episode["mimic_annotation"].parent.name, "/")

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_disabled_flag_writes_no_group(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        _append_rows(recorder, steps=2, with_mimic=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.hdf5"
            recorder.save(path, metadata={"robot_id": "openflex", "action_dimension": 22})
            import h5py

            with h5py.File(path, "r") as episode:
                self.assertNotIn("mimic_annotation", episode)
                metadata = json.loads(episode.attrs["metadata_json"])
                self.assertNotIn("mimic_annotation", metadata)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_inconsistent_annotation_mix_is_rejected(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        recorder.append(
            sim_time_ns=1_000_000,
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=0,
            mimic_annotation=_annotation_step(),
        )
        with self.assertRaises(ValueError):
            recorder.append(
                sim_time_ns=2_000_000,
                action=[0.0] * 22,
                joint_position=[0.0] * 19,
                joint_velocity=[0.0] * 19,
                next_joint_position=[0.0] * 19,
                command_seq=1,
            )
        recorder.discard()
        self.assertIs(recorder.state, EpisodeState.IDLE)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_object_name_change_is_rejected(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        recorder.append(
            sim_time_ns=1_000_000,
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=0,
            mimic_annotation=_annotation_step(),
        )
        with self.assertRaises(ValueError):
            recorder.append(
                sim_time_ns=2_000_000,
                action=[0.0] * 22,
                joint_position=[0.0] * 19,
                joint_velocity=[0.0] * 19,
                next_joint_position=[0.0] * 19,
                command_seq=1,
                mimic_annotation=_annotation_step(object_names=("pick_cube",)),
            )

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py is required for HDF5 recording tests")
    def test_nonfinite_pose_is_rejected(self) -> None:
        recorder = TeleopEpisodeRecorder()
        recorder.start()
        bad = _annotation_step()
        bad["object_pose"] = [[
            [1.0, 0.0, 0.0, float("nan")],
            [0.0, 1.0, 0.0, 0.1],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]]
        with self.assertRaises(ValueError):
            recorder.append(
                sim_time_ns=1_000_000,
                action=[0.0] * 22,
                joint_position=[0.0] * 19,
                joint_velocity=[0.0] * 19,
                next_joint_position=[0.0] * 19,
                command_seq=0,
                mimic_annotation=bad,
            )


class MimicAnnotationSignalTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required")
    def test_pick_cube_subtask_signal_switches_at_task_lift_threshold(self) -> None:
        import torch

        from mrs_robot_lab.recorders.mimic_annotation import MimicAnnotationCollector

        class AssetData:
            root_pos_w = torch.tensor([[0.40, 0.0, 0.69]], dtype=torch.float32)

        class Asset:
            data = AssetData()

        class Scene:
            rigid_objects = {"pick_cube": Asset()}

            def __getitem__(self, name):
                return self.rigid_objects[name]

        class Env:
            unwrapped = type("Unwrapped", (), {"scene": Scene()})()

        class Robot:
            @staticmethod
            def find_bodies(_patterns):
                return [1, 2], ["openarmx_left_link7", "openarmx_right_link7"]

        scene = Env.unwrapped.scene
        collector = MimicAnnotationCollector(
            Robot(), Env(), task_id="openflex_pick_cube"
        )
        collector.begin_episode()
        self.assertEqual(collector._collect_subtask_signal({}), (0, [0, 0]))
        scene["pick_cube"].data.root_pos_w[0, 2] += 0.20
        self.assertEqual(collector._collect_subtask_signal({}), (0, [0, 0]))
        scene["pick_cube"].data.root_pos_w[0, 2] += 0.02
        self.assertEqual(collector._collect_subtask_signal({}), (1, [1, 0]))


class MimicAnnotationConversionTransparencyTest(unittest.TestCase):
    """mimic_annotation 组对 25 维 LeRobot 导出透明：跳过、保留、不报 unknown-field。"""

    def _write_source_episode(self, directory: Path, *, with_mimic: bool) -> Path:
        import h5py
        import numpy as np

        from mrs_robot_lab.recorders.teleop_episode import JOINT_STATE_NAMES

        path = directory / "source.hdf5"
        steps = 4
        with h5py.File(path, "w") as episode:
            episode.attrs["format"] = "mrs_robot_capture_v1"
            episode.attrs["metadata_json"] = json.dumps(
                {
                    "robot_id": "openflex",
                    "action_dimension": 22,
                    "joint_state_order": list(JOINT_STATE_NAMES),
                    "data_mode": "ground_truth",
                }
            )
            episode.create_dataset("sim_time_ns", data=np.arange(1, steps + 1, dtype="uint64") * 33_000_000)
            episode.create_dataset("action", data=np.zeros((steps, 22), dtype="float32"))
            episode.create_dataset("joint_position", data=np.zeros((steps, 19), dtype="float32"))
            episode.create_dataset("joint_velocity", data=np.zeros((steps, 19), dtype="float32"))
            episode.create_dataset("next_joint_position", data=np.zeros((steps, 19), dtype="float32"))
            episode.create_dataset("lio_pose", data=np.tile(np.arange(steps, dtype="float32").reshape(-1, 1), (1, 3)))
            episode.create_dataset("lio_velocity", data=np.zeros((steps, 3), dtype="float32"))
            camera = episode.create_group("camera_frames/base_d435")
            jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
            jpeg = camera.create_dataset("jpeg", (steps,), dtype=jpeg_dtype)
            from PIL import Image
            import io

            buffer = io.BytesIO()
            Image.new("RGB", (4, 4), color=(128, 64, 32)).save(buffer, format="JPEG")
            frame = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
            for index in range(steps):
                jpeg[index] = frame
            camera.create_dataset("sim_time_ns", data=np.arange(1, steps + 1, dtype="uint64") * 33_000_000)
            if with_mimic:
                group = episode.create_group("mimic_annotation")
                group.attrs["version"] = 1
                group.create_dataset("object_names", data=np.asarray(["transport_box_rigid_body"], dtype=h5py.string_dtype("utf-8")))
                group.create_dataset("subtask_labels", data=np.asarray(["a", "b"], dtype=h5py.string_dtype("utf-8")))
                identity = np.eye(4, dtype="float32")
                group.create_dataset("object_pose", data=np.tile(identity, (steps, 1, 1, 1)))
                group.create_dataset("eef_pose", data=np.tile(identity, (steps, 2, 1, 1)))
                group.create_dataset("target_eef_pose", data=np.tile(identity, (steps, 2, 1, 1)))
                group.create_dataset("subtask_signal", data=np.zeros(steps, dtype="int32"))
                group.create_dataset("subtask_term_flags", data=np.zeros((steps, 2), dtype="uint8"))
        return path

    @unittest.skipUnless(
        importlib.util.find_spec("h5py") and importlib.util.find_spec("PIL"),
        "h5py and Pillow are required",
    )
    def test_conversion_skips_mimic_annotation_without_unknown_field_error(self) -> None:
        import tempfile
        from pathlib import Path as PathlibPath

        from mrs_robot_lab.recorders.lerobot_export import load_arena_hdf5_frames

        with tempfile.TemporaryDirectory() as directory:
            source = self._write_source_episode(PathlibPath(directory), with_mimic=True)
            frames, features = load_arena_hdf5_frames(source, fps=30, task="demo", state_dim=25)
            self.assertGreaterEqual(len(frames), 3)
            self.assertEqual(features["observation.state"]["shape"], (25,))
            self.assertNotIn("mimic_annotation", features)
            self.assertNotIn("mimic_annotation", frames[0])
            # HDF5 里原样保留
            import h5py

            with h5py.File(source, "r") as episode:
                self.assertIn("mimic_annotation", episode)

    @unittest.skipUnless(
        importlib.util.find_spec("h5py") and importlib.util.find_spec("PIL"),
        "h5py and Pillow are required",
    )
    def test_conversion_without_annotation_matches_with_annotation_schema(self) -> None:
        import tempfile
        from pathlib import Path as PathlibPath

        from mrs_robot_lab.recorders.lerobot_export import load_arena_hdf5_frames

        with tempfile.TemporaryDirectory() as directory:
            root = PathlibPath(directory)
            frames_with, features_with = load_arena_hdf5_frames(
                self._write_source_episode(root, with_mimic=True), fps=30, task="demo", state_dim=25
            )
            frames_without, features_without = load_arena_hdf5_frames(
                self._write_source_episode(root, with_mimic=False), fps=30, task="demo", state_dim=25
            )
            self.assertEqual(
                {name: (feature["dtype"], tuple(feature["shape"])) for name, feature in features_with.items()},
                {name: (feature["dtype"], tuple(feature["shape"])) for name, feature in features_without.items()},
            )
            self.assertEqual(len(frames_with), len(frames_without))


if __name__ == "__main__":
    unittest.main()
