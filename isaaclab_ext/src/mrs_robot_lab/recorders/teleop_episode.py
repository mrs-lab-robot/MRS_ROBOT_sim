"""Explicit start/save/discard HDF5 recording for one manually teleoperated robot."""

from __future__ import annotations

from enum import Enum
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.recorders.episode_schema import CURRENT_EPISODE_FORMAT


JOINT_STATE_DIMENSION = len(JOINT_STATE_NAMES)

# Mimic annotation layout (mimic_annotation/ group, sibling of lio_pose):
# - object_pose: (steps, num_objects, 4, 4) Mimic homogeneous pose matrices
# - eef_pose / target_eef_pose: (steps, 2, 4, 4) [left, right] Mimic pose matrices
# - subtask_signal: (steps,) int32 current subtask id (-1 = unknown)
# - subtask_term_flags: (steps, num_subtask_labels) uint8 completion flags
MIMIC_ANNOTATION_EEF_COUNT = 2
MIMIC_ANNOTATION_POSE_SHAPE = (4, 4)


class EpisodeState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"


class TeleopEpisodeRecorder:
    """Buffer one episode and persist it only after the operator presses Save."""

    def __init__(self) -> None:
        self.state = EpisodeState.IDLE
        self._samples: list[dict[str, Any]] = []
        self._camera_frames: dict[str, list[tuple[int, int, bytes]]] = {}  # (sim_time_ns, step_index, frame)
        self._last_sim_time_ns: int | None = None
        self._last_step_index: int | None = None
        self._mimic_object_names: tuple[str, ...] | None = None
        self._mimic_subtask_labels: tuple[str, ...] | None = None
        self._mimic_num_objects: int | None = None
        self._mimic_num_labels: int | None = None

    @property
    def num_steps(self) -> int:
        return len(self._samples)

    def start(self) -> None:
        if self.state is EpisodeState.RECORDING:
            raise RuntimeError("an episode is already recording")
        # 复用 discard() 的清理：相机帧、上一集的单调性游标、mimic 标签都要清，
        # 否则直接 start() 的路径会把上一集的状态带进来（游标残留会让第一条 append
        # 抛 "sim_time_ns must be monotonic increasing"）。
        self.discard()
        self.state = EpisodeState.RECORDING

    def append(
        self,
        *,
        sim_time_ns: int,
        step_index: int | None = None,
        operator_command: list[float] | tuple[float, ...] | None = None,
        applied_target: list[float] | tuple[float, ...] | None = None,
        action: list[float] | tuple[float, ...] | None = None,
        joint_position: list[float] | tuple[float, ...],
        joint_velocity: list[float] | tuple[float, ...],
        next_joint_position: list[float] | tuple[float, ...],
        command_seq: int,
        raw_command: dict[str, Any] | None = None,
        camera_frames: dict[str, bytes] | None = None,
        lio_pose: list[float] | tuple[float, ...] | None = None,
        lio_velocity: list[float] | tuple[float, ...] | None = None,
        mimic_annotation: dict[str, Any] | None = None,
    ) -> None:
        if self.state is not EpisodeState.RECORDING:
            return

        # 确定使用新API还是旧API
        using_new_api = operator_command is not None and applied_target is not None
        using_legacy_api = action is not None

        if using_new_api and using_legacy_api:
            raise TypeError("cannot use both new API (operator_command/applied_target) and legacy API (action)")
        if not using_new_api and not using_legacy_api:
            raise TypeError("must provide either (operator_command, applied_target) or action")

        # 新API要求显式step_index，旧API自动递增
        if using_new_api:
            if step_index is None:
                raise TypeError("step_index is required when using operator_command/applied_target")
            actual_step_index = step_index
        else:
            # 旧API：自动使用下一个索引
            actual_step_index = len(self._samples) if step_index is None else step_index

        # 验证时间戳和索引单调递增（在修改状态前验证）
        if self._last_sim_time_ns is not None and sim_time_ns <= self._last_sim_time_ns:
            raise ValueError(f"sim_time_ns must be monotonic increasing: {sim_time_ns} <= {self._last_sim_time_ns}")
        if self._last_step_index is not None and actual_step_index <= self._last_step_index:
            raise ValueError(f"step_index must be monotonic increasing: {actual_step_index} <= {self._last_step_index}")

        # 确定动作向量
        if using_new_api:
            operator_vec = operator_command
            applied_vec = applied_target
        else:
            # 旧API：action同时作为operator和applied
            operator_vec = action
            applied_vec = action

        vectors = {
            "operator_command": (operator_vec, ACTION_DIMENSION),
            "applied_target": (applied_vec, ACTION_DIMENSION),
            "joint_position": (joint_position, JOINT_STATE_DIMENSION),
            "joint_velocity": (joint_velocity, JOINT_STATE_DIMENSION),
            "next_joint_position": (next_joint_position, JOINT_STATE_DIMENSION),
        }
        sample: dict[str, Any] = {
            "sim_time_ns": int(sim_time_ns),
            "step_index": int(actual_step_index),
            "command_seq": int(command_seq),
        }
        for name, (values, expected_size) in vectors.items():
            if len(values) != expected_size:
                raise ValueError(f"{name} must contain {expected_size} values")
            converted = [float(value) for value in values]
            if not all(math.isfinite(value) for value in converted):
                raise ValueError(f"{name} values must be finite")
            sample[name] = converted
        if lio_pose is not None and lio_velocity is not None:
            for name, values in (("lio_pose", lio_pose), ("lio_velocity", lio_velocity)):
                if len(values) != 3:
                    raise ValueError(f"{name} must contain 3 values")
                converted = [float(value) for value in values]
                if not all(math.isfinite(value) for value in converted):
                    raise ValueError(f"{name} values must be finite")
                sample[name] = converted
        if mimic_annotation is not None:
            sample["mimic_annotation"] = self._validate_mimic_annotation(mimic_annotation)
        elif self._mimic_num_objects is not None:
            raise ValueError(
                "mimic_annotation was enabled for this episode; every step must provide it"
            )
        sample["raw_command"] = raw_command or {}
        encoded_frames: dict[str, bytes] = {}
        if camera_frames is not None:
            for camera_name, encoded_frame in camera_frames.items():
                if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", camera_name):
                    raise ValueError(f"invalid camera name: {camera_name!r}")
                frame = bytes(encoded_frame)
                if not frame:
                    raise ValueError(f"camera frame {camera_name!r} cannot be empty")
                encoded_frames[camera_name] = frame

        # 所有验证通过，更新状态
        self._samples.append(sample)
        self._last_sim_time_ns = sim_time_ns
        self._last_step_index = actual_step_index

        if camera_frames is not None:
            for camera_name, frame in encoded_frames.items():
                self._camera_frames.setdefault(camera_name, []).append((int(sim_time_ns), int(actual_step_index), frame))

    def _validate_mimic_annotation(self, annotation: dict[str, Any]) -> dict[str, Any]:
        """Validate one per-step Mimic annotation sample against the episode layout."""

        if not isinstance(annotation, dict):
            raise TypeError("mimic_annotation must be a dictionary")
        try:
            object_pose = annotation["object_pose"]
            eef_pose = annotation["eef_pose"]
            target_eef_pose = annotation["target_eef_pose"]
            subtask_signal = annotation["subtask_signal"]
            subtask_flags = annotation["subtask_term_flags"]
            object_names = annotation["object_names"]
            subtask_labels = annotation["subtask_labels"]
        except KeyError as error:
            raise ValueError(f"mimic_annotation is missing field: {error.args[0]}") from error

        if self._mimic_num_objects is None:
            names = tuple(str(name) for name in object_names)
            if not names:
                raise ValueError("mimic_annotation requires at least one tracked object")
            labels = tuple(str(label) for label in subtask_labels)
            if not labels:
                raise ValueError("mimic_annotation requires subtask labels")
            self._mimic_object_names = names
            self._mimic_subtask_labels = labels
            self._mimic_num_objects = len(names)
            self._mimic_num_labels = len(labels)
        elif (
            tuple(str(name) for name in object_names) != self._mimic_object_names
            or tuple(str(label) for label in subtask_labels) != self._mimic_subtask_labels
        ):
            raise ValueError("mimic_annotation object/subtask names must stay constant within one episode")

        num_objects = int(self._mimic_num_objects or 0)
        num_labels = int(self._mimic_num_labels or 0)
        if len(object_pose) != num_objects:
            raise ValueError(
                f"mimic object_pose must contain {num_objects} object poses per step"
            )
        pose_rows: list[list[list[float]]] = []
        for matrix in object_pose:
            rows = [[float(value) for value in row] for row in matrix]
            if len(rows) != 4 or any(len(row) != 4 for row in rows):
                raise ValueError("each mimic object_pose entry must be a 4x4 homogeneous transform")
            if not all(math.isfinite(value) for row in rows for value in row):
                raise ValueError("mimic object_pose values must be finite")
            pose_rows.append(rows)
        eef_rows: list[list[list[float]]] = []
        target_rows: list[list[list[float]]] = []
        for name, poses in (("eef_pose", eef_pose), ("target_eef_pose", target_eef_pose)):
            if len(poses) != MIMIC_ANNOTATION_EEF_COUNT:
                raise ValueError(f"mimic {name} must contain [left, right] pose matrices")
            converted = []
            for matrix in poses:
                rows = [[float(value) for value in row] for row in matrix]
                if len(rows) != 4 or any(len(row) != 4 for row in rows):
                    raise ValueError(f"each mimic {name} entry must be a 4x4 homogeneous transform")
                if not all(math.isfinite(value) for row in rows for value in row):
                    raise ValueError(f"mimic {name} values must be finite")
                converted.append(rows)
            if name == "eef_pose":
                eef_rows = converted
            else:
                target_rows = converted
        signal = int(subtask_signal)
        flags = [int(value) for value in subtask_flags]
        if len(flags) != num_labels:
            raise ValueError(f"mimic subtask_term_flags must contain {num_labels} values per step")
        if any(value not in (0, 1) for value in flags):
            raise ValueError("mimic subtask_term_flags must be 0 or 1")
        return {
            "object_pose": pose_rows,
            "eef_pose": eef_rows,
            "target_eef_pose": target_rows,
            "subtask_signal": signal,
            "subtask_term_flags": flags,
        }

    def _write_mimic_annotation_group(self, dataset: Any, np: Any) -> None:
        """Persist the mimic_annotation/ group (sibling of lio_pose/lio_velocity)."""

        import h5py

        num_objects = int(self._mimic_num_objects or 0)
        num_labels = int(self._mimic_num_labels or 0)
        group = dataset.create_group("mimic_annotation")
        group.attrs["version"] = 1
        group.create_dataset(
            "object_names",
            data=np.asarray(list(self._mimic_object_names or ()), dtype=h5py.string_dtype("utf-8")),
        )
        group.create_dataset(
            "subtask_labels",
            data=np.asarray(list(self._mimic_subtask_labels or ()), dtype=h5py.string_dtype("utf-8")),
        )
        object_pose = np.asarray(
            [sample["mimic_annotation"]["object_pose"] for sample in self._samples],
            dtype="float32",
        )
        group.create_dataset(
            "object_pose",
            data=object_pose.reshape((-1, num_objects, *MIMIC_ANNOTATION_POSE_SHAPE)),
            compression="gzip",
        )
        for field in ("eef_pose", "target_eef_pose"):
            values = np.asarray(
                [sample["mimic_annotation"][field] for sample in self._samples],
                dtype="float32",
            )
            group.create_dataset(
                field,
                data=values.reshape((-1, MIMIC_ANNOTATION_EEF_COUNT, *MIMIC_ANNOTATION_POSE_SHAPE)),
                compression="gzip",
            )
        group.create_dataset(
            "subtask_signal",
            data=np.asarray(
                [sample["mimic_annotation"]["subtask_signal"] for sample in self._samples],
                dtype="int32",
            ),
        )
        subtask_flags = np.asarray(
            [sample["mimic_annotation"]["subtask_term_flags"] for sample in self._samples],
            dtype="uint8",
        )
        group.create_dataset(
            "subtask_term_flags",
            data=subtask_flags.reshape((-1, num_labels)),
            compression="gzip",
        )

    def save(self, output_path: str | Path, *, metadata: dict[str, Any]) -> Path:
        if self.state is not EpisodeState.RECORDING:
            raise RuntimeError("no episode is recording")
        if not self._samples:
            raise RuntimeError("cannot save an empty episode")

        path = Path(output_path)
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing teleoperation episode: {path}")
        import h5py
        import numpy as np

        # 检查是否有完整的溯源字段，决定是否标记为新schema
        required_provenance_keys = {"episode_id", "config_hash", "contract_hash", "random_seed", "input_source"}
        has_complete_provenance = required_provenance_keys.issubset(metadata.keys())

        enriched_metadata = dict(metadata)
        if has_complete_provenance:
            # 只有完整溯源才添加schema版本，标记为可训练的新格式
            enriched_metadata["episode_schema_version"] = "1.0"
        if self._mimic_object_names is not None:
            enriched_metadata["mimic_annotation"] = {
                "enabled": True,
                "object_names": list(self._mimic_object_names),
                "subtask_labels": list(self._mimic_subtask_labels or ()),
                "pose_layout": "4x4 homogeneous transform matrix (world frame)",
                "eef_order": ["left", "right"],
                "version": 1,
            }
        metadata_json = json.dumps(enriched_metadata, sort_keys=True)

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with h5py.File(temporary_path, "w") as dataset:
                dataset.attrs["format"] = CURRENT_EPISODE_FORMAT
                dataset.attrs["metadata_json"] = metadata_json
                dataset.create_dataset("sim_time_ns", data=np.asarray([sample["sim_time_ns"] for sample in self._samples], dtype="uint64"))
                dataset.create_dataset("step_index", data=np.asarray([sample["step_index"] for sample in self._samples], dtype="int64"))
                dataset.create_dataset("command_seq", data=np.asarray([sample["command_seq"] for sample in self._samples], dtype="uint64"))

                # 保存新的动作语义字段
                for field, width in (
                    ("operator_command", ACTION_DIMENSION),
                    ("applied_target", ACTION_DIMENSION),
                    ("joint_position", JOINT_STATE_DIMENSION),
                    ("joint_velocity", JOINT_STATE_DIMENSION),
                    ("next_joint_position", JOINT_STATE_DIMENSION),
                ):
                    values = np.asarray([sample[field] for sample in self._samples], dtype="float32")
                    dataset.create_dataset(field, data=values.reshape((-1, width)), compression="gzip")

                if "lio_pose" in self._samples[0]:
                    for field in ("lio_pose", "lio_velocity"):
                        values = np.asarray([sample[field] for sample in self._samples], dtype="float32")
                        dataset.create_dataset(field, data=values.reshape((-1, 3)), compression="gzip")

                if "mimic_annotation" in self._samples[0]:
                    self._write_mimic_annotation_group(dataset, np)

                # 保留旧的action字段以兼容现有代码（使用applied_target）
                action_values = np.asarray([sample["applied_target"] for sample in self._samples], dtype="float32")
                dataset.create_dataset("action", data=action_values.reshape((-1, ACTION_DIMENSION)), compression="gzip")

                raw = [json.dumps(sample["raw_command"], sort_keys=True) for sample in self._samples]
                dataset.create_dataset("raw_command_json", data=np.asarray(raw, dtype=h5py.string_dtype("utf-8")))
                if self._camera_frames:
                    cameras = dataset.create_group("camera_frames")
                    jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
                    for camera_name, frames in sorted(self._camera_frames.items()):
                        camera = cameras.create_group(camera_name)
                        camera.create_dataset(
                            "sim_time_ns",
                            data=np.asarray([timestamp for timestamp, _step_idx, _frame in frames], dtype="uint64"),
                        )
                        camera.create_dataset(
                            "step_index",
                            data=np.asarray([step_idx for _timestamp, step_idx, _frame in frames], dtype="int64"),
                        )
                        jpeg = camera.create_dataset("jpeg", (len(frames),), dtype=jpeg_dtype)
                        for index, (_timestamp, _step_idx, frame) in enumerate(frames):
                            jpeg[index] = np.frombuffer(frame, dtype=np.uint8)
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self.discard()
        return path

    def discard(self) -> None:
        self._samples.clear()
        self._camera_frames.clear()
        self._last_sim_time_ns = None
        self._last_step_index = None
        self._mimic_object_names = None
        self._mimic_subtask_labels = None
        self._mimic_num_objects = None
        self._mimic_num_labels = None
        self.state = EpisodeState.IDLE
