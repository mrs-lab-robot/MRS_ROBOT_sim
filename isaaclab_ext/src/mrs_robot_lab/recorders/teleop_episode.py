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

    @property
    def num_steps(self) -> int:
        return len(self._samples)

    def start(self) -> None:
        if self.state is EpisodeState.RECORDING:
            raise RuntimeError("an episode is already recording")
        self._samples.clear()
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
        self.state = EpisodeState.IDLE
