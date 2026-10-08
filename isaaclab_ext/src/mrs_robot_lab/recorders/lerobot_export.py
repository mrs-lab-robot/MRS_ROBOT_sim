"""Convert one timestamped Arena teleoperation episode to LeRobot frames."""

from __future__ import annotations

import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from mrs_robot_lab.assets.robot_interface import (
    ACTION_DIMENSION,
    ACTION_JOINTS_BY_TERM,
    ACTION_SLICES,
    HEAD_JOINTS,
    JOINT_STATE_NAMES,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LIFT_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
)
from mrs_robot_lab.recorders.episode_schema import SUPPORTED_EPISODE_FORMATS


LEROBOT_JOINT_ORDER = (
    *LEFT_ARM_JOINTS,
    *LEFT_GRIPPER_JOINTS,
    *RIGHT_ARM_JOINTS,
    *RIGHT_GRIPPER_JOINTS,
    *HEAD_JOINTS,
    *LIFT_JOINTS,
)
LEROBOT_ACTION_NAMES = (
    *(f"{name}.pos" for name in LEROBOT_JOINT_ORDER),
    "chassis.kin_vx",
    "chassis.kin_vy",
    "chassis.kin_wz",
)
LEROBOT_STATE_NAMES = tuple(f"{name}.pos" for name in LEROBOT_JOINT_ORDER)
_CAMERA_KEY_NAMES = {
    "base_d435": "base",
    "head_d435": "head",
    "left_wrist_d405": "left_wrist",
    "right_wrist_d405": "right_wrist",
}
_CURRENT_STAGING_FORMAT = "mrs_robot_capture_lerobot_staging_v1"
_LEGACY_STAGING_FORMATS = frozenset({"mrs_robot_arena_lerobot_staging_v1"})
_EPISODE_PROVENANCE_FIELDS = frozenset(
    {
        "config_hash",
        "contract_hash",
        "random_seed",
        "input_source",
        "teleop_mode",
        "embodiment_id",
        "scene_id",
        "task_id",
        "dataset_id",
        "episode_schema_version",
        "quality_status",
        "success",
        "success_reason",
        "failure_reason",
    }
)


def _latest_not_after_index(timestamps, target: int) -> int:
    import numpy as np

    # Causal resampling: select the newest observation at or before the target
    # clock. Nearest-neighbour can silently attach a future camera/state sample
    # to the current action when streams have different rates.
    index = int(np.searchsorted(timestamps, target, side="right")) - 1
    if index < 0:
        raise ValueError("causal resampling target precedes the first available sample")
    return min(index, len(timestamps) - 1)


def load_arena_hdf5_frames(
    source_path: str | Path,
    *,
    fps: int,
    task: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read current or legacy HDF5 episodes into LeRobot feature frames."""
    import h5py
    import numpy as np
    from PIL import Image

    path = Path(source_path).expanduser().resolve(strict=True)
    if not 1 <= int(fps) <= 60:
        raise ValueError("FPS must be between 1 and 60")
    task_name = str(task).strip()
    if not task_name:
        raise ValueError("task description cannot be empty")

    with h5py.File(path, "r") as episode:
        if episode.attrs.get("format", "") not in SUPPORTED_EPISODE_FORMATS:
            raise ValueError("unsupported MRS Robot HDF5 episode format")
        try:
            metadata = json.loads(episode.attrs["metadata_json"])
            joint_order = tuple(metadata["joint_state_order"])
            state_times = episode["sim_time_ns"][:].astype(np.int64)
            actions = episode["action"][:].astype(np.float32)
            applied_targets = (
                episode["applied_target"][:].astype(np.float32)
                if "applied_target" in episode
                else actions
            )
            joint_positions = episode["joint_position"][:].astype(np.float32)
            next_joint_positions = episode["next_joint_position"][:].astype(np.float32)
            camera_group = episode["camera_frames"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Arena HDF5 episode is missing required data: {error}") from error

        if len(joint_order) != len(JOINT_STATE_NAMES) or set(joint_order) != set(JOINT_STATE_NAMES):
            raise ValueError("HDF5 joint_state_order does not match the OpenFlex embodiment contract")
        sample_count = len(state_times)
        if (
            sample_count == 0
            or actions.shape != (sample_count, ACTION_DIMENSION)
            or applied_targets.shape != (sample_count, ACTION_DIMENSION)
            or joint_positions.shape != (sample_count, len(joint_order))
            or next_joint_positions.shape != (sample_count, len(joint_order))
        ):
            raise ValueError("Arena HDF5 state/action arrays have inconsistent dimensions")
        if np.any(np.diff(state_times) <= 0):
            if sample_count != 1:
                raise ValueError("Arena HDF5 simulation timestamps must be strictly increasing")

        cameras = {}
        for camera_name in sorted(camera_group.keys()):
            if camera_name not in _CAMERA_KEY_NAMES:
                raise ValueError(f"unsupported OpenFlex camera in Arena episode: {camera_name}")
            camera = camera_group[camera_name]
            timestamps = camera["sim_time_ns"][:].astype(np.int64)
            jpeg = camera["jpeg"]
            if not len(timestamps) or len(timestamps) != len(jpeg):
                raise ValueError(f"Arena camera {camera_name} has no timestamp-aligned frames")
            if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"Arena camera {camera_name} timestamps must be strictly increasing")
            cameras[camera_name] = (timestamps, jpeg)
        if not cameras:
            raise ValueError("Arena HDF5 episode has no RGB camera frames; training export requires images")

        first_timestamp = max(
            int(state_times[0]),
            *(int(timestamps[0]) for timestamps, _jpeg in cameras.values()),
        )
        last_timestamp = min(
            int(state_times[-1]),
            *(int(timestamps[-1]) for timestamps, _jpeg in cameras.values()),
        )
        if last_timestamp < first_timestamp:
            raise ValueError("Arena camera frames do not overlap the recorded robot state timeline")
        interval_ns = 1_000_000_000 / int(fps)
        frame_count = int((last_timestamp - first_timestamp) / interval_ns) + 1
        target_times = [first_timestamp + round(index * interval_ns) for index in range(frame_count)]
        state_indices = [_latest_not_after_index(state_times, timestamp) for timestamp in target_times]
        camera_indices = {
            camera_name: [_latest_not_after_index(timestamps, timestamp) for timestamp in target_times]
            for camera_name, (timestamps, _jpeg) in cameras.items()
        }

        images_by_camera = {}
        for camera_name, (_timestamps, jpeg_frames) in cameras.items():
            selected = []
            decoded_cache = {}
            for frame_index in camera_indices[camera_name]:
                if frame_index not in decoded_cache:
                    raw = np.asarray(jpeg_frames[frame_index], dtype=np.uint8).tobytes()
                    try:
                        with Image.open(BytesIO(raw)) as image:
                            decoded_cache[frame_index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    except (OSError, ValueError) as error:
                        raise ValueError(f"Arena camera {camera_name} contains an invalid JPEG frame") from error
                selected.append(decoded_cache[frame_index])
            images_by_camera[camera_name] = selected

        first_images = {
            _CAMERA_KEY_NAMES[camera_name]: frames[0]
            for camera_name, frames in images_by_camera.items()
        }
        features: dict[str, dict[str, Any]] = {
            "action": {
                "dtype": "float32",
                "shape": (len(LEROBOT_ACTION_NAMES),),
                "names": list(LEROBOT_ACTION_NAMES),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(LEROBOT_STATE_NAMES),),
                "names": list(LEROBOT_STATE_NAMES),
            },
        }
        for camera_name, image in first_images.items():
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": tuple(image.shape),
                "names": ["height", "width", "channels"],
            }

        frames = []
        for sample_index, timestamp in zip(state_indices, target_times, strict=True):
            state_by_name = dict(zip(joint_order, joint_positions[sample_index], strict=True))
            target_by_joint = {}
            for term_name, joint_names in ACTION_JOINTS_BY_TERM.items():
                term_start, term_width = ACTION_SLICES[term_name]
                if term_width != len(joint_names):
                    raise ValueError(f"OpenFlex action term {term_name} does not match its joint contract")
                target_by_joint.update(
                    {
                        joint_name: applied_targets[sample_index, term_start + joint_index]
                        for joint_index, joint_name in enumerate(joint_names)
                    }
                )
            try:
                action_values = [target_by_joint[name] for name in LEROBOT_JOINT_ORDER]
            except KeyError as error:
                raise ValueError(f"OpenFlex action contract is missing joint target {error.args[0]}") from error
            base_start, base_width = ACTION_SLICES["base_twist_action"]
            if base_width != 3:
                raise ValueError("OpenFlex base action must contain vx, vy and wz")
            action_values.extend(applied_targets[sample_index, base_start:base_start + base_width])
            frame = {
                "action": np.asarray(action_values, dtype=np.float32),
                "observation.state": np.asarray(
                    [state_by_name[name] for name in LEROBOT_JOINT_ORDER], dtype=np.float32
                ),
                "timestamp": (timestamp - first_timestamp) / 1_000_000_000.0,
                "task": task_name,
            }
            for camera_name, images in images_by_camera.items():
                frame[f"observation.images.{_CAMERA_KEY_NAMES[camera_name]}"] = images[len(frames)]
            frames.append(frame)

    return frames, features


def export_arena_hdf5_to_lerobot(
    source_path: str | Path,
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int,
    task: str,
) -> Path:
    """Create one LeRobot dataset episode from a saved Arena HDF5 episode."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    frames, features = load_arena_hdf5_frames(source_path, fps=fps, task=task)
    return _write_lerobot_dataset(
        frames,
        features,
        output_root,
        repo_id,
        fps=fps,
    )


def append_hdf5_episode_to_lerobot(
    source_path: str | Path,
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int,
    task: str,
    dataset_class=None,
) -> Path:
    """Create or append one QC-approved HDF5 episode to a LeRobot dataset.

    A per-dataset process lock and atomic episode ledger make retries idempotent.
    Legacy Arena-format HDF5 remains readable, but newly recorded episodes use
    the current MRS capture format.
    """
    import h5py

    repo = str(repo_id).strip()
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?",
        repo,
    ):
        raise ValueError("repo_id must be a safe repository or namespace/repository ID")
    if not 1 <= int(fps) <= 60:
        raise ValueError("FPS must be between 1 and 60")

    source = Path(source_path).expanduser().resolve(strict=True)
    with h5py.File(source, "r") as episode:
        raw_metadata = episode.attrs.get("metadata_json", "{}")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8", errors="strict")
        try:
            episode_metadata = json.loads(str(raw_metadata))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid episode metadata_json: {error}") from error
    if not isinstance(episode_metadata, dict):
        raise ValueError("episode metadata_json must contain a JSON object")
    episode_id = str(episode_metadata.get("episode_id", "")).strip()
    if not episode_id or len(episode_id) > 128:
        raise ValueError("episode metadata must contain a non-empty episode_id")

    frames, features = load_arena_hdf5_frames(source, fps=int(fps), task=task)
    source_digest = _sha256_file(source)
    return _append_lerobot_episode_frames(
        frames,
        features,
        output_root,
        repo,
        fps=int(fps),
        episode_id=episode_id,
        source_identity=str(source),
        source_sha256=source_digest,
        source_reference=source.name,
        provenance=_episode_provenance(episode_metadata, episode_id),
        dataset_class=dataset_class,
    )


def _append_lerobot_episode_frames(
    frames: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int,
    episode_id: str,
    source_identity: str,
    source_sha256: str,
    source_reference: str,
    provenance: dict[str, Any],
    dataset_class=None,
) -> Path:
    output = Path(output_root).expanduser().resolve()
    destination = output.joinpath(*repo_id.split("/"))
    output.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.mrs-export.lock"
    ledger_path = destination.parent / f".{destination.name}.mrs-episode-ledger.json"

    if dataset_class is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset_class = LeRobotDataset

    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        ledger = _read_episode_export_ledger(ledger_path)
        dataset = _open_or_create_lerobot_dataset(
            dataset_class,
            repo_id,
            destination,
            fps,
            features,
        )
        episode_count = int(dataset.meta.total_episodes)
        export_record = ledger.get(episode_id)
        if isinstance(export_record, dict) and export_record.get("state") == "complete":
            _validate_episode_export_identity(
                export_record,
                source_identity=source_identity,
                source_sha256=source_sha256,
            )
            ledger[episode_id] = _complete_episode_record(
                export_record.get("episode_index", -1),
                source_identity=source_identity,
                source_sha256=source_sha256,
                source_reference=source_reference,
                provenance=provenance,
            )
            _write_episode_export_ledger(ledger_path, ledger)
            _write_portable_episode_manifest(destination, repo_id, ledger)
            return destination
        if isinstance(export_record, dict) and export_record.get("state") == "pending":
            _validate_episode_export_identity(
                export_record,
                source_identity=source_identity,
                source_sha256=source_sha256,
            )
            base_count = int(export_record.get("base_episode_count", -1))
            if episode_count == base_count + 1:
                ledger[episode_id] = _complete_episode_record(
                    base_count,
                    source_identity=source_identity,
                    source_sha256=source_sha256,
                    source_reference=source_reference,
                    provenance=provenance,
                )
                _write_episode_export_ledger(ledger_path, ledger)
                _write_portable_episode_manifest(destination, repo_id, ledger)
                return destination
            if episode_count != base_count:
                raise RuntimeError(
                    "LeRobot dataset episode count changed during an interrupted export; "
                    "manual integrity review is required"
                )
        else:
            base_count = episode_count

        ledger[episode_id] = {
            "state": "pending",
            "base_episode_count": base_count,
            "source": source_identity,
            "source_sha256": source_sha256,
            "source_reference": source_reference,
            "provenance": provenance,
        }
        _write_episode_export_ledger(ledger_path, ledger)

        for frame in frames:
            frame_to_write = dict(frame)
            frame_to_write.pop("timestamp", None)
            dataset.add_frame(frame_to_write)
        dataset.save_episode()
        new_count = int(dataset.meta.total_episodes)
        if new_count != base_count + 1:
            raise RuntimeError(
                f"LeRobot save did not commit exactly one episode: {base_count} -> {new_count}"
            )
        ledger[episode_id] = _complete_episode_record(
            base_count,
            source_identity=source_identity,
            source_sha256=source_sha256,
            source_reference=source_reference,
            provenance=provenance,
        )
        _write_episode_export_ledger(ledger_path, ledger)
        _write_portable_episode_manifest(destination, repo_id, ledger)
    return destination


def _episode_provenance(metadata: dict[str, Any], episode_id: str) -> dict[str, Any]:
    """Select portable, non-path episode metadata from the HDF5 manifest."""
    provenance = {
        key: value
        for key, value in metadata.items()
        if key in _EPISODE_PROVENANCE_FIELDS
        and (value is None or isinstance(value, (str, int, float, bool)))
    }
    provenance["episode_id"] = episode_id
    return provenance


def _complete_episode_record(
    episode_index: int,
    *,
    source_identity: str,
    source_sha256: str,
    source_reference: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state": "complete",
        "episode_index": int(episode_index),
        "source": source_identity,
        "source_sha256": source_sha256,
        "source_reference": _portable_source_reference(source_reference),
        "provenance": provenance,
    }


def _portable_source_reference(value: str) -> str:
    """Keep an HDF5 locator relative so dataset metadata does not leak host paths."""
    normalized = str(value).replace("\\", "/").strip()
    candidate = PurePosixPath(normalized)
    is_windows_absolute = bool(re.match(r"^[A-Za-z]:/", normalized))
    if candidate.is_absolute() or is_windows_absolute or ".." in candidate.parts:
        return ""
    reference = candidate.as_posix()
    return "" if reference in {"", "."} else reference


def _validate_episode_export_identity(
    record: dict[str, Any], *, source_identity: str, source_sha256: str
) -> None:
    recorded_digest = str(record.get("source_sha256", ""))
    if recorded_digest:
        if recorded_digest != source_sha256:
            raise ValueError("episode ID was already exported with different HDF5 content")
        return
    # Upgrade old ledgers that stored a source path/hash but no explicit digest.
    legacy_identity = str(record.get("source", ""))
    if legacy_identity not in {source_identity, source_sha256}:
        raise ValueError("episode ID was already exported with different HDF5 content")


def _write_portable_episode_manifest(
    destination: Path,
    dataset_id: str,
    ledger: dict[str, dict[str, Any]],
) -> None:
    episodes = {}
    for episode_id, record in ledger.items():
        if record.get("state") != "complete":
            continue
        provenance = record.get("provenance", {})
        if not isinstance(provenance, dict):
            provenance = {}
        episodes[episode_id] = {
            "episode_index": int(record.get("episode_index", -1)),
            "source_sha256": str(record.get("source_sha256", "")),
            "source_reference": _portable_source_reference(
                str(record.get("source_reference", ""))
            ),
            "provenance": _episode_provenance(provenance, episode_id),
        }
    _write_json_atomically(
        destination / "meta" / "mrs_episode_manifest.json",
        {"schema_version": 1, "dataset_id": dataset_id, "episodes": episodes},
    )


def _open_or_create_lerobot_dataset(
    dataset_class,
    repo_id: str,
    destination: Path,
    fps: int,
    features: dict[str, dict[str, Any]],
):
    if destination.exists():
        if not (destination / "meta" / "info.json").is_file():
            raise FileExistsError(
                f"LeRobot destination exists but is not a complete dataset: {destination}"
            )
        dataset = dataset_class(repo_id, root=destination, download_videos=False)
        if int(dataset.fps) != int(fps):
            raise ValueError(
                f"LeRobot dataset FPS is {dataset.fps}; requested export FPS is {fps}"
            )
        if _feature_signature(dataset.features) != _feature_signature(features):
            raise ValueError("LeRobot dataset feature schema does not match this episode")
        return dataset

    destination.parent.mkdir(parents=True, exist_ok=True)
    return dataset_class.create(
        repo_id,
        int(fps),
        features=features,
        root=destination,
        robot_type="openarmx_follower_ros2",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )


def _feature_signature(features: dict[str, dict[str, Any]]) -> dict[str, tuple]:
    signature = {}
    for name, feature in features.items():
        shape = tuple(feature.get("shape", ()))
        names = tuple(feature.get("names", ()))
        signature[name] = (str(feature.get("dtype", "")), shape, names)
    return signature


def _read_episode_export_ledger(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read LeRobot episode export ledger {path}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(item, dict) for item in value.values()):
        raise ValueError(f"LeRobot episode export ledger has an invalid structure: {path}")
    return value


def _write_episode_export_ledger(path: Path, ledger: dict[str, dict[str, Any]]) -> None:
    _write_json_atomically(path, ledger)


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=False, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_arena_hdf5_for_lerobot(
    source_path: str | Path,
    prepared_directory: str | Path,
    *,
    fps: int,
    task: str,
    source_reference: str | None = None,
) -> Path:
    """Extract an HDF5 episode into a compact, Python-version-neutral staging directory."""
    import h5py
    import numpy as np
    from PIL import Image

    source = Path(source_path).expanduser().resolve(strict=True)
    frames, features = load_arena_hdf5_frames(source, fps=fps, task=task)
    with h5py.File(source, "r") as episode:
        raw_metadata = episode.attrs.get("metadata_json", "{}")
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8", errors="strict")
        try:
            metadata = json.loads(str(raw_metadata))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid episode metadata_json: {error}") from error
    if not isinstance(metadata, dict):
        raise ValueError("episode metadata_json must contain a JSON object")
    source_digest = _sha256_file(source)
    episode_id = str(metadata.get("episode_id", "")).strip()
    if not episode_id:
        # Legacy source episodes may not carry an ID; content identity keeps
        # their staging/export retries deterministic without mutating the source.
        episode_id = f"legacy-{source_digest}"
    prepared = Path(prepared_directory).expanduser().resolve()
    prepared.mkdir(parents=True, exist_ok=False)
    np.save(
        prepared / "action.npy",
        np.stack([frame["action"] for frame in frames]).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        prepared / "observation_state.npy",
        np.stack([frame["observation.state"] for frame in frames]).astype(np.float32),
        allow_pickle=False,
    )
    np.save(
        prepared / "timestamp.npy",
        np.asarray([frame["timestamp"] for frame in frames], dtype=np.float64),
        allow_pickle=False,
    )

    cameras = []
    for feature_name in sorted(features):
        prefix = "observation.images."
        if not feature_name.startswith(prefix):
            continue
        camera_name = feature_name.removeprefix(prefix)
        camera_directory = prepared / "images" / camera_name
        camera_directory.mkdir(parents=True)
        cameras.append(camera_name)
        for frame_index, frame in enumerate(frames):
            Image.fromarray(frame[feature_name]).save(
                camera_directory / f"{frame_index:08d}.jpg",
                format="JPEG",
                quality=95,
                subsampling=0,
            )

    manifest = {
        "format": _CURRENT_STAGING_FORMAT,
        "episode_id": episode_id,
        "source_sha256": source_digest,
        "fps": int(fps),
        "task": str(task).strip(),
        "frame_count": len(frames),
        "cameras": cameras,
        "source_reference": _portable_source_reference(source_reference or source.name),
        "provenance": _episode_provenance(metadata, episode_id),
    }
    (prepared / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return prepared


def append_prepared_episode_to_lerobot(
    prepared_directory: str | Path,
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int,
    task: str,
    dataset_class=None,
) -> Path:
    """Append Python-version-neutral staging data without requiring HDF5 support."""
    import numpy as np
    from PIL import Image

    repo = str(repo_id).strip()
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?",
        repo,
    ):
        raise ValueError("repo_id must be a safe repository or namespace/repository ID")
    prepared = Path(prepared_directory).expanduser().resolve(strict=True)
    try:
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        cameras = tuple(str(name) for name in manifest["cameras"])
        frame_count = int(manifest["frame_count"])
        staged_fps = int(manifest["fps"])
        staged_task = str(manifest["task"])
        episode_id = str(manifest["episode_id"]).strip()
        raw_provenance = manifest.get("provenance", {})
        source_digest = str(manifest.get("source_sha256", ""))
        source_reference = _portable_source_reference(
            str(manifest.get("source_reference", ""))
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid MRS Robot LeRobot staging manifest: {error}") from error
    if manifest.get("format") not in {_CURRENT_STAGING_FORMAT, *_LEGACY_STAGING_FORMATS}:
        raise ValueError("unsupported MRS Robot LeRobot staging format")
    if staged_fps != int(fps) or staged_task != str(task).strip():
        raise ValueError("LeRobot export settings do not match the prepared episode")
    if not episode_id or len(episode_id) > 128:
        raise ValueError("prepared episode must contain a valid episode_id")
    if not isinstance(raw_provenance, dict):
        raise ValueError("prepared episode provenance must be a JSON object")
    if source_digest and not re.fullmatch(r"[0-9a-f]{64}", source_digest):
        raise ValueError("prepared episode source_sha256 must be a lowercase SHA-256 digest")
    if frame_count <= 0 or not cameras or any(name not in _CAMERA_KEY_NAMES.values() for name in cameras):
        raise ValueError("prepared episode has invalid frame or camera metadata")

    actions = np.load(prepared / "action.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(prepared / "observation_state.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(prepared / "timestamp.npy", mmap_mode="r", allow_pickle=False)
    if (
        actions.shape != (frame_count, len(LEROBOT_ACTION_NAMES))
        or states.shape != (frame_count, len(LEROBOT_STATE_NAMES))
        or timestamps.shape != (frame_count,)
    ):
        raise ValueError("prepared action/state arrays have inconsistent dimensions")
    if not np.isfinite(actions).all() or not np.isfinite(states).all() or not np.isfinite(timestamps).all():
        raise ValueError("prepared episode arrays contain non-finite values")
    if frame_count > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("prepared episode timestamps must be strictly increasing")

    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (len(LEROBOT_ACTION_NAMES),),
            "names": list(LEROBOT_ACTION_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(LEROBOT_STATE_NAMES),),
            "names": list(LEROBOT_STATE_NAMES),
        },
    }
    images_by_camera: dict[str, list[Any]] = {}
    for camera in cameras:
        image_directory = prepared / "images" / camera
        images = []
        for frame_index in range(frame_count):
            image_path = image_directory / f"{frame_index:08d}.jpg"
            try:
                with Image.open(image_path) as image:
                    images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"prepared camera {camera} frame {frame_index} is invalid"
                ) from error
        first_image = images[0]
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": tuple(first_image.shape),
            "names": ["height", "width", "channels"],
        }
        images_by_camera[camera] = images

    frames: list[dict[str, Any]] = []
    for frame_index in range(frame_count):
        frame = {
            "action": np.asarray(actions[frame_index], dtype=np.float32),
            "observation.state": np.asarray(states[frame_index], dtype=np.float32),
            "timestamp": float(timestamps[frame_index]),
            "task": staged_task,
        }
        for camera, images in images_by_camera.items():
            frame[f"observation.images.{camera}"] = images[frame_index]
        frames.append(frame)

    source_identity = source_digest or episode_id
    return _append_lerobot_episode_frames(
        frames,
        features,
        output_root,
        repo,
        fps=int(fps),
        episode_id=episode_id,
        source_identity=source_identity,
        source_sha256=source_digest,
        source_reference=source_reference,
        provenance=_episode_provenance(raw_provenance, episode_id),
        dataset_class=dataset_class,
    )


def export_prepared_episode_to_lerobot(
    prepared_directory: str | Path,
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int,
    task: str,
) -> Path:
    """Write an extracted episode without requiring HDF5 in the LeRobot Python environment."""
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    prepared = Path(prepared_directory).expanduser().resolve(strict=True)
    try:
        manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
        cameras = tuple(str(name) for name in manifest["cameras"])
        frame_count = int(manifest["frame_count"])
        staged_fps = int(manifest["fps"])
        staged_task = str(manifest["task"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Arena LeRobot staging manifest: {error}") from error
    if manifest.get("format") not in {_CURRENT_STAGING_FORMAT, *_LEGACY_STAGING_FORMATS}:
        raise ValueError("unsupported MRS Robot LeRobot staging format")
    if staged_fps != int(fps) or staged_task != str(task).strip():
        raise ValueError("LeRobot export settings do not match the prepared Arena episode")
    if frame_count <= 0 or not cameras or any(name not in _CAMERA_KEY_NAMES.values() for name in cameras):
        raise ValueError("prepared Arena episode has invalid frame or camera metadata")

    actions = np.load(prepared / "action.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(prepared / "observation_state.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(prepared / "timestamp.npy", mmap_mode="r", allow_pickle=False)
    if (
        actions.shape != (frame_count, len(LEROBOT_ACTION_NAMES))
        or states.shape != (frame_count, len(LEROBOT_STATE_NAMES))
        or timestamps.shape != (frame_count,)
    ):
        raise ValueError("prepared Arena action/state arrays have inconsistent dimensions")

    image_paths = {
        camera: prepared / "images" / camera / f"{0:08d}.jpg"
        for camera in cameras
    }
    features: dict[str, dict[str, Any]] = {
        "action": {
            "dtype": "float32",
            "shape": (len(LEROBOT_ACTION_NAMES),),
            "names": list(LEROBOT_ACTION_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(LEROBOT_STATE_NAMES),),
            "names": list(LEROBOT_STATE_NAMES),
        },
    }
    for camera, path in image_paths.items():
        try:
            with Image.open(path) as image:
                shape = tuple(np.asarray(image.convert("RGB")).shape)
        except (OSError, ValueError) as error:
            raise ValueError(f"prepared Arena camera {camera} is missing a valid first frame") from error
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": shape,
            "names": ["height", "width", "channels"],
        }

    root = Path(output_root).expanduser().resolve()
    destination = root.joinpath(*repo_id.split("/"))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing LeRobot dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id,
        staged_fps,
        features=features,
        root=destination,
        robot_type="openarmx_follower_ros2",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )
    for frame_index in range(frame_count):
        frame = {
            "action": np.asarray(actions[frame_index], dtype=np.float32),
            "observation.state": np.asarray(states[frame_index], dtype=np.float32),
            "task": staged_task,
        }
        for camera in cameras:
            image_path = prepared / "images" / camera / f"{frame_index:08d}.jpg"
            try:
                with Image.open(image_path) as image:
                    frame[f"observation.images.{camera}"] = np.asarray(
                        image.convert("RGB"), dtype=np.uint8
                    )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"prepared Arena camera {camera} frame {frame_index} is invalid"
                ) from error
        dataset.add_frame(frame)
    dataset.save_episode()
    info_path = destination / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"LeRobot export finished without dataset metadata: {info_path}")
    return destination


def _write_lerobot_dataset(
    frames: list[dict[str, Any]],
    features: dict[str, dict[str, Any]],
    output_root: str | Path,
    repo_id: str,
    *,
    fps: int,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(output_root).expanduser().resolve()
    destination = root.joinpath(*repo_id.split("/"))
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing LeRobot dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id,
        int(fps),
        features=features,
        root=destination,
        robot_type="openarmx_follower_ros2",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )
    for frame in frames:
        frame_to_write = dict(frame)
        frame_to_write.pop("timestamp", None)
        dataset.add_frame(frame_to_write)
    dataset.save_episode()
    info_path = destination / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"LeRobot export finished without dataset metadata: {info_path}")
    return destination


__all__ = [
    "LEROBOT_ACTION_NAMES",
    "LEROBOT_JOINT_ORDER",
    "LEROBOT_STATE_NAMES",
    "append_hdf5_episode_to_lerobot",
    "append_prepared_episode_to_lerobot",
    "export_arena_hdf5_to_lerobot",
    "export_prepared_episode_to_lerobot",
    "load_arena_hdf5_frames",
    "prepare_arena_hdf5_for_lerobot",
]
