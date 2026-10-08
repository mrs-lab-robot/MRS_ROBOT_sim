"""Read and replay validated MRS teleoperation episodes in an Isaac Lab task."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, TYPE_CHECKING

import h5py
import numpy as np

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.recorders.episode_schema import SUPPORTED_EPISODE_FORMATS

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


StateAdapter = Callable[[Any, Mapping[str, Any]], None]
StateExtractor = Callable[[Any], Mapping[str, Any]]
ActionAdapter = Callable[[np.ndarray], np.ndarray]


class EpisodeReplayValidator:
    """Replay the recorded executed target, and never infer state from policy obs.

    The HDF5 ``action`` dataset is retained for old readers, but the authoritative
    replay input is ``applied_target``. Legacy files remain readable and are
    explicitly marked non-trainable because their action meaning is ambiguous.
    """

    FORMAT = "mrs_robot_capture_v1"
    PROVENANCE_KEYS = (
        "episode_id",
        "config_hash",
        "contract_hash",
        "random_seed",
        "input_source",
        "task_id",
        "task_version",
        "scene_id",
        "initial_state",
        "data_quality",
    )

    def __init__(
        self,
        env: "DirectRLEnv",
        *,
        action_adapter: ActionAdapter | None = None,
        state_restorer: StateAdapter | None = None,
        state_extractor: StateExtractor | None = None,
    ) -> None:
        self.env = env
        self.device = getattr(env, "device", "cpu")
        self.action_adapter = action_adapter
        self.state_restorer = state_restorer
        self.state_extractor = state_extractor

    def load_episode_from_hdf5(self, hdf5_path: str | Path) -> dict[str, Any]:
        """Load one HDF5 episode and validate timing, dimensions, and provenance."""
        path = Path(hdf5_path).expanduser().resolve(strict=True)
        with h5py.File(path, "r") as file:
            format_name = file.attrs.get("format", "")
            if isinstance(format_name, bytes):
                format_name = format_name.decode("utf-8", errors="replace")
            if format_name not in SUPPORTED_EPISODE_FORMATS:
                raise ValueError(f"unsupported HDF5 episode format: {format_name!r}")

            metadata_raw = file.attrs.get("metadata_json", "{}")
            if isinstance(metadata_raw, bytes):
                metadata_raw = metadata_raw.decode("utf-8", errors="strict")
            try:
                metadata = json.loads(str(metadata_raw))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid episode metadata_json: {error}") from error
            if not isinstance(metadata, dict):
                raise ValueError("episode metadata_json must contain a JSON object")

            if "applied_target" in file:
                actions = np.asarray(file["applied_target"][:], dtype=np.float32)
                action_source = "applied_target"
                legacy = False
            elif "action" in file:
                actions = np.asarray(file["action"][:], dtype=np.float32)
                action_source = "legacy_action"
                legacy = True
            else:
                raise ValueError("HDF5文件中找不到applied_target或legacy action数据集")

            if actions.ndim != 2 or actions.shape[0] == 0:
                raise ValueError("episode must contain at least one 2D action row")
            if not np.isfinite(actions).all():
                raise ValueError("episode actions contain non-finite values")
            num_steps, action_width = actions.shape

            operator_commands = self._read_optional_matrix(file, "operator_command", num_steps)
            applied_targets = self._read_optional_matrix(file, "applied_target", num_steps)
            if operator_commands is not None and operator_commands.shape != actions.shape:
                raise ValueError("operator_command and applied_target dimensions must match")
            if applied_targets is not None and applied_targets.shape != actions.shape:
                raise ValueError("applied_target dimensions do not match replay actions")

            quality_errors: list[str] = []
            if legacy:
                quality_errors.append("legacy action has no trusted command/applied-target distinction")
            if operator_commands is None:
                quality_errors.append("operator_command is missing")

            if "action" in file and applied_targets is not None:
                compatibility_action = np.asarray(file["action"][:], dtype=np.float32)
                if compatibility_action.shape != applied_targets.shape or not np.array_equal(
                    compatibility_action, applied_targets
                ):
                    quality_errors.append("action compatibility alias differs from applied_target")

            timestamps = self._read_timeline(file, "sim_time_ns", num_steps, quality_errors)
            step_indices = self._read_timeline(file, "step_index", num_steps, quality_errors)
            command_sequences = self._read_timeline(file, "command_seq", num_steps, quality_errors)
            self._require_strictly_increasing(timestamps, "sim_time_ns")
            self._require_strictly_increasing(step_indices, "step_index")
            self._require_strictly_increasing(command_sequences, "command_seq")

            observations = {}
            if "observation" in file and isinstance(file["observation"], h5py.Group):
                observations = {
                    name: np.asarray(dataset[:])
                    for name, dataset in file["observation"].items()
                    if isinstance(dataset, h5py.Dataset)
                }

            states: dict[str, np.ndarray] = {}
            for field_name in ("joint_position", "joint_velocity", "next_joint_position"):
                if field_name in file:
                    states[field_name] = np.asarray(file[field_name][:], dtype=np.float32)
                    if states[field_name].ndim != 2 or states[field_name].shape[0] != num_steps:
                        raise ValueError(f"{field_name} must have one 2D row per action")
                    if not np.isfinite(states[field_name]).all():
                        raise ValueError(f"{field_name} contains non-finite values")
                else:
                    quality_errors.append(f"{field_name} is missing")

            joint_order = metadata.get("joint_state_order")
            if not isinstance(joint_order, list) or tuple(joint_order) != tuple(JOINT_STATE_NAMES):
                quality_errors.append("joint_state_order does not match the OpenFlex contract")
            for state_name in ("joint_position", "joint_velocity", "next_joint_position"):
                if state_name in states and states[state_name].shape[1] != len(JOINT_STATE_NAMES):
                    raise ValueError(f"{state_name} width must match the OpenFlex joint-state contract")

            quality_errors.extend(self._provenance_errors(metadata))
            if action_width != ACTION_DIMENSION:
                quality_errors.append(
                    f"action width {action_width} does not match OpenFlex contract width {ACTION_DIMENSION}"
                )

            return {
                "actions": actions,
                "action_source": action_source,
                "operator_commands": operator_commands,
                "applied_targets": applied_targets,
                "observations": observations,
                "states": states,
                "timestamps": timestamps,
                "step_indices": step_indices,
                "command_sequences": command_sequences,
                "metadata": metadata,
                "trainable": not quality_errors,
                "quality_errors": quality_errors,
                "source_path": str(path),
            }

    @staticmethod
    def _read_optional_matrix(file: h5py.File, name: str, num_steps: int) -> np.ndarray | None:
        if name not in file:
            return None
        value = np.asarray(file[name][:], dtype=np.float32)
        if value.ndim != 2 or value.shape[0] != num_steps:
            raise ValueError(f"{name} must have one 2D row per action")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
        return value

    @staticmethod
    def _read_timeline(file: h5py.File, name: str, num_steps: int, quality_errors: list[str]) -> np.ndarray:
        if name not in file:
            quality_errors.append(f"{name} is missing; synthetic indices are read-only")
            return np.arange(num_steps, dtype=np.int64)
        value = np.asarray(file[name][:])
        if value.ndim != 1 or len(value) != num_steps:
            raise ValueError(f"{name} must contain one value per action")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite numeric values")
        return value

    @staticmethod
    def _require_strictly_increasing(values: np.ndarray, name: str) -> None:
        if len(values) > 1 and np.any(np.diff(values.astype(np.float64)) <= 0):
            raise ValueError(f"{name} must be strictly increasing")

    @classmethod
    def _provenance_errors(cls, metadata: dict[str, Any]) -> list[str]:
        errors = []
        for key in cls.PROVENANCE_KEYS:
            if key not in metadata:
                errors.append(f"metadata.{key} is missing")
        if metadata.get("episode_schema_version") != "1.0":
            errors.append("episode_schema_version must be '1.0'")
        for key in ("episode_id", "task_id", "task_version", "scene_id", "input_source"):
            value = metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"metadata.{key} must be a non-empty string")
        for key in ("config_hash", "contract_hash"):
            value = metadata.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
                errors.append(f"metadata.{key} must be a SHA-256 hex digest")
        seed = metadata.get("random_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            errors.append("metadata.random_seed must be an integer")
        if not isinstance(metadata.get("initial_state"), dict) or not metadata["initial_state"]:
            errors.append("metadata.initial_state must be a non-empty object")
        quality = metadata.get("data_quality")
        if not isinstance(quality, dict) or quality.get("status") != "passed":
            errors.append("metadata.data_quality.status must be 'passed'")
        return errors

    def _expected_action_dimension(self) -> int | None:
        config = getattr(self.env, "cfg", None)
        dimension = getattr(config, "num_actions", None)
        if dimension is None:
            dimension = getattr(self.env, "num_actions", None)
        if dimension is None:
            manager = getattr(self.env, "action_manager", None)
            dimension = getattr(manager, "total_action_dim", None)
        return int(dimension) if dimension is not None else None

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def replay_episode(
        self,
        episode_data: dict[str, Any],
        record_trajectory: bool = True,
        *,
        reset_env: bool = True,
    ) -> dict[str, Any]:
        """Replay trusted physical targets through an optional task action adapter."""
        import torch

        if "actions" not in episode_data:
            raise ValueError("episode data is missing actions")
        actions = np.asarray(episode_data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError("episode must contain at least one action")
        if not np.isfinite(actions).all():
            raise ValueError("episode actions contain non-finite values")

        expected_dimension = self._expected_action_dimension()
        if self.action_adapter is None and expected_dimension is not None and actions.shape[1] != expected_dimension:
            raise ValueError(
                f"episode action dimension {actions.shape[1]} does not match environment action dimension "
                f"{expected_dimension}; provide a task-specific action_adapter"
            )

        if reset_env:
            self.env.reset()

        recorded_observations = [] if record_trajectory else None
        recorded_states: dict[str, list[np.ndarray]] = {}
        recorded_rewards: list[float] = []
        terminated_any = False
        truncated_any = False
        steps_replayed = 0

        for index, raw_action in enumerate(actions):
            if self.action_adapter is None:
                task_action = raw_action
            else:
                task_action = np.asarray(self.action_adapter(raw_action.copy()), dtype=np.float32)
            if task_action.ndim != 1 or not np.isfinite(task_action).all():
                raise ValueError(f"action adapter must return a finite 1D action at step {index}")
            if expected_dimension is not None and task_action.shape[0] != expected_dimension:
                raise ValueError(
                    f"adapted action dimension {task_action.shape[0]} does not match environment action dimension "
                    f"{expected_dimension} at step {index}"
                )

            action_tensor = torch.as_tensor(task_action, dtype=torch.float32, device=self.device).unsqueeze(0)
            observation, reward, terminated, truncated, _info = self.env.step(action_tensor)
            steps_replayed += 1
            recorded_rewards.append(float(self._to_numpy(reward).sum()))
            terminated_any = terminated_any or bool(self._to_numpy(terminated).any())
            truncated_any = truncated_any or bool(self._to_numpy(truncated).any())

            if record_trajectory and isinstance(observation, Mapping):
                recorded_observations.append({key: self._to_numpy(value) for key, value in observation.items()})
            if self.state_extractor is not None:
                state = self.state_extractor(self.env)
                for key, value in state.items():
                    recorded_states.setdefault(str(key), []).append(self._to_numpy(value))

            if terminated_any or truncated_any:
                break

        return {
            "num_steps_replayed": steps_replayed,
            "total_reward": float(sum(recorded_rewards)),
            "episode_terminated": terminated_any,
            "episode_truncated": truncated_any,
            "episode_finished": terminated_any or truncated_any,
            "recorded_observations": recorded_observations,
            "recorded_states": {key: np.asarray(values) for key, values in recorded_states.items()},
            "recorded_rewards": recorded_rewards,
        }

    def validate_trajectory_reproducibility(
        self,
        hdf5_path: str | Path,
        position_tolerance: float = 0.01,
        orientation_tolerance: float = 0.1,
    ) -> dict[str, Any]:
        """Validate state replay only when reset and measured-state adapters exist.

        A policy observation is not treated as a joint-state vector. Without the
        episode's initial state restore adapter and a named state extractor, the
        result is explicitly ``not_comparable`` rather than a false pass.
        """
        if position_tolerance <= 0 or orientation_tolerance <= 0:
            raise ValueError("replay tolerances must be positive")
        data = self.load_episode_from_hdf5(hdf5_path)
        metadata = data["metadata"]
        if not data["trainable"]:
            return self._not_comparable(data, "episode quality/provenance checks did not pass")
        if self.state_restorer is None or self.state_extractor is None:
            return self._not_comparable(
                data,
                "an initial-state restore adapter and named physical-state extractor are required",
            )

        try:
            self.env.reset(seed=metadata["random_seed"])
        except TypeError:
            self.env.reset()
        self.state_restorer(self.env, metadata["initial_state"])
        result = self.replay_episode(data, record_trajectory=True, reset_env=False)
        original = data["states"].get("next_joint_position")
        replayed = result["recorded_states"].get("joint_position")
        if original is None or replayed is None:
            return self._not_comparable(data, "joint_position is missing from source or replay state")
        if replayed.ndim == 3 and replayed.shape[1] == 1:
            replayed = replayed[:, 0, :]
        if original.shape[1:] != replayed.shape[1:]:
            return self._not_comparable(data, "source and replay joint-state dimensions differ")
        if len(original) != len(replayed):
            max_error = None
            reproducible = False
            reason = f"replay ended after {len(replayed)} of {len(original)} recorded steps"
        else:
            errors = np.linalg.norm(original - replayed, axis=-1)
            max_error = float(np.max(errors, initial=0.0))
            reproducible = max_error <= position_tolerance
            reason = None if reproducible else f"joint-state error {max_error:.6g} exceeds tolerance"
        return {
            "is_reproducible": reproducible,
            "validation_success": reproducible,
            "validation_status": "passed" if reproducible else "failed",
            "reason": reason,
            "max_position_error": max_error,
            "num_steps_replayed": result["num_steps_replayed"],
            "total_reward": result["total_reward"],
            "episode_id": metadata.get("episode_id"),
        }

    @staticmethod
    def _not_comparable(data: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "is_reproducible": None,
            "validation_success": False,
            "validation_status": "not_comparable",
            "reason": reason,
            "max_position_error": None,
            "num_steps_replayed": 0,
            "total_reward": 0.0,
            "episode_id": data["metadata"].get("episode_id"),
        }

    def batch_validate_episodes(
        self,
        hdf5_paths: list[str | Path],
        **validation_kwargs,
    ) -> list[dict[str, Any]]:
        """Batch results pass only when physical states were actually compared."""
        results = []
        for path in hdf5_paths:
            try:
                result = self.validate_trajectory_reproducibility(path, **validation_kwargs)
                result["hdf5_path"] = str(path)
                result["validation_success"] = result.get("is_reproducible") is True
                result["error_message"] = result.get("reason")
            except Exception as error:
                result = {
                    "hdf5_path": str(path),
                    "validation_success": False,
                    "error_message": str(error),
                    "is_reproducible": None,
                    "validation_status": "error",
                }
            results.append(result)
        return results


def validate_episode_replay(
    env: "DirectRLEnv",
    hdf5_path: str | Path,
    **kwargs,
) -> dict[str, Any]:
    """Validate a single episode with an explicitly adapted task environment."""
    validator_options = {
        key: kwargs.pop(key)
        for key in ("action_adapter", "state_restorer", "state_extractor")
        if key in kwargs
    }
    return EpisodeReplayValidator(env, **validator_options).validate_trajectory_reproducibility(
        hdf5_path, **kwargs
    )
