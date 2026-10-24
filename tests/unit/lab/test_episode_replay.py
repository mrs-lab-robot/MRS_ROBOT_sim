"""Behavior tests for trusted Episode replay and validation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.environments.learning.episode_replay import EpisodeReplayValidator


def _metadata(*, initial_state=True):
    value = {
        "episode_schema_version": "1.0",
        "episode_id": "episode-001",
        "config_hash": "c" * 64,
        "contract_hash": "a" * 64,
        "random_seed": 7,
        "input_source": "pico_vr",
        "task_id": "dual_arm_box",
        "task_version": "1.0",
        "scene_id": "flat_demo",
        "data_quality": {"status": "passed"},
        "joint_state_order": list(JOINT_STATE_NAMES),
    }
    if initial_state:
        value["initial_state"] = {"robot_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}
    return value


def _write_episode(
    path: Path, *, legacy=False, timestamps=None, initial_state=True, action_alias=0.2,
    steps=3, format_name="mrs_robot_arena_teleop_v1"
):
    state_width = len(JOINT_STATE_NAMES)
    state = np.zeros((steps, state_width), dtype=np.float32)
    action = np.full((steps, ACTION_DIMENSION), action_alias, dtype=np.float32)
    applied = np.full((steps, ACTION_DIMENSION), 0.2, dtype=np.float32)
    operator = np.full((steps, ACTION_DIMENSION), 0.4, dtype=np.float32)
    with h5py.File(path, "w") as episode:
        episode.attrs["format"] = format_name
        episode.attrs["metadata_json"] = json.dumps(_metadata(initial_state=initial_state))
        episode.create_dataset("sim_time_ns", data=np.asarray(timestamps if timestamps is not None else range(steps), dtype=np.uint64))
        episode.create_dataset("step_index", data=np.arange(steps, dtype=np.int64))
        episode.create_dataset("command_seq", data=np.arange(steps, dtype=np.uint64))
        episode.create_dataset("action", data=action)
        if not legacy:
            episode.create_dataset("operator_command", data=operator)
            episode.create_dataset("applied_target", data=applied)
        episode.create_dataset("joint_position", data=state)
        episode.create_dataset("joint_velocity", data=state)
        episode.create_dataset("next_joint_position", data=state)


class FakeEnv:
    def __init__(self, action_dim=ACTION_DIMENSION, stop_after=None):
        self.device = "cpu"
        self.cfg = SimpleNamespace(num_actions=action_dim)
        self.stop_after = stop_after
        self.step_calls = 0
        self.received_actions = []

    def reset(self, **_kwargs):
        self.step_calls = 0
        return {"policy": torch.zeros((1, 2))}, {}

    def step(self, action):
        if tuple(action.shape) != (1, self.cfg.num_actions):
            raise ValueError("fake environment action shape mismatch")
        self.step_calls += 1
        self.received_actions.append(action.detach().cpu().numpy())
        done = self.stop_after is not None and self.step_calls >= self.stop_after
        return (
            {"policy": torch.full((1, 2), float(self.step_calls))},
            torch.tensor([1.0]),
            torch.tensor([done]),
            torch.tensor([False]),
            {},
        )


class EpisodeReplayValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "episode.hdf5"
        _write_episode(self.path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_loader_uses_applied_targets_not_ambiguous_legacy_action(self):
        _write_episode(self.path, action_alias=0.8)
        validator = EpisodeReplayValidator(FakeEnv())

        data = validator.load_episode_from_hdf5(self.path)

        np.testing.assert_allclose(data["actions"], 0.2, atol=1e-7)
        np.testing.assert_allclose(data["operator_commands"], 0.4, atol=1e-7)
        self.assertEqual(data["action_source"], "applied_target")
        self.assertFalse(data["trainable"])
        self.assertTrue(any("action compatibility alias" in reason for reason in data["quality_errors"]))

    def test_loader_accepts_current_single_entry_capture_format(self):
        _write_episode(self.path, format_name="mrs_robot_capture_v1")
        validator = EpisodeReplayValidator(FakeEnv())

        data = validator.load_episode_from_hdf5(self.path)

        self.assertEqual(data["action_source"], "applied_target")

    def test_legacy_action_is_readable_but_not_marked_trainable(self):
        _write_episode(self.path, legacy=True, action_alias=0.8)
        validator = EpisodeReplayValidator(FakeEnv())

        data = validator.load_episode_from_hdf5(self.path)

        np.testing.assert_allclose(data["actions"], 0.8, atol=1e-7)
        self.assertEqual(data["action_source"], "legacy_action")
        self.assertFalse(data["trainable"])

    def test_loader_rejects_non_monotonic_or_misaligned_time(self):
        _write_episode(self.path, timestamps=[1, 1, 3])
        validator = EpisodeReplayValidator(FakeEnv())

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validator.load_episode_from_hdf5(self.path)

    def test_replay_checks_action_dimension_before_environment_step(self):
        env = FakeEnv(action_dim=3)
        validator = EpisodeReplayValidator(env)
        data = {"actions": np.zeros((2, ACTION_DIMENSION), dtype=np.float32)}

        with self.assertRaisesRegex(ValueError, "action dimension"):
            validator.replay_episode(data)
        self.assertEqual(env.step_calls, 0)

    def test_replay_applies_adapter_and_reports_scalar_reward_and_early_stop(self):
        env = FakeEnv(action_dim=3, stop_after=2)
        adapter_calls = []

        def adapter(action):
            adapter_calls.append(action.copy())
            return action[:3]

        validator = EpisodeReplayValidator(env, action_adapter=adapter)
        data = {"actions": np.ones((5, ACTION_DIMENSION), dtype=np.float32)}

        result = validator.replay_episode(data)

        self.assertEqual(env.step_calls, 2)
        self.assertEqual(len(adapter_calls), 2)
        self.assertEqual(result["num_steps_replayed"], 2)
        self.assertEqual(result["total_reward"], 2.0)
        self.assertTrue(result["episode_terminated"])

    def test_empty_action_episode_is_rejected_without_unbound_step_index(self):
        validator = EpisodeReplayValidator(FakeEnv())

        with self.assertRaisesRegex(ValueError, "at least one action"):
            validator.replay_episode({"actions": np.empty((0, ACTION_DIMENSION), dtype=np.float32)})

    def test_missing_state_restore_adapter_is_not_reported_as_reproducible(self):
        validator = EpisodeReplayValidator(FakeEnv())

        result = validator.validate_trajectory_reproducibility(self.path)

        self.assertIsNone(result["is_reproducible"])
        self.assertFalse(result["validation_success"])
        self.assertEqual(result["validation_status"], "not_comparable")

    def test_batch_validation_does_not_treat_unknown_reproducibility_as_success(self):
        validator = EpisodeReplayValidator(FakeEnv())

        result = validator.batch_validate_episodes([self.path])[0]

        self.assertFalse(result["validation_success"])
        self.assertIsNone(result["is_reproducible"])


if __name__ == "__main__":
    unittest.main()
