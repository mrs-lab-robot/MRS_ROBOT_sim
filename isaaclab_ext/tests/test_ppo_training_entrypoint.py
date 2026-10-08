from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_navigation_ppo.py"
SPEC = importlib.util.spec_from_file_location("train_navigation_ppo", SCRIPT_PATH)
TRAINING_SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TRAINING_SCRIPT)


def test_training_script_builds_a_checkpointable_rsl_ppo_runner():
    import torch
    from tensordict import TensorDict
    from rsl_rl.runners import OnPolicyRunner

    cfg = TRAINING_SCRIPT.build_runner_config(seed=23, steps_per_env=16, save_interval=7)
    assert cfg["seed"] == 23
    assert cfg["num_steps_per_env"] == 16
    assert cfg["save_interval"] == 7
    assert cfg["obs_groups"] == {"actor": ["policy"], "critic": ["policy"]}
    assert cfg["algorithm"]["class_name"] == "rsl_rl.algorithms.PPO"
    assert cfg["algorithm"]["rnd_cfg"] is None
    assert cfg["multi_gpu"] is None

    frozen_cfg, frozen_hash = TRAINING_SCRIPT._freeze_runner_config(cfg)
    runner_cfg_for_runner = copy.deepcopy(frozen_cfg)

    observations = TensorDict({"policy": torch.zeros((2, 8))}, batch_size=[2])

    class _Env:
        num_envs = 2
        num_actions = 3
        device = "cpu"
        max_episode_length = 10
        cfg = SimpleNamespace(is_finite_horizon=True)

        def get_observations(self):
            return observations

    runner = OnPolicyRunner(_Env(), runner_cfg_for_runner, log_dir=None, device="cpu")
    assert frozen_cfg == cfg
    assert frozen_hash == TRAINING_SCRIPT._canonical_hash(frozen_cfg)
    assert runner_cfg_for_runner != frozen_cfg
    assert runner.alg.storage.num_transitions_per_env == 16
    assert runner.alg.actor.distribution.output_dim == 3
    assert runner.alg.critic.mlp[-1].out_features == 1

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "navigation.pt"
        runner.logger.save_model = lambda *_args: None
        runner.save(str(checkpoint), infos={"task_id": "navigation_to_goal"})
        restored = OnPolicyRunner(_Env(), copy.deepcopy(frozen_cfg), log_dir=None, device="cpu")
        infos = restored.load(str(checkpoint), map_location="cpu")

        assert infos == {"task_id": "navigation_to_goal"}
        expected = runner.get_inference_policy(device="cpu")(observations)
        actual = restored.get_inference_policy(device="cpu")(observations)
        torch.testing.assert_close(actual, expected)


def test_training_config_rejects_nonpositive_rollout_and_checkpoint_intervals():
    with pytest.raises(ValueError, match="steps_per_env"):
        TRAINING_SCRIPT.build_runner_config(seed=1, steps_per_env=0, save_interval=1)
    with pytest.raises(ValueError, match="save_interval"):
        TRAINING_SCRIPT.build_runner_config(seed=1, steps_per_env=1, save_interval=0)


def test_finalized_training_manifest_binds_checkpoint_path_and_sha256(tmp_path):
    checkpoint_path = tmp_path / "checkpoints" / "model_final.pt"
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint bytes from runner.save")
    running_manifest = {
        "schema_version": 2,
        "status": "running",
        "task_id": "navigation_to_goal",
        "training_config_sha256": "training-config-hash",
    }

    completed = TRAINING_SCRIPT._finalize_training_manifest(
        running_manifest,
        checkpoint_path=checkpoint_path,
        iterations_completed=10,
        runner_iteration_index=9,
        previous_runner_iteration_index=0,
        completed_at_utc="2026-10-07T00:00:00+00:00",
    )

    expected_hash = hashlib.sha256(b"checkpoint bytes from runner.save").hexdigest()
    assert completed["status"] == "completed"
    assert completed["checkpoint"] == str(checkpoint_path.resolve())
    assert completed["checkpoint_sha256"] == expected_hash
    assert completed["iterations_completed_this_run"] == 10
    assert completed["runner_iteration_index"] == 9
    assert completed["previous_runner_iteration_index"] == 0
    assert completed["training_config_sha256"] == "training-config-hash"
    assert running_manifest["status"] == "running"


def test_create_task_environment_forwards_exact_factory_arguments():
    sentinel = object()
    captured_kwargs = {}

    def strict_factory(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    result = TRAINING_SCRIPT._create_task_environment(
        factory=strict_factory,
        task_spec_path=Path("/task.yaml"),
        scene_spec_path=Path("/scene.yaml"),
        device="cuda:0",
        num_envs=42,
        seed=23,
    )

    assert result is sentinel
    assert captured_kwargs == {
        "task_spec_path": Path("/task.yaml"),
        "scene_spec_path": Path("/scene.yaml"),
        "device": "cuda:0",
        "num_envs": 42,
        "seed": 23,
    }


def test_make_navigation_task_env_forwards_seed_to_environment_constructor():
    from unittest.mock import MagicMock, patch

    captured_kwargs = {}

    def capturing_constructor(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch(
        "mrs_robot_lab.environments.learning.navigation_task.NavigationTaskEnvironment",
        side_effect=capturing_constructor,
    ):
        from mrs_robot_lab.environments.learning.navigation_task import make_navigation_task_env

        env = make_navigation_task_env(
            task_spec_path=Path("/task.yaml"),
            scene_spec_path=Path("/scene.yaml"),
            device="cpu",
            num_envs=1,
            seed=23,
        )

        assert env is not None
        assert "seed" in captured_kwargs
        assert captured_kwargs["seed"] == 23


def test_configure_project_paths_restores_openflex_isaac_contract_to_sys_path():
    import sys

    original_sys_path = sys.path.copy()
    try:
        sys.path = [p for p in sys.path if "openflex_isaac_contract" not in p]

        TRAINING_SCRIPT._configure_project_paths()

        repo_root = SCRIPT_PATH.resolve().parents[2]
        expected_contract_path = str(
            repo_root / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_contract"
        )
        assert expected_contract_path in sys.path
    finally:
        sys.path[:] = original_sys_path
