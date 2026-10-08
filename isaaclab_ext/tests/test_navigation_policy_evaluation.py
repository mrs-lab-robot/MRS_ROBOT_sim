"""测试导航策略评估 API 的纯辅助函数。

验证 episode result 构建和聚合逻辑，包括状态判断、边界验证和统计计算。
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch
from tensordict import TensorDict

from mrs_robot_lab.environments.learning.navigation_evaluation import (
    aggregate_episode_results,
    make_episode_result,
)

EVALUATION_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_navigation_ppo.py"
EVALUATION_SPEC = importlib.util.spec_from_file_location(
    "evaluate_navigation_ppo", EVALUATION_SCRIPT_PATH
)
EVALUATION_SCRIPT = importlib.util.module_from_spec(EVALUATION_SPEC)
assert EVALUATION_SPEC.loader is not None
EVALUATION_SPEC.loader.exec_module(EVALUATION_SCRIPT)


def _completed_manifest(checkpoint_path: Path, **fields):
    """Build a complete schema-v2 training manifest for a real test file."""
    return {
        "schema_version": 2,
        "status": "completed",
        "task_id": "navigation_to_goal",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        **fields,
    }


class TestMakeEpisodeResult:
    """测试 episode result 构建函数。"""

    def test_successful_episode(self):
        """成功的 episode：terminated=True, truncated=False。"""
        result = make_episode_result(index=0, seed=42, steps=100, terminated=True, truncated=False)

        assert result["index"] == 0
        assert result["seed"] == 42
        assert result["steps"] == 100
        assert result["terminated"] is True
        assert result["truncated"] is False
        assert result["success"] is True
        assert result["timeout"] is False

    def test_timeout_episode(self):
        """超时的 episode：terminated=False, truncated=True。"""
        result = make_episode_result(index=1, seed=123, steps=500, terminated=False, truncated=True)

        assert result["index"] == 1
        assert result["seed"] == 123
        assert result["steps"] == 500
        assert result["terminated"] is False
        assert result["truncated"] is True
        assert result["success"] is False
        assert result["timeout"] is True

    def test_ongoing_episode(self):
        """拒绝未结束的 episode：terminated=False, truncated=False。"""
        with pytest.raises(ValueError, match="terminated.*truncated.*both.*False"):
            make_episode_result(index=2, seed=999, steps=50, terminated=False, truncated=False)

    def test_reject_both_terminated_and_truncated(self):
        """拒绝同时 terminated 和 truncated 的非法状态。"""
        with pytest.raises(ValueError, match="terminated.*truncated.*both.*True"):
            make_episode_result(index=0, seed=42, steps=100, terminated=True, truncated=True)

    def test_reject_zero_steps(self):
        """拒绝步数为零的非法输入。"""
        with pytest.raises(ValueError, match="steps.*positive"):
            make_episode_result(index=0, seed=42, steps=0, terminated=True, truncated=False)

    def test_reject_negative_steps(self):
        """拒绝负步数的非法输入。"""
        with pytest.raises(ValueError, match="steps.*positive"):
            make_episode_result(index=0, seed=42, steps=-10, terminated=True, truncated=False)


class TestAggregateEpisodeResults:
    """测试 episode results 聚合函数。"""

    def test_all_successful(self):
        """所有 episodes 都成功。"""
        records = [
            make_episode_result(0, 42, 100, True, False),
            make_episode_result(1, 43, 120, True, False),
            make_episode_result(2, 44, 90, True, False),
        ]

        summary = aggregate_episode_results(records)

        assert summary["total_episodes"] == 3
        assert summary["success_count"] == 3
        assert summary["timeout_count"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["mean_steps"] == pytest.approx(103.333, abs=0.01)

    def test_two_of_three_successful(self):
        """3 个 episodes 中 2 个成功，1 个超时。"""
        records = [
            make_episode_result(0, 42, 100, True, False),  # 成功
            make_episode_result(1, 43, 500, False, True),  # 超时
            make_episode_result(2, 44, 120, True, False),  # 成功
        ]

        summary = aggregate_episode_results(records)

        assert summary["total_episodes"] == 3
        assert summary["success_count"] == 2
        assert summary["timeout_count"] == 1
        assert summary["success_rate"] == pytest.approx(2.0 / 3.0, abs=0.001)
        assert summary["mean_steps"] == pytest.approx(240.0, abs=0.01)

    def test_all_timeout(self):
        """所有 episodes 都超时。"""
        records = [
            make_episode_result(0, 42, 500, False, True),
            make_episode_result(1, 43, 500, False, True),
        ]

        summary = aggregate_episode_results(records)

        assert summary["total_episodes"] == 2
        assert summary["success_count"] == 0
        assert summary["timeout_count"] == 2
        assert summary["success_rate"] == 0.0
        assert summary["mean_steps"] == 500.0

    def test_mixed_episodes(self):
        """混合状态：成功和超时。"""
        records = [
            make_episode_result(0, 42, 100, True, False),   # 成功
            make_episode_result(1, 43, 500, False, True),   # 超时
            make_episode_result(2, 45, 150, True, False),   # 成功
        ]

        summary = aggregate_episode_results(records)

        assert summary["total_episodes"] == 3
        assert summary["success_count"] == 2
        assert summary["timeout_count"] == 1
        assert summary["success_rate"] == pytest.approx(2.0 / 3.0, abs=0.001)
        assert summary["mean_steps"] == pytest.approx(250.0, abs=0.01)

    def test_empty_records(self):
        """空记录列表应返回零值统计。"""
        summary = aggregate_episode_results([])

        assert summary["total_episodes"] == 0
        assert summary["success_count"] == 0
        assert summary["timeout_count"] == 0
        assert summary["success_rate"] == 0.0
        assert summary["mean_steps"] == 0.0


class TestValidateCheckpointProvenance:
    """测试 checkpoint 溯源验证函数。"""

    def test_valid_checkpoint_all_hashes_match(self, tmp_path):
        """有效的 checkpoint：所有哈希值都匹配。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        # 构造 checkpoint 文件
        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        # 构造 manifest
        manifest = _completed_manifest(
            checkpoint_path,
            task_spec_sha256="abc123",
            scene_spec_sha256="def456",
            embodiment_contract_sha256="ghi789",
            runner_config_sha256="jkl012",
            training_config_sha256="mno345",
        )

        # 验证通过
        validate_checkpoint_provenance(
            checkpoint_path,
            manifest,
            task_spec_sha256="abc123",
            scene_spec_sha256="def456",
            embodiment_contract_sha256="ghi789",
            runner_config_sha256="jkl012",
            training_config_sha256="mno345",
        )

    def test_reject_checkpoint_path_not_recorded_in_manifest(self, tmp_path):
        """拒绝虽存在、但不是 manifest 所记录的 checkpoint。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")
        manifest = _completed_manifest(checkpoint_path)
        manifest["checkpoint"] = str(tmp_path / "other.pt")

        with pytest.raises(ValueError, match="checkpoint.*manifest"):
            validate_checkpoint_provenance(checkpoint_path, manifest)

    def test_reject_checkpoint_content_changed_after_training(self, tmp_path):
        """拒绝内容与训练 manifest SHA256 不一致的 checkpoint。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("original checkpoint")
        manifest = _completed_manifest(checkpoint_path)
        checkpoint_path.write_text("modified checkpoint")

        with pytest.raises(ValueError, match="checkpoint_sha256.*mismatch"):
            validate_checkpoint_provenance(checkpoint_path, manifest)

    def test_reject_schema_v2_manifest_without_checkpoint_hash(self, tmp_path):
        """schema v2 的已完成训练记录必须绑定 checkpoint 内容哈希。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("checkpoint")
        manifest = _completed_manifest(checkpoint_path)
        del manifest["checkpoint_sha256"]

        with pytest.raises(ValueError, match="checkpoint_sha256.*required"):
            validate_checkpoint_provenance(checkpoint_path, manifest)

    def test_reject_wrong_task_id(self, tmp_path):
        """拒绝非导航任务的 checkpoint manifest。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")
        manifest = _completed_manifest(checkpoint_path)
        manifest["task_id"] = "dual_arm_box_transport"

        with pytest.raises(ValueError, match="task_id.*navigation_to_goal"):
            validate_checkpoint_provenance(checkpoint_path, manifest)

    def test_reject_nonexistent_checkpoint(self, tmp_path):
        """拒绝不存在的 checkpoint 路径。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        nonexistent = tmp_path / "does_not_exist.pt"
        manifest = {
            "schema_version": 2,
            "status": "completed",
            "task_id": "navigation_to_goal",
            "checkpoint": str(nonexistent),
            "checkpoint_sha256": "0" * 64,
        }

        with pytest.raises(FileNotFoundError, match="Checkpoint.*not found"):
            validate_checkpoint_provenance(nonexistent, manifest)

    def test_reject_non_completed_status(self, tmp_path):
        """拒绝状态不是 completed 的 manifest。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        manifest = _completed_manifest(checkpoint_path)
        manifest["status"] = "failed"

        with pytest.raises(ValueError, match="status.*completed"):
            validate_checkpoint_provenance(checkpoint_path, manifest)

    def test_reject_wrong_task_spec(self, tmp_path):
        """拒绝 task_spec_sha256 不匹配。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        manifest = _completed_manifest(checkpoint_path, task_spec_sha256="abc123")

        with pytest.raises(ValueError, match="task_spec_sha256.*mismatch"):
            validate_checkpoint_provenance(
                checkpoint_path, manifest, task_spec_sha256="wrong_hash"
            )

    def test_reject_wrong_scene_spec(self, tmp_path):
        """拒绝 scene_spec_sha256 不匹配。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        manifest = _completed_manifest(checkpoint_path, scene_spec_sha256="def456")

        with pytest.raises(ValueError, match="scene_spec_sha256.*mismatch"):
            validate_checkpoint_provenance(
                checkpoint_path, manifest, scene_spec_sha256="wrong_hash"
            )

    def test_reject_wrong_embodiment_contract(self, tmp_path):
        """拒绝 embodiment_contract_sha256 不匹配。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        manifest = _completed_manifest(
            checkpoint_path, embodiment_contract_sha256="ghi789"
        )

        with pytest.raises(ValueError, match="embodiment_contract_sha256.*mismatch"):
            validate_checkpoint_provenance(
                checkpoint_path, manifest, embodiment_contract_sha256="wrong_hash"
            )

    def test_reject_wrong_runner_config(self, tmp_path):
        """拒绝 runner_config_sha256 不匹配。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        manifest = _completed_manifest(checkpoint_path, runner_config_sha256="jkl012")

        with pytest.raises(ValueError, match="runner_config_sha256.*mismatch"):
            validate_checkpoint_provenance(
                checkpoint_path, manifest, runner_config_sha256="wrong_hash"
            )

    def test_reject_wrong_training_config(self, tmp_path):
        """拒绝训练超参数及运行设置的综合哈希不匹配。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")
        manifest = _completed_manifest(checkpoint_path, training_config_sha256="mno345")

        with pytest.raises(ValueError, match="training_config_sha256.*mismatch"):
            validate_checkpoint_provenance(
                checkpoint_path, manifest, training_config_sha256="wrong_hash"
            )

    def test_partial_hash_validation(self, tmp_path):
        """只验证提供的哈希值，未提供的跳过。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            validate_checkpoint_provenance,
        )

        checkpoint_path = tmp_path / "policy.pt"
        checkpoint_path.write_text("fake checkpoint")

        manifest = _completed_manifest(
            checkpoint_path,
            task_spec_sha256="abc123",
            scene_spec_sha256="def456",
        )

        # 只验证 task_spec，scene_spec 不提供
        validate_checkpoint_provenance(
            checkpoint_path, manifest, task_spec_sha256="abc123"
        )


class TestEvaluationManifestLoading:
    """测试评估脚本按实际训练字段验证可复现配置与 checkpoint。"""

    @staticmethod
    def _digest_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _canonical_hash(value) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def test_help_flag_succeeds_without_checkpoint_argument(self):
        completed = subprocess.run(
            [sys.executable, str(EVALUATION_SCRIPT_PATH), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0
        assert "--checkpoint" in completed.stdout

    def test_load_validates_training_artifact_and_returns_verified_checkpoint_hash(self, tmp_path):
        repo_root = tmp_path / "repo"
        task_path = repo_root / "sim_runtime/config/tasks/navigation_to_goal.yaml"
        scene_path = repo_root / "sim_runtime/config/scenes/flat_navigation.yaml"
        contract_path = repo_root / (
            "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        )
        for path in (task_path, scene_path, contract_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name + "\n", encoding="utf-8")

        checkpoint_path = tmp_path / "run/checkpoints/model_final.pt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"verified model bytes")
        runner_config = {"seed": 7, "obs_groups": {"actor": ["policy"]}}
        task_hash = self._digest_bytes(task_path.read_bytes())
        scene_hash = self._digest_bytes(scene_path.read_bytes())
        contract_hash = self._digest_bytes(contract_path.read_bytes())
        training_config = {
            "runner_config": runner_config,
            "num_envs": 4,
            "iterations": 25,
            "device": "cuda:0",
            "task_spec_sha256": task_hash,
            "scene_spec_sha256": scene_hash,
        }
        manifest = {
            "schema_version": 2,
            "status": "completed",
            "task_id": "navigation_to_goal",
            "task_spec": str(task_path),
            "scene_spec": str(scene_path),
            "task_spec_sha256": task_hash,
            "scene_spec_sha256": scene_hash,
            "embodiment_contract_sha256": contract_hash,
            "runner_config": runner_config,
            "runner_config_sha256": self._canonical_hash(runner_config),
            "training_config_sha256": self._canonical_hash(training_config),
            "num_envs": 4,
            "iterations_requested": 25,
            "device": "cuda:0",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": self._digest_bytes(checkpoint_path.read_bytes()),
        }
        manifest_path = checkpoint_path.parent.parent / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        actual = EVALUATION_SCRIPT._load_and_validate_run_manifest(
            checkpoint_path, manifest_path, repo_root
        )

        assert actual[0] == manifest
        assert actual[1:4] == (task_path, scene_path, contract_path)
        assert actual[4] == manifest["checkpoint_sha256"]

    def test_rejects_changed_task_file_before_launching_isaac(self, tmp_path):
        repo_root = tmp_path / "repo"
        task_path = repo_root / "sim_runtime/config/tasks/navigation_to_goal.yaml"
        scene_path = repo_root / "sim_runtime/config/scenes/flat_navigation.yaml"
        contract_path = repo_root / (
            "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        )
        for path in (task_path, scene_path, contract_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original\n", encoding="utf-8")

        checkpoint_path = tmp_path / "run/checkpoints/model_final.pt"
        checkpoint_path.parent.mkdir(parents=True)
        checkpoint_path.write_bytes(b"verified model bytes")
        runner_config = {"seed": 7}
        task_hash = self._digest_bytes(task_path.read_bytes())
        scene_hash = self._digest_bytes(scene_path.read_bytes())
        training_config = {
            "runner_config": runner_config,
            "num_envs": 4,
            "iterations": 25,
            "device": "cuda:0",
            "task_spec_sha256": task_hash,
            "scene_spec_sha256": scene_hash,
        }
        manifest = {
            "schema_version": 2,
            "status": "completed",
            "task_id": "navigation_to_goal",
            "task_spec": str(task_path),
            "scene_spec": str(scene_path),
            "task_spec_sha256": task_hash,
            "scene_spec_sha256": scene_hash,
            "embodiment_contract_sha256": self._digest_bytes(contract_path.read_bytes()),
            "runner_config": runner_config,
            "runner_config_sha256": self._canonical_hash(runner_config),
            "training_config_sha256": self._canonical_hash(training_config),
            "num_envs": 4,
            "iterations_requested": 25,
            "device": "cuda:0",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": self._digest_bytes(checkpoint_path.read_bytes()),
        }
        manifest_path = checkpoint_path.parent.parent / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        task_path.write_text("modified task\n", encoding="utf-8")

        with pytest.raises(ValueError, match="task_spec_sha256 mismatch"):
            EVALUATION_SCRIPT._load_and_validate_run_manifest(
                checkpoint_path, manifest_path, repo_root
            )


class TestRunNavigationEpisode:
    """测试单个 navigation episode，fake 仅替代 Isaac Lab simulator。"""

    @staticmethod
    def _step_result(observations, *, terminated=False, truncated=False):
        return (
            observations,
            torch.tensor([0.0]),
            torch.tensor([terminated]),
            torch.tensor([truncated]),
            {},
        )

    @staticmethod
    def _environment(policy_obs, max_episode_length=10):
        env = Mock()
        env.num_envs = 1
        env.max_episode_length = max_episode_length
        observations = {"policy": policy_obs}
        env.reset.return_value = (observations, {})
        return env, observations

    def test_episode_calls_reset_with_seed(self):
        """重置 episode 时将指定 seed 传给 Isaac Lab。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        env, observations = self._environment(torch.zeros((1, 8)))
        env.step.return_value = self._step_result(observations, terminated=True)
        policy = Mock(return_value=torch.zeros((1, 3)))

        run_navigation_episode(env, policy, index=0, seed=42)

        env.reset.assert_called_once_with(seed=42)

    def test_policy_receives_real_rsl_observation_contract(self):
        """RSL-RL policy 输入含 policy 项的批处理 TensorDict。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        policy_obs = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        env, observations = self._environment(policy_obs)
        env.step.return_value = self._step_result(observations, terminated=True)
        policy = Mock(return_value=torch.zeros((1, 3)))

        run_navigation_episode(env, policy, index=0, seed=42)

        policy_obs_arg = policy.call_args.args[0]
        assert isinstance(policy_obs_arg, TensorDict)
        assert policy_obs_arg.batch_size == torch.Size([1])
        assert set(policy_obs_arg.keys()) == {"policy"}
        torch.testing.assert_close(policy_obs_arg["policy"], policy_obs)

    def test_episode_advances_via_direct_env_step_with_tensor_actions(self):
        """每个控制步都用动作张量调用 DirectRLEnv.step。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        env, observations = self._environment(torch.zeros((1, 8)))
        env.step.side_effect = [
            self._step_result(observations),
            self._step_result(observations, terminated=True),
        ]
        actions = torch.tensor([[0.1, -0.2, 0.3]])
        policy = Mock(return_value=actions)

        run_navigation_episode(env, policy, index=0, seed=42)

        assert env.step.call_count == 2
        torch.testing.assert_close(env.step.call_args.args[0], actions)

    def test_direct_env_path_preserves_rsl_action_clipping(self):
        """绕过 RSL wrapper 时仍把策略动作限幅到训练时的 [-1, 1]。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        env, observations = self._environment(torch.zeros((1, 8)))
        env.step.return_value = self._step_result(observations, terminated=True)
        policy_actions = torch.tensor([[2.0, -3.0, 0.5]])
        policy = Mock(return_value=policy_actions)

        run_navigation_episode(env, policy, index=0, seed=42)

        torch.testing.assert_close(
            env.step.call_args.args[0], torch.tensor([[1.0, -1.0, 0.5]])
        )

    def test_second_step_terminated_returns_success(self):
        """DirectRLEnv 第二步 terminated 时记录成功 episode。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        env, observations = self._environment(torch.zeros((1, 8)))
        env.step.side_effect = [
            self._step_result(observations),
            self._step_result(observations, terminated=True),
        ]
        policy = Mock(return_value=torch.zeros((1, 3)))

        result = run_navigation_episode(env, policy, index=5, seed=123)

        assert result["index"] == 5
        assert result["seed"] == 123
        assert result["steps"] == 2
        assert result["terminated"] is True
        assert result["truncated"] is False
        assert result["success"] is True
        assert result["timeout"] is False

    def test_max_episode_length_truncated_returns_timeout(self):
        """DirectRLEnv 在 max_episode_length 报 truncated 时记录超时。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        env, observations = self._environment(torch.zeros((1, 8)), max_episode_length=10)
        env.step.side_effect = [
            self._step_result(observations)
            for _ in range(9)
        ] + [self._step_result(observations, truncated=True)]
        policy = Mock(return_value=torch.zeros((1, 3)))

        result = run_navigation_episode(env, policy, index=7, seed=999)

        assert result["index"] == 7
        assert result["seed"] == 999
        assert result["steps"] == 10
        assert result["terminated"] is False
        assert result["truncated"] is True
        assert result["success"] is False
        assert result["timeout"] is True

    def test_guard_rejects_environment_that_never_terminates(self):
        """评估器不能无限 rollout；必须在环境上限处显式失败。"""
        from mrs_robot_lab.environments.learning.navigation_evaluation import (
            run_navigation_episode,
        )

        env, observations = self._environment(torch.zeros((1, 8)), max_episode_length=2)
        env.step.return_value = self._step_result(observations)
        policy = Mock(return_value=torch.zeros((1, 3)))

        with pytest.raises(RuntimeError, match="max_episode_length"):
            run_navigation_episode(env, policy, index=0, seed=42)

        assert env.step.call_count == 2
