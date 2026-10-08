"""阶段1契约基线红色测试 - Episode契约v1核心行为

运行方式：
  cd isaaclab_ext
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../arena_benchmark/.venv/bin/python -B -m pytest \
    -p no:cacheprovider tests/test_episode_contract_v1_red.py -v

预期结果：测试1-5失败（缺少实现），测试6通过（验证现有行为）
"""

from pathlib import Path
import tempfile
import h5py
import json
import pytest


class TestContractExtensions:
    """契约扩展：action_units字段"""

    def test_contract_exposes_action_units(self):
        """验证EmbodimentContract解析并暴露units字段"""
        from mrs_robot_lab.contracts import load_embodiment_contract

        contract_path = (
            Path(__file__).resolve().parents[2]
            / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        )
        contract = load_embodiment_contract(contract_path)

        # 预期失败：EmbodimentContract没有action_units属性
        assert hasattr(contract, "action_units"), "contract缺少action_units字段"
        assert isinstance(contract.action_units, dict)

        # 验证base_twist单位（从embodiment.yaml line 28-29）
        assert contract.action_units["base_twist"] == ("m/s", "m/s", "rad/s")
        # 验证arm单位
        assert contract.action_units["left_arm_position"] == ("rad",) * 7


class TestEpisodeActionSemantics:
    """Episode动作语义：区分operator_command和applied_target"""

    def test_saved_episode_contains_operator_and_applied_actions(self):
        """验证保存的HDF5包含operator_command(22D)和applied_target(22D)且值不同"""
        from mrs_robot_lab.recorders.teleop_episode import TeleopEpisodeRecorder
        import numpy as np

        recorder = TeleopEpisodeRecorder()
        recorder.start()

        # 预期失败：append()不接受operator_command/applied_target参数
        recorder.append(
            sim_time_ns=1000000,
            step_index=0,
            operator_command=[1.0] * 22,    # 原始输入
            applied_target=[0.8] * 22,      # 裁剪后不同
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=1,
        )

        metadata = {
            "episode_id": "ep_test_001",
            "config_hash": "c" * 64,
            "contract_hash": "a" * 64,
            "random_seed": 42,
            "input_source": "simulation_vr_teleop",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = recorder.save(f"{tmpdir}/ep.h5", metadata=metadata)

            with h5py.File(path, "r") as f:
                # 验证saver添加了episode_schema_version
                saved_meta = json.loads(f.attrs["metadata_json"])
                assert saved_meta["episode_schema_version"] == "1.0"

                # 预期失败：HDF5缺少operator_command和applied_target字段
                assert "operator_command" in f, "HDF5缺少operator_command数据集"
                assert "applied_target" in f, "HDF5缺少applied_target数据集"

                # 验证维度：22D动作空间
                assert f["operator_command"].shape == (1, 22)
                assert f["applied_target"].shape == (1, 22)

                # 验证值确实不同（按提供的值保存）
                np.testing.assert_allclose(f["operator_command"][0], [1.0] * 22, rtol=1e-6)
                np.testing.assert_allclose(f["applied_target"][0], [0.8] * 22, rtol=1e-6)

                # 验证joint_position维度：19D关节状态
                assert f["joint_position"].shape == (1, 19)


class TestEpisodeProvenance:
    """Episode溯源：metadata包含所有必需字段"""

    def test_saved_episode_metadata_includes_all_provenance_fields(self):
        """验证保存的HDF5 metadata包含episode_schema_version和所有溯源字段"""
        from mrs_robot_lab.recorders.teleop_episode import TeleopEpisodeRecorder

        recorder = TeleopEpisodeRecorder()
        recorder.start()
        recorder.append(
            sim_time_ns=1000000,
            step_index=0,
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=1,
        )

        # 提供完整的溯源metadata
        metadata = {
            "episode_id": "ep_test_001",
            "config_hash": "c" * 64,  # SHA256
            "contract_hash": "a" * 64,
            "random_seed": 42,
            "input_source": "simulation_vr_teleop",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = recorder.save(f"{tmpdir}/ep.h5", metadata=metadata)

            with h5py.File(path, "r") as f:
                # 新采集格式不再伪装成 Arena 数据；旧格式仍由只读导入器兼容。
                assert f.attrs["format"] == "mrs_robot_capture_v1"

                # 验证metadata_json包含所有溯源字段
                saved_meta = json.loads(f.attrs["metadata_json"])

                # 预期失败：缺少episode_schema_version和其他溯源字段
                assert "episode_schema_version" in saved_meta, "缺少episode_schema_version"
                assert saved_meta["episode_schema_version"] == "1.0"

                # 验证所有溯源字段
                assert "episode_id" in saved_meta, "缺少episode_id"
                assert "config_hash" in saved_meta, "缺少config_hash"
                assert "contract_hash" in saved_meta, "缺少contract_hash"
                assert "random_seed" in saved_meta, "缺少random_seed"
                assert "input_source" in saved_meta, "缺少input_source"


class TestEpisodeTemporalConstraints:
    """Episode时序约束：单调性和对齐"""

    def test_append_rejects_non_monotonic_timestamps(self):
        """验证append()拒绝非单调的sim_time_ns"""
        from mrs_robot_lab.recorders.teleop_episode import TeleopEpisodeRecorder

        recorder = TeleopEpisodeRecorder()
        recorder.start()

        recorder.append(
            sim_time_ns=2000000,
            step_index=0,
            operator_command=[0.0] * 22,
            applied_target=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=1,
        )

        # 预期失败：append()不检查时间戳单调性，应该raise但不会
        with pytest.raises(ValueError, match="monotonic|timestamp"):
            recorder.append(
                sim_time_ns=1000000,  # 时间倒退
                step_index=1,
                operator_command=[0.0] * 22,
                applied_target=[0.0] * 22,
                joint_position=[0.0] * 19,
                joint_velocity=[0.0] * 19,
                next_joint_position=[0.0] * 19,
                command_seq=2,
            )

    def test_saved_episode_includes_step_index_for_actions_and_cameras(self):
        """验证HDF5包含动作step_index，相机帧step_index与动作关联"""
        from mrs_robot_lab.recorders.teleop_episode import TeleopEpisodeRecorder
        import numpy as np

        recorder = TeleopEpisodeRecorder()
        recorder.start()

        # 第0步：无相机
        recorder.append(
            sim_time_ns=1000000,
            step_index=0,
            operator_command=[0.0] * 22,
            applied_target=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=1,
        )

        # 第1步：有相机
        recorder.append(
            sim_time_ns=2000000,
            step_index=1,
            operator_command=[1.0] * 22,
            applied_target=[1.0] * 22,
            joint_position=[0.1] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.1] * 19,
            command_seq=2,
            camera_frames={"base_camera": b"\xff\xd8\xff\xe0"},
        )

        metadata = {
            "episode_id": "ep_test_002",
            "config_hash": "c" * 64,
            "contract_hash": "a" * 64,
            "random_seed": 42,
            "input_source": "simulation_vr_teleop",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = recorder.save(f"{tmpdir}/ep.h5", metadata=metadata)

            with h5py.File(path, "r") as f:
                # 验证saver添加了episode_schema_version
                saved_meta = json.loads(f.attrs["metadata_json"])
                assert saved_meta["episode_schema_version"] == "1.0"

                # 预期失败：顶层缺少step_index数据集
                assert "step_index" in f, "HDF5缺少step_index数据集"
                action_step_indices = f["step_index"][:]
                np.testing.assert_array_equal(action_step_indices, [0, 1])

                # 预期失败：相机帧缺少step_index
                camera = f["camera_frames/base_camera"]
                assert "step_index" in camera, "相机帧缺少step_index数据集"
                camera_step_indices = camera["step_index"][:]
                np.testing.assert_array_equal(camera_step_indices, [1])

                # 验证相机sim_time与第1步动作的sim_time一致
                camera_sim_time = camera["sim_time_ns"][0]
                action_sim_times = f["sim_time_ns"][:]
                assert camera_sim_time == action_sim_times[1] == 2000000


class TestEpisodeCameraFlexibility:
    """Episode相机灵活性：任务可选相机子集"""

    def test_episode_allows_per_task_camera_subset(self):
        """验证Episode允许任务只使用部分相机（现有行为应正确）"""
        from mrs_robot_lab.recorders.teleop_episode import TeleopEpisodeRecorder

        recorder = TeleopEpisodeRecorder()
        recorder.start()

        # 只使用head_camera（不是全部4个相机）
        recorder.append(
            sim_time_ns=1000000,
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=1,
            camera_frames={"head_camera": b"\xff\xd8\xff\xe0"},
        )

        recorder.append(
            sim_time_ns=2000000,
            action=[0.0] * 22,
            joint_position=[0.0] * 19,
            joint_velocity=[0.0] * 19,
            next_joint_position=[0.0] * 19,
            command_seq=2,
            camera_frames={"head_camera": b"\xff\xd8\xff\xe0"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = recorder.save(f"{tmpdir}/ep.h5", metadata={})

            with h5py.File(path, "r") as f:
                # 验证只保存了使用的相机
                assert "camera_frames" in f
                assert "head_camera" in f["camera_frames"]
                assert "base_camera" not in f["camera_frames"]
                assert "left_wrist_camera" not in f["camera_frames"]
                assert "right_wrist_camera" not in f["camera_frames"]

                # 验证帧数正确
                assert len(f["camera_frames/head_camera/jpeg"]) == 2
