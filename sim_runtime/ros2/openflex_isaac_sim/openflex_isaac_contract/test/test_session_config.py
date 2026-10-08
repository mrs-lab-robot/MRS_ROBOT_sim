"""Tests for session configuration contract.

Tests frequency validation, sensor configuration, and immutability.
"""

import unittest

from openflex_isaac_contract.session_config import (
    FrequencyConfig,
    SensorConfig,
    SessionConfig,
    WorkerState,
)


class TestWorkerState(unittest.TestCase):
    """Test WorkerState enum."""

    def test_has_idle_state(self):
        """状态机应包含IDLE状态"""
        self.assertEqual(WorkerState.IDLE.value, "idle")

    def test_has_all_required_states(self):
        """状态机应包含所有必需状态"""
        required_states = [
            "idle", "preflight", "launching_kit", "building_env", "resetting",
            "warming_up", "starting_sidecars", "health_check", "env_ready",
            "recording", "saving", "qc", "exporting", "stopping", "failed"
        ]
        actual_states = [state.value for state in WorkerState]
        for required in required_states:
            self.assertIn(required, actual_states)


class TestFrequencyConfig(unittest.TestCase):
    """Test frequency configuration validation."""

    def test_valid_frequencies(self):
        """有效的频率配置应通过验证"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=30.0,
            render_hz=30.0,
            output_fps=30.0,
        )
        config.validate()  # Should not raise

    def test_physics_control_must_be_integer_ratio(self):
        """physics_hz/control_hz必须为整数"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=35.0,  # 120/35 = 3.428... not integer
            render_hz=30.0,
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("physics_hz", str(ctx.exception))
        self.assertIn("control_hz", str(ctx.exception))

    def test_physics_render_must_be_integer_ratio(self):
        """physics_hz/render_hz必须为整数"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=30.0,
            render_hz=35.0,  # 120/35 = 3.428... not integer
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("physics_hz", str(ctx.exception))
        self.assertIn("render_hz", str(ctx.exception))

    def test_output_fps_cannot_exceed_render_hz(self):
        """output_fps不能超过render_hz"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=30.0,
            render_hz=30.0,
            output_fps=60.0,  # Exceeds render_hz
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("output_fps", str(ctx.exception))
        self.assertIn("render_hz", str(ctx.exception))

    def test_output_fps_must_divide_render_hz(self):
        """render_hz必须能被output_fps整除"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=30.0,
            render_hz=30.0,
            output_fps=25.0,  # 30/25 = 1.2 not integer
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("output_fps", str(ctx.exception))
        self.assertIn("render_hz", str(ctx.exception))

    def test_negative_frequency_rejected(self):
        """负频率应被拒绝"""
        config = FrequencyConfig(
            physics_hz=-120.0,
            control_hz=30.0,
            render_hz=30.0,
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("正数", str(ctx.exception))

    def test_zero_frequency_rejected(self):
        """零频率应被拒绝"""
        config = FrequencyConfig(
            physics_hz=0.0,
            control_hz=30.0,
            render_hz=30.0,
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("正数", str(ctx.exception))

    def test_frequency_config_is_frozen(self):
        """FrequencyConfig应该是不可变的"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=30.0,
            render_hz=30.0,
        )
        with self.assertRaises(Exception):  # FrozenInstanceError or AttributeError
            config.physics_hz = 60.0

    def test_frequency_config_is_hashable(self):
        """FrequencyConfig应该可哈希"""
        config = FrequencyConfig(
            physics_hz=120.0,
            control_hz=30.0,
            render_hz=30.0,
        )
        # Should not raise
        hash(config)
        # Can be used in set
        {config}


class TestSensorConfig(unittest.TestCase):
    """Test sensor configuration."""

    def test_sensor_config_is_frozen(self):
        """SensorConfig应该是不可变的"""
        sensor = SensorConfig(
            sensor_id="cam_front",
            sensor_type="camera",
            enabled=True,
            frequency_hz=30.0,
        )
        with self.assertRaises(Exception):
            sensor.enabled = False

    def test_sensor_config_is_hashable(self):
        """SensorConfig应该可哈希"""
        sensor = SensorConfig(
            sensor_id="cam_front",
            sensor_type="camera",
            enabled=True,
            frequency_hz=30.0,
        )
        # Should not raise
        hash(sensor)
        {sensor}


class TestSessionConfig(unittest.TestCase):
    """Test session configuration validation."""

    def test_valid_config(self):
        """有效配置应通过验证"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(
                physics_hz=120.0,
                control_hz=30.0,
                render_hz=30.0,
            ),
        )
        config.validate()  # Should not raise

    def test_recording_dataset_id_must_be_a_safe_repository_identifier(self):
        config = SessionConfig(
            session_id="test_dataset_id",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            episode_root="/data/episodes",
            dataset_root="/data/lerobot",
            dataset_id="../outside",
        )

        with self.assertRaisesRegex(ValueError, "dataset_id"):
            config.validate()

    def test_empty_session_id_rejected(self):
        """空session_id应被拒绝"""
        config = SessionConfig(
            session_id="",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("session_id", str(ctx.exception))

    def test_duplicate_sensor_ids_rejected(self):
        """重复的sensor_id应被拒绝"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("cam1", "camera", frequency_hz=30.0),
                SensorConfig("cam1", "camera", frequency_hz=15.0),  # Duplicate
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("重复", str(ctx.exception))
        self.assertIn("cam1", str(ctx.exception))

    def test_camera_frequency_must_not_exceed_render(self):
        """相机频率不能超过render_hz"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("cam_front", "camera", frequency_hz=60.0),  # Exceeds render
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("相机", str(ctx.exception))
        self.assertIn("render_hz", str(ctx.exception))

    def test_camera_frequency_must_divide_render(self):
        """相机频率必须整除render_hz"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("cam_front", "camera", frequency_hz=20.0),  # 30/20=1.5
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("相机", str(ctx.exception))
        self.assertIn("render_hz", str(ctx.exception))

    def test_imu_frequency_must_divide_physics_or_control(self):
        """IMU频率必须整除physics_hz或control_hz"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("imu_base", "imu", frequency_hz=25.0),  # Neither 120/25=4.8 nor 30/25=1.2
            ),
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("imu", str(ctx.exception).lower())
        self.assertIn("control_hz", str(ctx.exception))

    def test_imu_frequency_divides_control_hz_valid(self):
        """IMU频率整除control_hz时有效"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("imu_base", "imu", frequency_hz=10.0),  # 30/10=3
            ),
        )
        config.validate()  # Should not raise

    def test_lidar_frequency_divides_physics_hz_valid(self):
        """LiDAR频率整除physics_hz时有效"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("lidar_mid360", "lidar", frequency_hz=60.0),  # 120/60=2
            ),
        )
        config.validate()  # Should not raise

    def test_disabled_sensor_skips_frequency_validation(self):
        """禁用的传感器跳过频率验证"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=(
                SensorConfig("cam_broken", "camera", enabled=False, frequency_hz=99.0),
            ),
        )
        config.validate()  # Should not raise even with invalid frequency

    def test_session_config_is_frozen(self):
        """SessionConfig应该是不可变的"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
        )
        with self.assertRaises(Exception):
            config.session_id = "new_id"

    def test_session_config_is_hashable(self):
        """SessionConfig应该可哈希"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
        )
        # Should not raise
        hash(config)
        {config}

    def test_sensors_list_converted_to_tuple(self):
        """sensors列表应自动转换为tuple"""
        config = SessionConfig(
            session_id="test_001",
            embodiment_id="openflex",
            scene_id="flat_navigation",
            task_id="navigation_to_goal",
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            sensors=[  # Pass as list
                SensorConfig("cam1", "camera", frequency_hz=30.0),
            ],
        )
        self.assertIsInstance(config.sensors, tuple)


if __name__ == "__main__":
    unittest.main()
