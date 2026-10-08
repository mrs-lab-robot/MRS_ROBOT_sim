"""Session configuration for simulation runtime.

Defines physics/control/render frequencies and sensor configuration.
All frequencies must satisfy integer divisibility constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any


class WorkerState(Enum):
    """Isaac Lab worker state machine states."""

    IDLE = "idle"  # Initial state before any configuration
    PREFLIGHT = "preflight"  # Configuration loaded, ready to launch
    LAUNCHING_KIT = "launching_kit"
    BUILDING_ENV = "building_env"
    RESETTING = "resetting"
    WARMING_UP = "warming_up"
    STARTING_SIDECARS = "starting_sidecars"
    HEALTH_CHECK = "health_check"
    ENV_READY = "env_ready"
    RECORDING = "recording"
    SAVING = "saving"
    QC = "qc"
    EXPORTING = "exporting"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True)
class FrequencyConfig:
    """Simulation frequency configuration with validation.

    All frequencies are in Hz. Constraints:
    - physics_hz must be divisible by control_hz
    - physics_hz must be divisible by render_hz
    - output_fps must not exceed render_hz
    - Camera sensors must not exceed render_hz and map to integer render ticks
    - IMU/LiDAR/tactile/localization must map to integer physics or control ticks
    """

    physics_hz: float
    control_hz: float
    render_hz: float
    output_fps: float = 30.0

    def __post_init__(self) -> None:
        """Validate frequency constraints after initialization."""
        # Validation is done explicitly via validate() to allow construction
        # of invalid configs for testing
        pass

    def validate(self) -> None:
        """Validate that all frequency ratios are valid integers.

        Raises:
            ValueError: If any frequency constraint is violated.
        """
        # Validate all frequencies are positive
        for name, freq in [
            ("physics_hz", self.physics_hz),
            ("control_hz", self.control_hz),
            ("render_hz", self.render_hz),
            ("output_fps", self.output_fps),
        ]:
            if not math.isfinite(freq) or freq <= 0:
                raise ValueError(f"{name}必须为正数，当前值: {freq}")

        # Validate physics_hz / control_hz is an integer
        ratio_pc = self.physics_hz / self.control_hz
        if not self._is_integer_ratio(ratio_pc):
            raise ValueError(
                f"physics_hz ({self.physics_hz}) 必须能被 control_hz ({self.control_hz}) 整除，"
                f"当前比值: {ratio_pc:.4f}"
            )

        # Validate physics_hz / render_hz is an integer
        ratio_pr = self.physics_hz / self.render_hz
        if not self._is_integer_ratio(ratio_pr):
            raise ValueError(
                f"physics_hz ({self.physics_hz}) 必须能被 render_hz ({self.render_hz}) 整除，"
                f"当前比值: {ratio_pr:.4f}"
            )

        # Validate output_fps does not exceed render_hz
        if self.output_fps > self.render_hz:
            raise ValueError(
                f"output_fps ({self.output_fps}) 不能超过 render_hz ({self.render_hz})"
            )

        # Validate output_fps maps to integer render ticks
        if self.output_fps > 0:
            ratio_ro = self.render_hz / self.output_fps
            if not self._is_integer_ratio(ratio_ro):
                raise ValueError(
                    f"render_hz ({self.render_hz}) 必须能被 output_fps ({self.output_fps}) 整除，"
                    f"当前比值: {ratio_ro:.4f}"
                )

    def tick_for(self, sensor_type: str, frequency_hz: float) -> tuple[str, int]:
        """Return the simulation clock and integer tick interval for a sensor.

        Cameras are driven by render ticks. Other physical sensors are driven
        by the control clock when possible and otherwise by the physics clock.
        The method deliberately returns an integer interval so a recorder can
        never silently sample a sensor at a fractional tick.
        """
        try:
            frequency = float(frequency_hz)
        except (TypeError, ValueError) as exc:
            raise ValueError("传感器频率必须是有限正数") from exc
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("传感器频率必须是有限正数")

        clock = "render" if sensor_type.lower() == "camera" else "control"
        base_frequency = getattr(self, f"{clock}_hz")
        ratio = base_frequency / frequency
        if not self._is_integer_ratio(ratio):
            clock = "physics"
            base_frequency = self.physics_hz
            ratio = base_frequency / frequency
        if not self._is_integer_ratio(ratio) or ratio < 1:
            raise ValueError(
                f"{sensor_type} 频率 {frequency_hz} Hz 无法映射到整数 {clock} tick"
            )
        return clock, int(round(ratio))

    @staticmethod
    def _is_integer_ratio(value: float, tolerance: float = 1e-6) -> bool:
        """Check if a float is close to an integer."""
        return abs(value - round(value)) < tolerance


@dataclass(frozen=True)
class SensorConfig:
    """Individual sensor configuration."""

    sensor_id: str
    sensor_type: str  # camera, lidar, imu, tactile, localization
    enabled: bool = True
    frequency_hz: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Make sensor config hashable for use in sets/dicts."""
        # params dict is not hashable, so we exclude it or convert to frozenset
        params_tuple = tuple(sorted(self.params.items())) if self.params else ()
        return hash((self.sensor_id, self.sensor_type, self.enabled, self.frequency_hz, params_tuple))


@dataclass(frozen=True)
class SessionConfig:
    """Complete session configuration for a simulation run.

    Contains all parameters needed to launch and configure the Isaac Lab
    environment, sensors, and recording pipeline.

    Immutable and hashable for caching and contract verification.
    """

    session_id: str
    embodiment_id: str
    scene_id: str
    task_id: str

    frequency: FrequencyConfig
    sensors: tuple[SensorConfig, ...] = field(default_factory=tuple)

    headless: bool = False
    gpu_id: int = 0

    episode_root: str = ""
    dataset_id: str = ""
    enable_recording: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    # The fields below are the single-entry runtime contract. They are
    # optional for backwards compatibility with older CLI callers.
    machine_id: str = "local"
    simulation_repo_root: str = ""
    isaac_lab_python: str = ""
    lerobot_python: str = ""
    dataset_root: str = ""
    sim_client_address: str = ""
    teleop_mode: str = "keyboard"
    enable_rviz: bool = False
    enable_rerun: bool = False
    random_seed: int = 0
    capture_fields: tuple[str, ...] = (
        "action",
        "state",
        "task",
    )

    def __post_init__(self) -> None:
        """Ensure sensors is a tuple for immutability."""
        # Convert list to tuple if needed
        if isinstance(self.sensors, list):
            object.__setattr__(self, 'sensors', tuple(self.sensors))
        if isinstance(self.capture_fields, list):
            object.__setattr__(self, "capture_fields", tuple(self.capture_fields))

    def __hash__(self) -> int:
        """Make session config hashable."""
        return hash(self.config_hash)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the frozen session without framework-specific objects."""
        return {
            "schema_version": 1,
            "session_id": self.session_id,
            "embodiment_id": self.embodiment_id,
            "scene_id": self.scene_id,
            "task_id": self.task_id,
            "frequency": {
                "physics_hz": self.frequency.physics_hz,
                "control_hz": self.frequency.control_hz,
                "render_hz": self.frequency.render_hz,
                "output_fps": self.frequency.output_fps,
            },
            "sensors": [
                {
                    "sensor_id": sensor.sensor_id,
                    "sensor_type": sensor.sensor_type,
                    "enabled": sensor.enabled,
                    "frequency_hz": sensor.frequency_hz,
                    "params": sensor.params,
                }
                for sensor in self.sensors
            ],
            "headless": self.headless,
            "gpu_id": self.gpu_id,
            "episode_root": self.episode_root,
            "dataset_id": self.dataset_id,
            "enable_recording": self.enable_recording,
            "metadata": self.metadata,
            "machine_id": self.machine_id,
            "simulation_repo_root": self.simulation_repo_root,
            "isaac_lab_python": self.isaac_lab_python,
            "lerobot_python": self.lerobot_python,
            "dataset_root": self.dataset_root,
            "sim_client_address": self.sim_client_address,
            "teleop_mode": self.teleop_mode,
            "enable_rviz": self.enable_rviz,
            "enable_rerun": self.enable_rerun,
            "random_seed": self.random_seed,
            "capture_fields": list(self.capture_fields),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionConfig":
        """Build a session from a YAML/JSON-compatible mapping."""
        frequency_data = data.get("frequency")
        if not isinstance(frequency_data, dict):
            raise ValueError("frequency 必须是字典")
        frequency = FrequencyConfig(
            physics_hz=frequency_data["physics_hz"],
            control_hz=frequency_data["control_hz"],
            render_hz=frequency_data["render_hz"],
            output_fps=frequency_data.get("output_fps", 30.0),
        )
        sensors = tuple(
            SensorConfig(
                sensor_id=str(sensor["sensor_id"]),
                sensor_type=str(sensor["sensor_type"]),
                enabled=bool(sensor.get("enabled", True)),
                frequency_hz=sensor.get("frequency_hz"),
                params=dict(sensor.get("params", {})),
            )
            for sensor in data.get("sensors", ())
        )
        return cls(
            session_id=str(data.get("session_id", "")),
            embodiment_id=str(data.get("embodiment_id", "")),
            scene_id=str(data.get("scene_id", "")),
            task_id=str(data.get("task_id", "")),
            frequency=frequency,
            sensors=sensors,
            headless=bool(data.get("headless", False)),
            gpu_id=data.get("gpu_id", 0),
            episode_root=str(data.get("episode_root", "")),
            dataset_id=str(data.get("dataset_id", "")),
            enable_recording=bool(data.get("enable_recording", True)),
            metadata=dict(data.get("metadata", {})),
            machine_id=str(data.get("machine_id", "local")),
            simulation_repo_root=str(data.get("simulation_repo_root", "")),
            isaac_lab_python=str(data.get("isaac_lab_python", "")),
            lerobot_python=str(data.get("lerobot_python", "")),
            dataset_root=str(data.get("dataset_root", "")),
            sim_client_address=str(data.get("sim_client_address", "")),
            teleop_mode=str(data.get("teleop_mode", "keyboard")),
            enable_rviz=bool(data.get("enable_rviz", False)),
            enable_rerun=bool(data.get("enable_rerun", False)),
            random_seed=data.get("random_seed", 0),
            capture_fields=tuple(data.get("capture_fields", ("action", "state", "task"))),
        )

    @property
    def config_hash(self) -> str:
        """Return the stable SHA-256 identity of the frozen session."""
        encoded = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        """Validate the entire session configuration.

        Raises:
            ValueError: If any validation check fails.
        """
        # Validate session_id non-empty
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id不能为空")

        # Validate embodiment/scene/task IDs
        if not self.embodiment_id or not self.embodiment_id.strip():
            raise ValueError("embodiment_id不能为空")
        if not self.scene_id or not self.scene_id.strip():
            raise ValueError("scene_id不能为空")
        if not self.task_id or not self.task_id.strip():
            raise ValueError("task_id不能为空")

        if not self.machine_id or not self.machine_id.strip():
            raise ValueError("machine_id不能为空")
        if self.teleop_mode not in {"vr", "keyboard"}:
            raise ValueError("teleop_mode必须是vr或keyboard")
        if isinstance(self.gpu_id, bool) or not isinstance(self.gpu_id, int) or self.gpu_id < 0:
            raise ValueError("gpu_id必须是非负整数")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed必须是非负整数")
        if not self.capture_fields or any(
            not isinstance(field_name, str) or not field_name.strip()
            for field_name in self.capture_fields
        ):
            raise ValueError("capture_fields不能为空")
        if self.enable_recording and self.dataset_root.strip() and not self.dataset_id.strip():
            raise ValueError("LeRobot 数据集输出已启用时必须填写 dataset_id")
        if self.enable_recording and self.dataset_root.strip() and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?",
            self.dataset_id.strip(),
        ):
            raise ValueError("dataset_id 必须是安全的 repo 或 namespace/repo 标识")

        # Validate frequency configuration
        self.frequency.validate()

        # Validate sensor configurations
        seen_ids = set()
        for sensor in self.sensors:
            if not sensor.sensor_id or not sensor.sensor_id.strip():
                raise ValueError("sensor_id不能为空")
            if sensor.sensor_id in seen_ids:
                raise ValueError(f"重复的sensor_id: {sensor.sensor_id}")
            seen_ids.add(sensor.sensor_id)

            # Validate sensor frequency based on sensor type
            if sensor.enabled and sensor.frequency_hz is not None:
                if sensor.frequency_hz <= 0:
                    raise ValueError(f"传感器 {sensor.sensor_id} 频率必须为正数")

                self._validate_sensor_frequency(sensor)

    def _validate_sensor_frequency(self, sensor: SensorConfig) -> None:
        """Validate sensor frequency against appropriate simulation tick rate.

        Camera sensors must map to render ticks.
        IMU/LiDAR/tactile/localization must map to physics or control ticks.
        """
        sensor_type = sensor.sensor_type.lower()

        if sensor_type == "camera":
            # Camera must not exceed render_hz and must map to integer render ticks
            if sensor.frequency_hz > self.frequency.render_hz:
                raise ValueError(
                    f"相机传感器 {sensor.sensor_id} 频率 ({sensor.frequency_hz} Hz) "
                    f"不能超过 render_hz ({self.frequency.render_hz} Hz)"
                )
            ratio = self.frequency.render_hz / sensor.frequency_hz
            if not self._is_integer_ratio(ratio):
                raise ValueError(
                    f"相机传感器 {sensor.sensor_id} 频率 ({sensor.frequency_hz} Hz) "
                    f"必须整除 render_hz ({self.frequency.render_hz} Hz)，当前比值: {ratio:.4f}"
                )

        elif sensor_type in ("imu", "lidar", "tactile", "localization"):
            # These sensors should map to physics or control ticks
            # Try control_hz first (most common), then physics_hz
            control_ratio = self.frequency.control_hz / sensor.frequency_hz
            physics_ratio = self.frequency.physics_hz / sensor.frequency_hz

            if self._is_integer_ratio(control_ratio):
                # Maps to control ticks - valid
                pass
            elif self._is_integer_ratio(physics_ratio):
                # Maps to physics ticks - valid
                pass
            else:
                raise ValueError(
                    f"传感器 {sensor.sensor_id} ({sensor.sensor_type}) 频率 ({sensor.frequency_hz} Hz) "
                    f"必须整除 control_hz ({self.frequency.control_hz} Hz) 或 physics_hz ({self.frequency.physics_hz} Hz)，"
                    f"当前比值: control={control_ratio:.4f}, physics={physics_ratio:.4f}"
                )
        else:
            # Unknown sensor type - apply generic render_hz constraint for safety
            if sensor.frequency_hz > self.frequency.render_hz:
                raise ValueError(
                    f"传感器 {sensor.sensor_id} 频率 ({sensor.frequency_hz} Hz) "
                    f"不能超过 render_hz ({self.frequency.render_hz} Hz)"
                )

    @staticmethod
    def _is_integer_ratio(value: float, tolerance: float = 1e-6) -> bool:
        """Check if a float is close to an integer."""
        return abs(value - round(value)) < tolerance
