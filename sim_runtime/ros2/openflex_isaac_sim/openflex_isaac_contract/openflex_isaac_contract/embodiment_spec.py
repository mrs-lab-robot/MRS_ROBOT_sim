"""Framework-independent robot embodiment capability specification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SensorCapability:
    """A sensor role the embodiment can instantiate."""

    sensor_id: str
    sensor_type: str
    supported_frequencies_hz: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SensorCapability":
        sensor_id = str(data.get("sensor_id", data.get("id", ""))).strip()
        sensor_type = str(data.get("sensor_type", data.get("type", ""))).strip()
        if not sensor_id or not sensor_type:
            raise ValueError("传感器能力必须包含 sensor_id 和 sensor_type")
        frequencies = data.get("supported_frequencies_hz", data.get("frequencies_hz", ()))
        if not isinstance(frequencies, (list, tuple)):
            raise ValueError(f"传感器 {sensor_id} 的频率列表无效")
        return cls(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            supported_frequencies_hz=tuple(float(value) for value in frequencies),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class EmbodimentSpec:
    """Robot asset, action and sensor capabilities shared by all runtimes."""

    spec_version: str
    embodiment_id: str
    display_name: str
    asset_path: str
    teleop_modes: tuple[str, ...] = ()
    sensors: tuple[SensorCapability, ...] = ()
    capabilities: tuple[str, ...] = ()
    action_contract: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbodimentSpec":
        embodiment_id = str(data.get("embodiment_id", data.get("id", ""))).strip()
        if not embodiment_id:
            raise ValueError("embodiment_id不能为空")
        sensors = tuple(
            SensorCapability.from_dict(sensor)
            for sensor in data.get("sensors", ())
        )
        sensor_ids = [sensor.sensor_id for sensor in sensors]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError("embodiment传感器ID不能重复")
        return cls(
            spec_version=str(data.get("spec_version", "1.0")),
            embodiment_id=embodiment_id,
            display_name=str(data.get("display_name", embodiment_id)),
            asset_path=str(data.get("asset_path", "")),
            teleop_modes=tuple(str(value) for value in data.get("teleop_modes", ())),
            sensors=sensors,
            capabilities=tuple(str(value) for value in data.get("capabilities", ())),
            action_contract=dict(data.get("action_contract", {})),
        )

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(sensor.sensor_id for sensor in self.sensors)

    def supports_sensor(self, sensor_id: str) -> bool:
        return str(sensor_id).strip() in self.sensor_ids

    def supports_teleop(self, mode: str) -> bool:
        return str(mode).strip() in self.teleop_modes

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "embodiment_id": self.embodiment_id,
            "display_name": self.display_name,
            "asset_path": self.asset_path,
            "teleop_modes": list(self.teleop_modes),
            "sensors": [
                {
                    "sensor_id": sensor.sensor_id,
                    "sensor_type": sensor.sensor_type,
                    "supported_frequencies_hz": list(sensor.supported_frequencies_hz),
                    "metadata": sensor.metadata,
                }
                for sensor in self.sensors
            ],
            "capabilities": list(self.capabilities),
            "action_contract": self.action_contract,
        }
