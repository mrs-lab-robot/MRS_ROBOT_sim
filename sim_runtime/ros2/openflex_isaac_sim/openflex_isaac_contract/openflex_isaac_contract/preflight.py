"""Static validation performed before starting Isaac Sim/Kit."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

from openflex_isaac_contract.embodiment_spec import EmbodimentSpec
from openflex_isaac_contract.session_config import SessionConfig


@dataclass(frozen=True)
class PreflightReport:
    """Structured preflight result suitable for a GUI checklist."""

    checks: dict[str, str] = field(default_factory=dict)
    failed_checks: dict[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failed_checks


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def run_preflight(
    config: SessionConfig,
    *,
    embodiment: EmbodimentSpec | None = None,
    scene: object | None = None,
    task: object | None = None,
) -> PreflightReport:
    """Validate the static session contract without importing Kit or Arena."""
    checks: dict[str, str] = {}
    failures: dict[str, str] = {}

    try:
        config.validate()
    except ValueError as exc:
        failures["session"] = str(exc)
    else:
        checks["session"] = "SessionConfig 有效"

    if embodiment is not None:
        if embodiment.embodiment_id != config.embodiment_id:
            failures["embodiment"] = (
                f"选择的机器人是 {embodiment.embodiment_id}，"
                f"SessionConfig 要求 {config.embodiment_id}"
            )
        elif not embodiment.supports_teleop(config.teleop_mode):
            failures["teleop_mode"] = f"{config.teleop_mode} 不在机器人支持的遥操作方式中"
        else:
            checks["embodiment"] = "机器人与遥操作方式兼容"

    if scene is not None:
        scene_id = _value(scene, "scene_id", _value(scene, "id", ""))
        if str(scene_id) != config.scene_id:
            failures["scene"] = f"场景 {scene_id} 与 SessionConfig 的 {config.scene_id} 不一致"
        else:
            checks["scene"] = "场景配置匹配"

    if task is not None:
        task_id = _value(task, "task_id", _value(task, "id", ""))
        if str(task_id) != config.task_id:
            failures["task"] = f"任务 {task_id} 与 SessionConfig 的 {config.task_id} 不一致"
        capabilities = set(_value(task, "capabilities", ()) or ())
        if "capture" not in capabilities:
            failures["capabilities"] = "任务没有声明 capture 能力，不能进入示教采集入口"
        else:
            checks["capabilities"] = "任务声明支持采集"
        allowed_modes = set(_value(task, "allowed_teleop_modes", ()) or ())
        if allowed_modes and config.teleop_mode not in allowed_modes:
            failures["teleop_mode"] = f"任务不支持 {config.teleop_mode} 遥操作"
        required_sensors = tuple(_value(task, "required_sensors", ()) or ())
        enabled_sensors = {sensor.sensor_id for sensor in config.sensors if sensor.enabled}
        missing = [sensor_id for sensor_id in required_sensors if sensor_id not in enabled_sensors]
        unsupported = [
            sensor_id
            for sensor_id in required_sensors
            if embodiment is not None and not embodiment.supports_sensor(sensor_id)
        ]
        if missing or unsupported:
            details = []
            if missing:
                details.append(f"未启用: {', '.join(missing)}")
            if unsupported:
                details.append(f"机器人不支持: {', '.join(unsupported)}")
            failures["required_sensors"] = "；".join(details)
        else:
            checks["required_sensors"] = "任务必需传感器已启用"

    return PreflightReport(checks=checks, failed_checks=failures)
