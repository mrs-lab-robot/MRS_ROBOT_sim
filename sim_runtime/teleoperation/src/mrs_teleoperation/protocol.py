"""Versioned, ROS-independent command frames for the Arena teleoperation link."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


PROTOCOL_VERSION = 1
_VECTOR_SIZES = {
    "base_twist": 3,
    "left_controller_pose": 7,
    "right_controller_pose": 7,
    "left_grip": 1,
    "right_grip": 1,
    "left_trigger": 1,
    "right_trigger": 1,
    "left_joystick": 2,
    "right_joystick": 2,
    "left_arm_position": 7,
    "right_arm_position": 7,
    "lift_position": 1,
    "lift_velocity": 1,
    "head_position": 2,
    "left_gripper_position": 1,
    "right_gripper_position": 1,
}


class ProtocolError(ValueError):
    """A command frame does not match the public Arena teleop protocol."""


@dataclass(frozen=True)
class CommandFrame:
    """Latest setpoints produced by the ROS teleoperation nodes.

    Joint fields are absolute SI targets; ``lift_velocity`` is a signed m/s jog.
    Missing fields mean that the ROS node has not produced a setpoint for that group.
    """

    seq: int
    source_time_ns: int = 0
    deadman: bool = False
    estop: bool = False
    base_twist: tuple[float, ...] | None = None
    left_controller_pose: tuple[float, ...] | None = None
    right_controller_pose: tuple[float, ...] | None = None
    left_grip: tuple[float, ...] | None = None
    right_grip: tuple[float, ...] | None = None
    left_trigger: tuple[float, ...] | None = None
    right_trigger: tuple[float, ...] | None = None
    left_joystick: tuple[float, ...] | None = None
    right_joystick: tuple[float, ...] | None = None
    button_states: dict[str, bool] | None = None
    left_arm_position: tuple[float, ...] | None = None
    right_arm_position: tuple[float, ...] | None = None
    lift_position: tuple[float, ...] | None = None
    lift_velocity: tuple[float, ...] | None = None
    head_position: tuple[float, ...] | None = None
    left_gripper_position: tuple[float, ...] | None = None
    right_gripper_position: tuple[float, ...] | None = None
    version: int = PROTOCOL_VERSION
    session_id: int = 0

    def __post_init__(self) -> None:
        if self.version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {self.version}")
        if self.seq < 0 or self.session_id < 0 or self.source_time_ns < 0:
            raise ProtocolError("sequence, session, and source timestamp must be non-negative")
        if not isinstance(self.deadman, bool) or not isinstance(self.estop, bool):
            raise ProtocolError("deadman and estop must be booleans")
        for name, expected_size in _VECTOR_SIZES.items():
            values = getattr(self, name)
            if values is None:
                continue
            if len(values) != expected_size:
                raise ProtocolError(f"{name} must contain exactly {expected_size} values")
            if not all(math.isfinite(float(value)) for value in values):
                raise ProtocolError(f"{name} values must be finite")
        if self.button_states is not None and not all(
            isinstance(name, str) and isinstance(value, bool) for name, value in self.button_states.items()
        ):
            raise ProtocolError("button_states must map button names to booleans")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "magic": "MRSAT",
            "version": self.version,
            "type": "command",
            "session_id": self.session_id,
            "seq": self.seq,
            "source_time_ns": self.source_time_ns,
            "deadman": self.deadman,
            "estop": self.estop,
        }
        for name in _VECTOR_SIZES:
            values = getattr(self, name)
            if values is not None:
                result[name] = list(values)
        if self.button_states is not None:
            result["button_states"] = dict(self.button_states)
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CommandFrame":
        if payload.get("magic", "MRSAT") != "MRSAT" or payload.get("type", "command") != "command":
            raise ProtocolError("not an MRSAT command frame")
        if payload.get("version", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {payload.get('version')}")
        vectors: dict[str, tuple[float, ...] | None] = {}
        for name in _VECTOR_SIZES:
            raw_values = payload.get(name)
            if raw_values is None:
                vectors[name] = None
                continue
            if not isinstance(raw_values, (list, tuple)):
                raise ProtocolError(f"{name} must be an array")
            try:
                vectors[name] = tuple(float(value) for value in raw_values)
            except (TypeError, ValueError) as error:
                raise ProtocolError(f"{name} must contain numeric values") from error
        try:
            return cls(
                version=int(payload.get("version", PROTOCOL_VERSION)),
                session_id=int(payload.get("session_id", 0)),
                seq=int(payload["seq"]),
                source_time_ns=int(payload.get("source_time_ns", 0)),
                deadman=payload.get("deadman", False),
                estop=payload.get("estop", False),
                button_states=payload.get("button_states"),
                **vectors,
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ProtocolError):
                raise
            raise ProtocolError(f"invalid command frame: {error}") from error
