"""Owned, timestamped RGB-D packets shared by all output sinks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _own(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "numpy"):
        return value.numpy().copy()
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:
            pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return value


@dataclass(slots=True)
class FramePacket:
    episode_id: int
    snapshot_id: int | None
    camera_name: str
    frame_id: int
    sample_sim_time_ns: int
    capture_wall_time_ns: int
    calibration_id: str
    rgb: object | None
    depth_m: object | None
    depth_semantics: str
    source_state_seq: int
    rgb_encoding: str
    depth_encoding: str
    imu: dict[str, object] | None = None

    @classmethod
    def owned(cls, **kwargs: Any) -> "FramePacket":
        """Build a packet with independent image memory for asynchronous sinks."""
        kwargs["rgb"] = _own(kwargs.get("rgb"))
        kwargs["depth_m"] = _own(kwargs.get("depth_m"))
        if kwargs.get("imu") is not None:
            kwargs["imu"] = {key: _own(value) for key, value in kwargs["imu"].items()}
        return cls(**kwargs)

    def validate(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if self.sample_sim_time_ns < 0:
            raise ValueError("sample_sim_time_ns must be non-negative")
        if not self.camera_name or not self.calibration_id:
            raise ValueError("camera_name and calibration_id are required")
        if self.depth_semantics not in {"z_depth_ideal_aligned", "z_depth_stereo_noise"}:
            raise ValueError(f"unsupported depth semantics: {self.depth_semantics}")
