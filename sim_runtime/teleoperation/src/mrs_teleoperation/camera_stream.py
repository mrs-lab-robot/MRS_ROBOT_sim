"""Small local UDP protocol for forwarding Arena RGB observations to ROS 2."""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
import time
from typing import Final


CAMERA_IDS: Final = {
    "base_d435": 1,
    "head_d435": 2,
    "left_wrist_d405": 3,
    "right_wrist_d405": 4,
}
_CAMERA_NAMES = {camera_id: name for name, camera_id in CAMERA_IDS.items()}
_ROS_FRAME_IDS: Final = {
    "base_d435": "cam_base",
    "head_d435": "cam_head",
    "left_wrist_d405": "cam_left",
    "right_wrist_d405": "cam_right",
}
_MAGIC: Final = b"MRC1"
_HEADER: Final = struct.Struct("!4sBIQHHI")
MAX_JPEG_BYTES: Final = 8 * 1024 * 1024
MAX_UDP_DATAGRAM_BYTES: Final = 65_507
DEFAULT_CHUNK_SIZE: Final = 1_200
PREVIEW_REGISTER_PACKET: Final = b"MRSR1"
PREVIEW_PEER_LEASE_S: Final = 3.0
CAMERA_CHUNK_SIZE_MIN: Final = 200
CAMERA_CHUNK_SIZE_MAX: Final = MAX_UDP_DATAGRAM_BYTES


def validate_camera_chunk_size(chunk_size: int) -> int:
    """Validate the teleop CLI's configured camera payload chunk size."""
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or not CAMERA_CHUNK_SIZE_MIN <= chunk_size <= CAMERA_CHUNK_SIZE_MAX
    ):
        raise ValueError(
            f"camera_chunk_size 必须在 {CAMERA_CHUNK_SIZE_MIN} 到 "
            f"{CAMERA_CHUNK_SIZE_MAX} 之间"
        )
    return chunk_size


class PreviewPeerRegistry:
    """Track the most recent valid preview registration using a short lease."""

    def __init__(self, *, lease_s: float = PREVIEW_PEER_LEASE_S):
        if lease_s <= 0:
            raise ValueError("PreviewPeerRegistry lease_s must be positive")
        self.lease_s = float(lease_s)
        self._active_peer: tuple[str, int] | None = None
        self._expires_at = 0.0

    @property
    def active_peer(self) -> tuple[str, int] | None:
        return self._active_peer

    def register(
        self,
        packet: bytes,
        peer: tuple[str, int],
        *,
        now: float | None = None,
    ) -> bool:
        """Accept only the exact registration token and refresh its lease.

        Return True when the registry becomes active or switches to a different
        peer; the sender can log those transitions without logging heartbeats.
        """
        if not isinstance(packet, bytes) or packet != PREVIEW_REGISTER_PACKET:
            return False
        current_time = time.monotonic() if now is None else float(now)
        newly_active = (
            self._active_peer is None
            or current_time >= self._expires_at
            or peer != self._active_peer
        )
        self._active_peer = peer
        self._expires_at = current_time + self.lease_s
        return newly_active

    def expire(self, *, now: float | None = None) -> bool:
        """Clear an expired peer, returning whether the active state flipped."""
        current_time = time.monotonic() if now is None else float(now)
        if self._active_peer is None or current_time < self._expires_at:
            return False
        self._active_peer = None
        self._expires_at = 0.0
        return True

    def target(
        self,
        fallback: tuple[str, int],
        *,
        now: float | None = None,
    ) -> tuple[str, int]:
        """Return the leased peer while active, otherwise the legacy target."""
        self.expire(now=now)
        return self._active_peer if self._active_peer is not None else fallback


def camera_topic_frame(camera_name: str) -> str:
    """Return the existing LeRobot/ROS camera frame label for an Arena camera."""
    try:
        return _ROS_FRAME_IDS[camera_name]
    except KeyError as error:
        raise ValueError(f"未知 Arena 相机：{camera_name}") from error


def encode_camera_frame(
    camera_name: str,
    *,
    frame_id: int,
    sim_time_ns: int,
    jpeg_bytes: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[bytes, ...]:
    """Encode one JPEG into independently droppable local UDP datagrams."""
    try:
        camera_id = CAMERA_IDS[camera_name]
    except KeyError as error:
        raise ValueError(f"未知 Arena 相机：{camera_name}") from error
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or not 0 <= frame_id <= 0xFFFFFFFF:
        raise ValueError("frame_id 必须是 uint32")
    if isinstance(sim_time_ns, bool) or not isinstance(sim_time_ns, int) or not 0 <= sim_time_ns <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("sim_time_ns 必须是 uint64")
    if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes or len(jpeg_bytes) > MAX_JPEG_BYTES:
        raise ValueError(f"JPEG 帧长度必须在 1 到 {MAX_JPEG_BYTES} 字节之间")
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
        or chunk_size > MAX_UDP_DATAGRAM_BYTES
    ):
        raise ValueError("chunk_size 超出 UDP 数据报上限")

    payload_chunk_size = min(chunk_size, MAX_UDP_DATAGRAM_BYTES - _HEADER.size)
    chunk_count = (len(jpeg_bytes) + payload_chunk_size - 1) // payload_chunk_size
    if chunk_count > 0xFFFF:
        raise ValueError("JPEG 分片数量超过协议上限")
    packets = []
    for chunk_index in range(chunk_count):
        start = chunk_index * payload_chunk_size
        payload = jpeg_bytes[start:start + payload_chunk_size]
        header = _HEADER.pack(
            _MAGIC, camera_id, frame_id, sim_time_ns, chunk_index,
            chunk_count, len(jpeg_bytes),
        )
        packets.append(header + payload)
    return tuple(packets)


@dataclass
class _PendingFrame:
    sim_time_ns: int
    total_bytes: int
    chunk_count: int
    updated_at: float
    chunks: dict[int, bytes] = field(default_factory=dict)


class CameraFrameAssembler:
    """Reassemble camera datagrams, dropping stale/incomplete frames safely."""

    def __init__(
        self,
        *,
        timeout_s: float = 1.0,
        max_pending_frames: int = 24,
        max_frame_bytes: int = MAX_JPEG_BYTES,
    ):
        if timeout_s <= 0 or max_pending_frames < 1 or max_frame_bytes < 1:
            raise ValueError("CameraFrameAssembler limits must be positive")
        self.timeout_s = float(timeout_s)
        self.max_pending_frames = int(max_pending_frames)
        self.max_frame_bytes = min(int(max_frame_bytes), MAX_JPEG_BYTES)
        self._pending: dict[tuple[int, int], _PendingFrame] = {}

    def feed(
        self,
        packet: bytes,
        *,
        now: float | None = None,
    ) -> tuple[str, int, bytes] | None:
        current_time = time.monotonic() if now is None else float(now)
        self._discard_expired(current_time)
        if not isinstance(packet, bytes) or len(packet) <= _HEADER.size:
            return None
        try:
            magic, camera_id, frame_id, sim_time_ns, chunk_index, chunk_count, total_bytes = (
                _HEADER.unpack_from(packet)
            )
        except struct.error:
            return None
        camera_name = _CAMERA_NAMES.get(camera_id)
        payload = packet[_HEADER.size:]
        if (
            magic != _MAGIC
            or camera_name is None
            or chunk_count < 1
            or chunk_count > self.max_frame_bytes
            or chunk_index >= chunk_count
            or total_bytes < 1
            or total_bytes > self.max_frame_bytes
            or len(payload) > total_bytes
        ):
            return None

        key = (camera_id, frame_id)
        pending = self._pending.get(key)
        if pending is None:
            if len(self._pending) >= self.max_pending_frames:
                oldest_key = min(self._pending, key=lambda item: self._pending[item].updated_at)
                del self._pending[oldest_key]
            pending = _PendingFrame(sim_time_ns, total_bytes, chunk_count, current_time)
            self._pending[key] = pending
        elif (
            pending.sim_time_ns != sim_time_ns
            or pending.total_bytes != total_bytes
            or pending.chunk_count != chunk_count
        ):
            del self._pending[key]
            return None

        pending.updated_at = current_time
        previous = pending.chunks.get(chunk_index)
        if previous is not None:
            if previous != payload:
                del self._pending[key]
            return None
        pending.chunks[chunk_index] = payload
        if len(pending.chunks) != pending.chunk_count:
            return None

        jpeg_bytes = b"".join(pending.chunks[index] for index in range(pending.chunk_count))
        del self._pending[key]
        if len(jpeg_bytes) != pending.total_bytes:
            return None
        return camera_name, pending.sim_time_ns, jpeg_bytes

    def _discard_expired(self, now: float) -> None:
        expired = [
            key for key, frame in self._pending.items()
            if now - frame.updated_at > self.timeout_s
        ]
        for key in expired:
            del self._pending[key]
