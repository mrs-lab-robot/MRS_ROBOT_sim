"""Wall-clock watchdog and replay protection for external commands."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from mrs_teleoperation.protocol import CommandFrame


@dataclass
class CommandWatchdog:
    timeout_seconds: float = 0.25
    latest: CommandFrame | None = None
    received_at: float | None = None
    _last_now: float | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # A non-finite or non-positive timeout would keep a stale deadman
        # command valid forever (e.g. float("inf")), so reject it up front.
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a finite positive number")

    def accept(self, frame: CommandFrame, now: float) -> bool:
        if not math.isfinite(now):
            return False
        if self._last_now is not None and now < self._last_now:
            return False
        self._last_now = now
        if self.received_at is not None and now < self.received_at:
            return False
        if self.latest is not None:
            if frame.session_id < self.latest.session_id:
                return False
            if frame.session_id == self.latest.session_id and frame.seq <= self.latest.seq:
                return False
        self.latest = frame
        self.received_at = now
        return True

    def motion_enabled(self, now: float) -> bool:
        if not math.isfinite(now):
            return False
        if self._last_now is not None and now < self._last_now:
            return False
        self._last_now = now
        if self.latest is None or self.received_at is None:
            return False
        if now < self.received_at:
            return False
        return (
            now - self.received_at <= self.timeout_seconds
            and self.latest.deadman
            and not self.latest.estop
        )

    def reset(self) -> None:
        self.latest = None
        self.received_at = None
