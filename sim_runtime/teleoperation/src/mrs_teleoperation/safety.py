"""Wall-clock watchdog and replay protection for external commands."""

from __future__ import annotations

from dataclasses import dataclass

from mrs_teleoperation.protocol import CommandFrame


@dataclass
class CommandWatchdog:
    timeout_seconds: float = 0.25
    latest: CommandFrame | None = None
    received_at: float | None = None

    def accept(self, frame: CommandFrame, now: float) -> bool:
        if self.latest is not None:
            if frame.session_id < self.latest.session_id:
                return False
            if frame.session_id == self.latest.session_id and frame.seq <= self.latest.seq:
                return False
        self.latest = frame
        self.received_at = now
        return True

    def motion_enabled(self, now: float) -> bool:
        if self.latest is None or self.received_at is None:
            return False
        return (
            now - self.received_at <= self.timeout_seconds
            and self.latest.deadman
            and not self.latest.estop
        )

    def reset(self) -> None:
        self.latest = None
        self.received_at = None
