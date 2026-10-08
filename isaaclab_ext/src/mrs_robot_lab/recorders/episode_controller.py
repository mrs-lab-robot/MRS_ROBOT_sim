"""Episode lifecycle and provenance gate for the single-entry recorder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mrs_robot_lab.recorders.teleop_episode import TeleopEpisodeRecorder


REQUIRED_PROVENANCE = frozenset(
    {"episode_id", "config_hash", "contract_hash", "random_seed", "input_source"}
)


@dataclass(frozen=True)
class EpisodeQualityReport:
    """Quality decision made before an episode can enter a dataset."""

    passed: bool
    reasons: tuple[str, ...] = ()


class EpisodeQualityError(RuntimeError):
    """Raised when an episode is incomplete or lacks provenance."""

    def __init__(self, report: EpisodeQualityReport) -> None:
        self.report = report
        super().__init__("episode quality check failed: " + "; ".join(report.reasons))


def quality_check(metadata: dict[str, Any], num_steps: int) -> EpisodeQualityReport:
    """Check the minimum evidence required for a trainable episode."""
    reasons = [key for key in sorted(REQUIRED_PROVENANCE) if key not in metadata]
    if num_steps <= 0:
        reasons.append("episode contains no samples")
    return EpisodeQualityReport(not reasons, tuple(reasons))


class EpisodeController:
    """Own one recorder buffer and gate its final HDF5 commit."""

    def __init__(self, recorder: TeleopEpisodeRecorder | None = None) -> None:
        self.recorder = recorder or TeleopEpisodeRecorder()
        self._metadata: dict[str, Any] | None = None
        self.output_path: Path | None = None

    @property
    def num_steps(self) -> int:
        return self.recorder.num_steps

    def start(self, metadata: dict[str, Any]) -> None:
        if self._metadata is not None and self.recorder.num_steps:
            raise RuntimeError("an episode is already active")
        self._metadata = dict(metadata)
        self.output_path = None
        self.recorder.start()

    def append(self, **sample: Any) -> None:
        if self._metadata is None:
            raise RuntimeError("no episode is active")
        self.recorder.append(**sample)

    def annotate(self, metadata: dict[str, Any]) -> None:
        """Merge runtime events into the active episode's persisted metadata."""
        if self._metadata is None:
            raise RuntimeError("no episode is active")
        if not isinstance(metadata, dict):
            raise TypeError("episode annotations must be a dictionary")
        self._metadata.update(metadata)

    def save(self, output_path: str | Path) -> Path:
        if self._metadata is None:
            raise RuntimeError("no episode is active")
        report = quality_check(self._metadata, self.recorder.num_steps)
        if not report.passed:
            raise EpisodeQualityError(report)
        metadata = dict(self._metadata)
        metadata["quality_status"] = (
            "failed" if metadata.get("recording_faults") else "passed"
        )
        path = self.recorder.save(output_path, metadata=metadata)
        self.output_path = path
        self._metadata = None
        return path

    def discard(self) -> None:
        self.recorder.discard()
        self._metadata = None
        self.output_path = None
