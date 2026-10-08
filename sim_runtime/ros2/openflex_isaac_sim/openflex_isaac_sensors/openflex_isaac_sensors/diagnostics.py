"""Auditable simulation and sensor timing statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


@dataclass
class SensorDiagnostics:
    name: str
    physics_sim_times: list[float] = field(default_factory=list)
    physics_wall_times: list[int] = field(default_factory=list)
    frame_sim_times: list[float] = field(default_factory=list)
    frame_wall_times: list[int] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)

    def reset_measurement(self) -> None:
        """Drop warm-up samples before the measured run starts."""

        self.physics_sim_times.clear()
        self.physics_wall_times.clear()
        self.frame_sim_times.clear()
        self.frame_wall_times.clear()
        self.frame_ids.clear()

    def record_physics(self, sim_time_s: float, wall_time_ns: int | None = None) -> None:
        self.physics_sim_times.append(float(sim_time_s))
        self.physics_wall_times.append(wall_time_ns or time.perf_counter_ns())

    def record_frame(self, frame_id: int, sim_time_s: float, wall_time_ns: int | None = None) -> None:
        self.frame_ids.append(int(frame_id))
        self.frame_sim_times.append(float(sim_time_s))
        self.frame_wall_times.append(wall_time_ns or time.time_ns())

    @staticmethod
    def _rate(times: list[float]) -> float | None:
        if len(times) < 2 or times[-1] <= times[0]:
            return None
        return (len(times) - 1) / (times[-1] - times[0])

    @staticmethod
    def _wall_rate(times: list[int]) -> float | None:
        if len(times) < 2 or times[-1] <= times[0]:
            return None
        return (len(times) - 1) / ((times[-1] - times[0]) / 1e9)

    def summary(self) -> dict[str, Any]:
        sim_gaps = [b - a for a, b in zip(self.physics_sim_times, self.physics_sim_times[1:])]
        frame_gaps = [b - a for a, b in zip(self.frame_sim_times, self.frame_sim_times[1:])]
        unique = len(self.frame_ids) == len(set(self.frame_ids))
        physics_wall_duration_s = (
            (self.physics_wall_times[-1] - self.physics_wall_times[0]) / 1e9
            if len(self.physics_wall_times) >= 2
            else None
        )
        physics_sim_duration_s = (
            self.physics_sim_times[-1] - self.physics_sim_times[0]
            if len(self.physics_sim_times) >= 2
            else None
        )
        return {
            "name": self.name,
            "physics_samples": len(self.physics_sim_times),
            "physics_sim_hz": self._rate(self.physics_sim_times),
            "physics_wall_hz": self._wall_rate(self.physics_wall_times),
            "physics_wall_duration_s": physics_wall_duration_s,
            "physics_sim_duration_s": physics_sim_duration_s,
            "physics_rtf": (
                physics_sim_duration_s / physics_wall_duration_s
                if physics_sim_duration_s is not None and physics_wall_duration_s
                else None
            ),
            "physics_gap_p99_s": _percentile(sim_gaps, 0.99),
            "physics_gap_max_s": max(sim_gaps) if sim_gaps else None,
            "frames": len(self.frame_ids),
            "frame_sim_hz": self._rate(self.frame_sim_times),
            "frame_wall_hz": self._wall_rate(self.frame_wall_times),
            "frame_gap_p99_s": _percentile(frame_gaps, 0.99),
            "frame_gap_max_s": max(frame_gaps) if frame_gaps else None,
            "unique_frame_ids": unique,
        }
