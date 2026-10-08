"""Apply the shared session frequencies to DirectRLEnv configuration."""

from __future__ import annotations

from typing import Any

from openflex_isaac_contract.session_config import FrequencyConfig


def apply_runtime_rates(
    cfg: Any,
    *,
    physics_hz: float,
    control_hz: float,
    render_hz: float,
) -> None:
    """Set physics dt, action decimation and render interval from one contract."""
    rates = FrequencyConfig(
        physics_hz=float(physics_hz),
        control_hz=float(control_hz),
        render_hz=float(render_hz),
    )
    rates.validate()
    cfg.sim.dt = 1.0 / rates.physics_hz
    cfg.decimation = int(round(rates.physics_hz / rates.control_hz))
    cfg.sim.render_interval = int(round(rates.physics_hz / rates.render_hz))
