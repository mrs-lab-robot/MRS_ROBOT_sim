from __future__ import annotations

from types import SimpleNamespace

import pytest

from mrs_robot_lab.environments.learning.runtime_rates import apply_runtime_rates


def _cfg():
    return SimpleNamespace(
        sim=SimpleNamespace(dt=0.0, render_interval=0),
        decimation=0,
    )


def test_apply_runtime_rates_sets_physics_control_and_render_ticks():
    cfg = _cfg()

    apply_runtime_rates(
        cfg,
        physics_hz=120,
        control_hz=20,
        render_hz=30,
    )

    assert cfg.sim.dt == pytest.approx(1 / 120)
    assert cfg.decimation == 6
    assert cfg.sim.render_interval == 4


def test_apply_runtime_rates_rejects_fractional_environment_steps():
    cfg = _cfg()

    with pytest.raises(ValueError, match="physics_hz"):
        apply_runtime_rates(
            cfg,
            physics_hz=120,
            control_hz=25,
            render_hz=30,
        )
