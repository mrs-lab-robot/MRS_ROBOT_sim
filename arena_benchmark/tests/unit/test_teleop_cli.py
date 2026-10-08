from __future__ import annotations

import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace


try:
    from mrs_arena.environments.openflex_graph_loader import (
        adapt_graph_spec_for_openflex,
        apply_reset_randomization_overrides,
        apply_simulation_rate_overrides,
    )
except ImportError:
    adapt_graph_spec_for_openflex = None
    apply_reset_randomization_overrides = None
    apply_simulation_rate_overrides = None


class TeleopCliTest(unittest.TestCase):
    def test_graph_adapter_replaces_the_droid_registry_without_changing_graph_ids(self) -> None:
        self.assertIsNotNone(adapt_graph_spec_for_openflex)
        graph = SimpleNamespace(
            embodiment=SimpleNamespace(
                id="droid",
                registry_name="droid_abs_joint_pos",
                params={},
            )
        )

        adapted = adapt_graph_spec_for_openflex(graph)

        self.assertIs(adapted, graph)
        self.assertEqual(graph.embodiment.id, "droid")
        self.assertEqual(graph.embodiment.registry_name, "openflex")

    def test_graph_adapter_rejects_droid_specific_embodiment_parameters(self) -> None:
        self.assertIsNotNone(adapt_graph_spec_for_openflex)
        graph = SimpleNamespace(
            embodiment=SimpleNamespace(
                id="droid",
                registry_name="droid_abs_joint_pos",
                params={"stand_height_m": 0.8},
            )
        )

        with self.assertRaisesRegex(ValueError, "Droid 专用参数"):
            adapt_graph_spec_for_openflex(graph)

    def test_simulation_rate_overrides_wrap_existing_yaml_callback(self) -> None:
        self.assertIsNotNone(apply_simulation_rate_overrides)
        environment = SimpleNamespace(env_cfg_callback=lambda cfg: self._mark_original(cfg))
        config = SimpleNamespace(sim=SimpleNamespace(dt=0.1, render_interval=1))

        adapted = apply_simulation_rate_overrides(environment, physics_hz=120, render_hz=30)
        result = adapted.env_cfg_callback(config)

        self.assertTrue(result.original_callback_applied)
        self.assertAlmostEqual(result.sim.dt, 1 / 120)
        self.assertEqual(result.sim.render_interval, 4)

    def test_reset_randomization_overrides_add_only_enabled_reset_terms(self) -> None:
        self.assertIsNotNone(apply_reset_randomization_overrides)

        class FakeEventTermCfg:
            def __init__(self, *, func, mode, params):
                self.func = func
                self.mode = mode
                self.params = params

        def make_configclass(name, fields):
            class Config:
                pass
            Config.__name__ = name
            for field_name, _field_type, default in fields:
                setattr(Config, field_name, default)
            return Config

        def combine_configclass_instances(_name, *configs):
            merged = SimpleNamespace()
            for config in configs:
                for key, value in vars(config).items():
                    setattr(merged, key, value)
                for key, value in vars(type(config)).items():
                    if not key.startswith("__"):
                        setattr(merged, key, value)
            return merged

        fake_modules = {
            "isaaclab": SimpleNamespace(),
            "isaaclab.managers": SimpleNamespace(EventTermCfg=FakeEventTermCfg),
            "isaaclab_arena": SimpleNamespace(),
            "isaaclab_arena.utils": SimpleNamespace(),
            "isaaclab_arena.utils.configclass": SimpleNamespace(
                combine_configclass_instances=combine_configclass_instances,
                make_configclass=make_configclass,
            ),
        }
        config = SimpleNamespace(events=SimpleNamespace(original="kept"))
        environment = SimpleNamespace(
            env_cfg_callback=lambda cfg: self._mark_original(cfg)
        )
        with patch.dict(sys.modules, fake_modules):
            apply_reset_randomization_overrides(environment, {
                "objects": {"banana": {"position_xy_m": [-0.02, 0.02]}},
                "light": {"prim_path": "/World/Light", "intensity": [1000, 1800]},
            })
            result = environment.env_cfg_callback(config)

        self.assertTrue(result.original_callback_applied)
        self.assertEqual(result.events.original, "kept")
        self.assertEqual(result.events.gui_reset_object_pose_0.mode, "reset")
        self.assertEqual(
            result.events.gui_reset_object_pose_0.params["asset_name"], "banana"
        )
        self.assertEqual(result.events.gui_reset_light_intensity.mode, "reset")

    @staticmethod
    def _mark_original(config):
        config.original_callback_applied = True
        return config

    def test_help_exits_cleanly_without_starting_kit(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts" / "teleop_vr_openflex.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--record-dir", result.stdout)
        self.assertIn("--reset-randomization-json", result.stdout)
        self.assertIn("--task-yaml", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
