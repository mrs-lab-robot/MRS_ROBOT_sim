#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "ros2_pkgs/openflex_isaac_sim/openflex_isaac_bringup/scripts/start_robot_control_sim.py"
)
SPEC = importlib.util.spec_from_file_location("start_robot_control_sim", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SimulationAppStartupTest(unittest.TestCase):
    def test_runtime_environment_adds_sensor_package_to_current_python_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sensor_source = (
                Path(temp_dir)
                / "ros2_pkgs/openflex_isaac_sim/openflex_isaac_sensors"
            )
            package = sensor_source / "openflex_isaac_sensors"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            (package / "runtime_control.py").touch()

            with patch.object(MODULE, "_isaac_install_root", return_value=None), \
                    patch.object(sys, "path", []), \
                    patch.dict(os.environ, {
                        "OPENFLEX_ISAAC_SENSOR_ASSET_DIR": temp_dir,
                        "AMENT_PREFIX_PATH": "",
                        "PYTHONPATH": "/legacy/python/path",
                    }):
                MODULE._clean_isaac_runtime_environment()

                self.assertEqual(sys.path[0], str(sensor_source))
                self.assertIsNotNone(
                    importlib.util.find_spec("openflex_isaac_sensors.runtime_control")
                )

    def test_uninitialized_kit_app_lookup_returns_none_for_isaac_6_startup(self) -> None:
        def get_app_before_startup():
            raise RuntimeError(
                "Failed to acquire interface: omni::kit::IApp (pluginName: nullptr)"
            )

        omni = types.ModuleType("omni")
        omni.__path__ = []
        kit = types.ModuleType("omni.kit")
        kit.__path__ = []
        app = types.ModuleType("omni.kit.app")
        app.get_app = get_app_before_startup
        omni.kit = kit
        kit.app = app

        with patch.dict(sys.modules, {
            "omni": omni,
            "omni.kit": kit,
            "omni.kit.app": app,
        }):
            MODULE._patch_kit_app_lookup_before_startup()

            self.assertIsNone(app.get_app())

    def test_headless_startup_hides_ui_and_keeps_sensor_rendering_enabled(self) -> None:
        calls = []

        def simulation_app(config, *args, **kwargs):
            calls.append((config, args, kwargs))
            return "app"

        fake_isaacsim = type("IsaacSim", (), {"SimulationApp": simulation_app})
        with patch.dict(sys.modules, {"isaacsim": fake_isaacsim}):
            MODULE._patch_simulation_app_startup()
            result = fake_isaacsim.SimulationApp({
                "renderer": "RayTracedLighting",
                "headless": True,
                "hide_ui": False,
                "open_usd": "/tmp/scene.usd",
                "extra_args": ["--custom-arg=true"],
            })

        self.assertEqual(result, "app")
        config, args, kwargs = calls[0]
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {})
        self.assertTrue(config["hide_ui"])
        self.assertTrue(config["disable_viewport_updates"])
        self.assertEqual(config["renderer"], "RayTracedLighting")
        self.assertEqual(config["open_usd"], "/tmp/scene.usd")
        self.assertIn("--custom-arg=true", config["extra_args"])
        self.assertIn("--/renderer/multiGpu/enabled=false", config["extra_args"])

    def test_interactive_startup_keeps_requested_ui_visibility(self) -> None:
        calls = []

        def simulation_app(config, *args, **kwargs):
            calls.append(config)
            return "app"

        fake_isaacsim = type("IsaacSim", (), {"SimulationApp": simulation_app})
        with patch.dict(sys.modules, {"isaacsim": fake_isaacsim}):
            MODULE._patch_simulation_app_startup()
            fake_isaacsim.SimulationApp({"headless": False, "hide_ui": False})

        self.assertFalse(calls[0]["hide_ui"])
        self.assertNotIn("disable_viewport_updates", calls[0])


if __name__ == "__main__":
    unittest.main()
