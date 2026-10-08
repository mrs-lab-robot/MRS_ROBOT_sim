from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


LAUNCH_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/launch/sim.launch.py"
)


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load_sim_launch_module():
    placeholder = object
    modules = {
        "ament_index_python": _module("ament_index_python", __path__=[]),
        "ament_index_python.packages": _module(
            "ament_index_python.packages",
            get_package_prefix=placeholder,
            get_package_share_directory=placeholder,
        ),
        "launch": _module("launch", __path__=[], LaunchDescription=placeholder),
        "launch.actions": _module(
            "launch.actions",
            DeclareLaunchArgument=placeholder,
            ExecuteProcess=placeholder,
            OpaqueFunction=placeholder,
            RegisterEventHandler=placeholder,
            SetEnvironmentVariable=placeholder,
            Shutdown=placeholder,
            TimerAction=placeholder,
            UnsetEnvironmentVariable=placeholder,
        ),
        "launch.event_handlers": _module("launch.event_handlers", OnProcessExit=placeholder),
        "launch.substitutions": _module("launch.substitutions", LaunchConfiguration=placeholder),
        "launch_ros": _module("launch_ros", __path__=[]),
        "launch_ros.actions": _module("launch_ros.actions", Node=placeholder),
        "launch_ros.parameter_descriptions": _module(
            "launch_ros.parameter_descriptions",
            ParameterFile=placeholder,
            ParameterValue=placeholder,
        ),
    }
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("mrs_test_sim_launch", LAUNCH_SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class IsaacSimPathTest(unittest.TestCase):
    def test_explicit_install_path_is_used_before_home_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            install = Path(temp_dir) / "custom-isaac-sim"
            install.mkdir()
            (install / "python.sh").touch()

            with patch.dict(os.environ, {"ISAACSIM_PATH": str(install)}), patch.object(
                Path, "home", return_value=Path(temp_dir) / "unconfigured-home"
            ):
                resolved = _load_sim_launch_module()._default_isaac_path()

        self.assertEqual(resolved, install)

    def test_install_root_with_runtime_child_resolves_to_python_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            install_root = Path(temp_dir) / "isaac-sim-package"
            runtime = install_root / "runtime"
            runtime.mkdir(parents=True)
            (runtime / "python.sh").touch()

            with patch.dict(os.environ, {"ISAACSIM_PATH": str(install_root)}), patch.object(
                Path, "home", return_value=Path(temp_dir) / "unconfigured-home"
            ):
                resolved = _load_sim_launch_module()._default_isaac_path()

        self.assertEqual(resolved, runtime)


if __name__ == "__main__":
    unittest.main()
