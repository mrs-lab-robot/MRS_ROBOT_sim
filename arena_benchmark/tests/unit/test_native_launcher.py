from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


_ARENA_ROOT = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = _ARENA_ROOT.parent
_LAUNCHER = _ARENA_ROOT / "scripts" / "run_native_isaac.sh"
_KIT_ROS2_LIB = (
    _ARENA_ROOT
    / ".venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib"
)


class NativeLauncherTest(unittest.TestCase):
    def test_launcher_prefers_integrated_arena_package_over_installed_copy(self) -> None:
        arena_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            installed_copy = temporary_path / "installed"
            fake_package = installed_copy / "mrs_robot_arena"
            fake_package.mkdir(parents=True)
            (fake_package / "__init__.py").write_text("source = 'installed-copy'\n", encoding="utf-8")
            probe = temporary_path / "probe.py"
            probe.write_text(
                "import mrs_robot_arena\n"
                "print(mrs_robot_arena.__file__)\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["MRS_ARENA_PYTHON"] = sys.executable
            environment["MRS_ROBOT_SIM_ROOT"] = str(arena_root.parent)
            environment["PYTHONPATH"] = str(installed_copy)

            result = subprocess.run(
                ["bash", str(arena_root / "scripts" / "run_native_isaac.sh"), str(probe)],
                cwd=arena_root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            Path(result.stdout.strip()).resolve(),
            (arena_root / "src" / "mrs_robot_arena" / "__init__.py").resolve(),
        )

    def test_launcher_resolves_openflex_isaac_contract_from_sim_runtime(self) -> None:
        arena_root = Path(__file__).resolve().parents[2]
        repo_root = arena_root.parent
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            probe = temporary_path / "probe.py"
            probe.write_text(
                "import openflex_isaac_contract\n"
                "print(openflex_isaac_contract.__file__)\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["MRS_ARENA_PYTHON"] = sys.executable
            environment["MRS_ROBOT_SIM_ROOT"] = str(repo_root)
            # Remove any inherited PYTHONPATH to isolate launcher's configuration
            environment.pop("PYTHONPATH", None)

            result = subprocess.run(
                ["bash", str(arena_root / "scripts" / "run_native_isaac.sh"), str(probe)],
                cwd=arena_root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=20,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        resolved_contract_file = Path(result.stdout.strip()).resolve()
        expected_contract_base = (
            repo_root / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_contract"
        )
        self.assertTrue(
            resolved_contract_file.is_relative_to(expected_contract_base),
            f"Expected contract under {expected_contract_base}, got {resolved_contract_file}",
        )

    def test_launcher_exposes_bundled_humble_libraries_without_system_ros_paths(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arena-native-env-") as temp_dir:
            probe = Path(temp_dir) / "probe.py"
            probe.write_text(
                "import json, os; print(json.dumps({key: os.environ.get(key, '') "
                "for key in ('ROS_DISTRO', 'RMW_IMPLEMENTATION', 'LD_LIBRARY_PATH', 'PYTHONPATH')}))\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment.pop("ROS_DISTRO", None)
            environment.pop("RMW_IMPLEMENTATION", None)
            environment["LD_LIBRARY_PATH"] = "/opt/ros/humble/lib:/tmp/native-test-lib"
            environment["MRS_ARENA_PYTHON"] = sys.executable
            environment["MRS_ROBOT_SIM_ROOT"] = str(_REPOSITORY_ROOT)
            result = subprocess.run(
                ["bash", str(_LAUNCHER), str(probe)],
                cwd=_ARENA_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        runtime = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(runtime["ROS_DISTRO"], "humble")
        self.assertEqual(runtime["RMW_IMPLEMENTATION"], "rmw_fastrtps_cpp")
        paths = runtime["LD_LIBRARY_PATH"].split(":")
        self.assertNotIn("/opt/ros/humble/lib", paths)
        self.assertIn("/tmp/native-test-lib", paths)
        self.assertIn(str(_KIT_ROS2_LIB), paths)
        self.assertNotIn("/opt/ros/humble/lib/python3.10/site-packages", runtime["PYTHONPATH"])

    def _run_eula_launcher(self, overrides: dict[str, str | None]) -> dict[str, str]:
        probe_source = (
            "import json, os; print(json.dumps({key: os.environ.get(key, '') "
            "for key in ('OMNI_KIT_ACCEPT_EULA', 'ACCEPT_EULA')}))\n"
        )
        with tempfile.TemporaryDirectory(prefix="arena-native-eula-") as temp_dir:
            probe = Path(temp_dir) / "probe.py"
            probe.write_text(probe_source, encoding="utf-8")
            environment = os.environ.copy()
            environment.pop("OMNI_KIT_ACCEPT_EULA", None)
            environment.pop("ACCEPT_EULA", None)
            for key, value in overrides.items():
                if value is None:
                    environment.pop(key, None)
                else:
                    environment[key] = value
            environment["MRS_ARENA_PYTHON"] = sys.executable
            environment["MRS_ROBOT_SIM_ROOT"] = str(_REPOSITORY_ROOT)
            result = subprocess.run(
                ["bash", str(_LAUNCHER), str(probe)],
                cwd=_ARENA_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout.strip().splitlines()[-1])

    def test_launcher_does_not_accept_eula_without_explicit_opt_in(self) -> None:
        runtime = self._run_eula_launcher({})
        self.assertEqual(runtime["OMNI_KIT_ACCEPT_EULA"], "")
        self.assertEqual(runtime["ACCEPT_EULA"], "")

    def test_launcher_preserves_explicit_eula_acceptance(self) -> None:
        runtime = self._run_eula_launcher({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y"})
        self.assertEqual(runtime["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(runtime["ACCEPT_EULA"], "Y")

    def test_launcher_does_not_override_explicit_refusal(self) -> None:
        runtime = self._run_eula_launcher({"OMNI_KIT_ACCEPT_EULA": "NO"})
        self.assertEqual(runtime["OMNI_KIT_ACCEPT_EULA"], "NO")
        self.assertEqual(runtime["ACCEPT_EULA"], "")


if __name__ == "__main__":
    unittest.main()
