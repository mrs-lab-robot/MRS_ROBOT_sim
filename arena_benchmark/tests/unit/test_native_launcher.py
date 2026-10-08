from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
