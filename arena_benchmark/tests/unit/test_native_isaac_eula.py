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

_PROBE = (
    "import json, os; print(json.dumps({key: os.environ.get(key, '') "
    "for key in ('OMNI_KIT_ACCEPT_EULA', 'ACCEPT_EULA')}))\n"
)


class NativeIsaacEulaEnvironmentTest(unittest.TestCase):
    """Verify the launcher never accepts the Omniverse EULA on the user's behalf.

    The launcher is executed with a trivial probe script (never Kit), so this
    checks only the environment contract of the wrapper.
    """

    def _run_launcher(self, overrides: dict[str, str | None]) -> dict[str, str]:
        with tempfile.TemporaryDirectory(prefix="arena-native-eula-") as temp_dir:
            probe = Path(temp_dir) / "probe.py"
            probe.write_text(_PROBE, encoding="utf-8")
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
        runtime = self._run_launcher({})
        self.assertEqual(runtime["OMNI_KIT_ACCEPT_EULA"], "")
        self.assertEqual(runtime["ACCEPT_EULA"], "")

    def test_launcher_preserves_explicit_eula_acceptance(self) -> None:
        runtime = self._run_launcher({"OMNI_KIT_ACCEPT_EULA": "YES", "ACCEPT_EULA": "Y"})
        self.assertEqual(runtime["OMNI_KIT_ACCEPT_EULA"], "YES")
        self.assertEqual(runtime["ACCEPT_EULA"], "Y")

    def test_launcher_does_not_override_explicit_refusal(self) -> None:
        runtime = self._run_launcher({"OMNI_KIT_ACCEPT_EULA": "NO"})
        self.assertEqual(runtime["OMNI_KIT_ACCEPT_EULA"], "NO")
        self.assertEqual(runtime["ACCEPT_EULA"], "")


if __name__ == "__main__":
    unittest.main()
