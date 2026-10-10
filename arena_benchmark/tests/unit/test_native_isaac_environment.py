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


class NativeIsaacEnvironmentTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
