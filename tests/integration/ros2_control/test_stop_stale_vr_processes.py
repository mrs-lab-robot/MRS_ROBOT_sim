#!/usr/bin/env python3
"""Tests for the VR launch stale-process cleanup helper."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_DIR / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_bringup"
SCRIPT_PATH = PACKAGE_DIR / "scripts" / "stop_stale_vr_processes.py"
SPEC = importlib.util.spec_from_file_location("stop_stale_vr_processes", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StopStaleVrProcessesTest(unittest.TestCase):
    @staticmethod
    def _write_process(proc_root: Path, pid: int, argv) -> None:
        process_dir = proc_root / str(pid)
        process_dir.mkdir()
        (process_dir / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")

    def test_finds_only_vr_stack_processes_and_excludes_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_process(proc_root, 101, ["/opt/ws/openarmx_teleop_vr_node", "--ros-args"])
            self._write_process(proc_root, 102, ["python3", "/opt/ws/unrelated_node"])
            self._write_process(
                proc_root,
                103,
                ["ros2", "launch", "openflex_isaac_bringup", "vr_teleop.launch.py"],
            )
            self._write_process(proc_root, 104, ["/opt/ws/pico_pose_bridge_node"])

            matches = MODULE.find_stale_vr_processes(
                proc_root=proc_root,
                exclude_pids={104},
            )

            self.assertEqual(matches, [101, 103])

    def test_matches_python_wrapped_vr_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            proc_root = Path(tmp)
            self._write_process(
                proc_root,
                201,
                ["python3", "/opt/ws/vr_lift_control_node", "--ros-args"],
            )

            self.assertEqual(
                MODULE.find_stale_vr_processes(proc_root=proc_root, exclude_pids=set()),
                [201],
            )


if __name__ == "__main__":
    unittest.main()
