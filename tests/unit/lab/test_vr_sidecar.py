from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import os
import subprocess

from openflex_isaac_contract.session_config import FrequencyConfig, SessionConfig
from mrs_robot_lab.teleoperation.vr_sidecar import (
    build_vr_sidecar_script,
    load_vr_sidecar_config,
)


class VrSidecarConfigTest(unittest.TestCase):
    def _config(self, root: Path, task_id: str = "dual_arm_box_transport") -> SessionConfig:
        ros_setup = root / "ros setup.bash"
        underlay = root / "openflex setup.bash"
        urdf = root / "robot control.urdf"
        bridge = root / "isaaclab_ext/scripts/worker_ros2_bridge.py"
        synthetic_input = (
            root
            / "openflex_ws/src/openflex_integrated/openflex_gui/openflex_gui/simulated_vr_input.py"
        )
        for path in (ros_setup, underlay, urdf, bridge, synthetic_input):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# test fixture\n", encoding="utf-8")
        return SessionConfig(
            session_id="vr-sidecar",
            embodiment_id="openflex",
            scene_id=("dual_arm_box_tabletop" if task_id == "dual_arm_box_transport" else "flat_navigation"),
            task_id=task_id,
            frequency=FrequencyConfig(120.0, 30.0, 30.0),
            teleop_mode="vr",
            simulation_repo_root=str(root),
            metadata={
                "ros2_runtime": {
                    "ros_setup": str(ros_setup),
                    "underlay_setup": str(underlay),
                    "workspace_root": str(root / "openflex_ws"),
                    "domain_id": 49,
                    "isaac_urdf_path": str(urdf),
                }
            },
        )

    def test_arm_task_launch_reuses_pico_ik_but_disables_ros_sim_controllers(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            spec = load_vr_sidecar_config(config)

            script = build_vr_sidecar_script(config, spec, "http://127.0.0.1:24105")

        self.assertIn("enable_arms:=true", script)
        self.assertIn("enable_chassis:=false", script)
        self.assertIn("enable_head:=false", script)
        self.assertIn("enable_lift:=false", script)
        self.assertIn("stop_existing_vr:=false", script)
        self.assertIn("robot control.urdf", script)
        self.assertIn("worker_ros2_bridge.py", script)

    def test_navigation_launch_uses_vr_chassis_and_rejects_missing_ros_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, "navigation_to_goal")
            script = build_vr_sidecar_script(
                config,
                load_vr_sidecar_config(config),
                "http://127.0.0.1:24105",
            )
            self.assertIn("enable_arms:=false", script)
            self.assertIn("enable_chassis:=true", script)

            config.metadata["ros2_runtime"]["underlay_setup"] = str(root / "missing.bash")
            with self.assertRaisesRegex(FileNotFoundError, "underlay setup"):
                load_vr_sidecar_config(config)

    def test_simulated_vr_is_started_and_stopped_as_a_worker_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory), "navigation_to_goal")
            config.metadata["simulated_vr"] = True
            sidecar = load_vr_sidecar_config(config)

            script = build_vr_sidecar_script(config, sidecar, "http://127.0.0.1:24105")

        self.assertIn("simulated_vr_input.py", script)
        self.assertIn("SYNTHETIC_INPUT_PID", script)
        self.assertIn("kill -INT", script)

    def test_existing_pico_stack_is_reused_without_launching_a_duplicate_udp_listener(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root, "navigation_to_goal")
            sidecar = load_vr_sidecar_config(config)
            script = build_vr_sidecar_script(config, sidecar, "http://127.0.0.1:24105")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log_path = root / "calls.log"
            ros2 = fake_bin / "ros2"
            ros2.write_text(
                "#!/bin/bash\n"
                f"printf '%s\\n' \"$*\" >> {log_path}\n"
                "if [[ $1 == node && $2 == list ]]; then "
                "printf '%s\\n' /pico_pose_bridge /vr_teleop_chassis; fi\n",
                encoding="utf-8",
            )
            python = fake_bin / "python3"
            python.write_text(
                "#!/bin/bash\n"
                f"printf 'python %s\\n' \"$*\" >> {log_path}\n",
                encoding="utf-8",
            )
            ros2.chmod(0o755)
            python.chmod(0o755)
            config.metadata["ros2_runtime"]["ros_setup"] = str(
                root / "ros setup.bash"
            )
            Path(config.metadata["ros2_runtime"]["ros_setup"]).write_text(
                f"export PATH={fake_bin}:$PATH\n", encoding="utf-8"
            )
            completed = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = log_path.read_text(encoding="utf-8").splitlines()
            self.assertFalse(any(line.startswith("launch ") for line in calls))
            self.assertTrue(any(line.startswith("python ") and "worker_ros2_bridge.py" in line for line in calls))


if __name__ == "__main__":
    unittest.main()
