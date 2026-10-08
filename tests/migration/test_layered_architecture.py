"""Static contracts for the sim_runtime -> isaaclab_ext -> arena layering."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class LayeredArchitectureTests(unittest.TestCase):
    def test_canonical_runtime_assets_and_ros_packages_live_under_sim_runtime(self):
        self.assertTrue(
            (ROOT / "sim_runtime/assets/robots/openflex_robot.usda").is_file()
        )
        self.assertTrue(
            (
                ROOT
                / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
            ).is_file()
        )
        self.assertTrue(
            (
                ROOT
                / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/launch/sim.launch.py"
            ).is_file()
        )

    def test_isaaclab_interface_has_its_own_resolver_robot_cfg_and_smoke_entry(self):
        lab_root = ROOT / "isaaclab_ext/src/mrs_robot_lab"
        self.assertTrue((lab_root / "assets/asset_resolver.py").is_file())
        self.assertTrue((lab_root / "assets/robot_interface.py").is_file())
        self.assertTrue((lab_root / "assets/mrs_robot_cfg.py").is_file())
        self.assertTrue((lab_root / "actions/openflex_actions.py").is_file())
        self.assertTrue((lab_root / "observations/proprioception.py").is_file())
        self.assertTrue((lab_root / "environments/smoke/openflex_smoke.py").is_file())
        scripts = ("spawn_robot.py", "test_joint_control.py", "test_camera.py", "test_actions.py", "teleop_robot.py")
        for script in scripts:
            with self.subTest(script=script):
                self.assertTrue((ROOT / "isaaclab_ext/scripts" / script).is_file())

    def test_lab_entrypoints_do_not_depend_on_the_benchmark_layer(self):
        script_root = ROOT / "isaaclab_ext/scripts"
        for source in script_root.glob("*.py"):
            text = source.read_text(encoding="utf-8")
            with self.subTest(source=source):
                self.assertNotIn("mrs_arena", text)
                self.assertNotIn("isaaclab_arena", text)

    def test_teleoperation_adapter_reuses_the_canonical_lab_interface(self):
        adapter = ROOT / "isaaclab_ext/src/mrs_robot_lab/adapters/teleop_adapter.py"
        source = adapter.read_text(encoding="utf-8")
        self.assertIn("mrs_robot_lab.assets.robot_interface", source)
        self.assertNotIn("JOINT_STATE_NAMES = (", source)

    def test_arena_embodiment_reuses_the_lab_robot_cfg_instead_of_redeclaring_it(self):
        lab_cfg = ROOT / "isaaclab_ext/src/mrs_robot_lab/assets/mrs_robot_cfg.py"
        arena_embodiment = ROOT / "arena_benchmark/src/mrs_arena/embodiments/openflex.py"
        self.assertIn("MRS_ROBOT_CFG", lab_cfg.read_text(encoding="utf-8"))
        arena_source = arena_embodiment.read_text(encoding="utf-8")
        self.assertIn("from mrs_robot_lab.assets.mrs_robot_cfg import", arena_source)
        self.assertIn("MRS_ROBOT_CFG", arena_source)
        self.assertNotIn("ArticulationCfg(", arena_source)

    def test_lab_smoke_is_constructible_without_an_arena_environment(self):
        smoke = ROOT / "isaaclab_ext/src/mrs_robot_lab/environments/smoke/openflex_smoke.py"
        source = smoke.read_text(encoding="utf-8")
        self.assertIn("def make_openflex_smoke_environment", source)
        self.assertNotIn("isaaclab_arena", source)
        self.assertNotIn("mrs_arena", source)

    def test_arena_benchmark_has_a_separate_python_package(self):
        self.assertTrue((ROOT / "arena_benchmark/pyproject.toml").is_file())
        self.assertTrue((ROOT / "arena_benchmark/src/mrs_arena/embodiments/openflex.py").is_file())

    def test_arena_smoke_covers_registered_task_success_and_metric_recording(self):
        task = ROOT / "arena_benchmark/src/mrs_arena/tasks/openflex_smoke_task.py"
        smoke = ROOT / "arena_benchmark/scripts/smoke_openflex_task.py"
        self.assertTrue(task.is_file())
        self.assertTrue(smoke.is_file())
        task_source = task.read_text(encoding="utf-8")
        smoke_source = smoke.read_text(encoding="utf-8")
        self.assertIn("@register_task", task_source)
        self.assertIn("SuccessRateMetric", task_source)
        self.assertIn("TerminationTermCfg", task_source)
        self.assertIn("make_openflex_task_smoke_environment", smoke_source)
        self.assertIn("AssetRegistry", smoke_source)
        self.assertIn("get_asset_by_name", smoke_source)
        self.assertIn("termination_manager", smoke_source)
        self.assertIn("success_rate", smoke_source)

    def test_global_version_lock_and_upstream_arena_are_at_repository_root(self):
        self.assertTrue((ROOT / "configs/versions.yaml").is_file())
        self.assertTrue((ROOT / "third_party/IsaacLab-Arena/AGENTS.md").is_file())

    def test_external_gui_legacy_roots_resolve_to_the_canonical_projects(self):
        self.assertEqual((ROOT / "arena").resolve(), (ROOT / "arena_benchmark").resolve())
        self.assertEqual((ROOT / "isaac_sim_core").resolve(), (ROOT / "sim_runtime").resolve())
        self.assertEqual(
            (ROOT / "ros2_pkgs/openflex_isaac_sim").resolve(),
            (ROOT / "sim_runtime/ros2/openflex_isaac_sim").resolve(),
        )

    def test_legacy_runtime_paths_resolve_to_the_single_canonical_files(self):
        legacy_usd = ROOT / "isaac_sim_core/assets/robots/openflex_robot.usda"
        canonical_usd = ROOT / "sim_runtime/assets/robots/openflex_robot.usda"
        legacy_contract = (
            ROOT
            / "ros2_pkgs/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        )
        canonical_contract = (
            ROOT
            / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        )
        self.assertTrue(legacy_usd.exists())
        self.assertEqual(legacy_usd.resolve(), canonical_usd.resolve())
        self.assertTrue(legacy_contract.exists())
        self.assertEqual(legacy_contract.resolve(), canonical_contract.resolve())

    def test_ros_sensor_package_uses_canonical_runtime_realsense_configs(self):
        package_config = ROOT / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_sensors/config"
        runtime_config = ROOT / "sim_runtime/config/sensors/realsense"
        for name in (
            "realsense_quad.yaml",
            "realsense_robot_mounts.yaml",
            "realsense_standalone.yaml",
            "calibration/d405_nominal_640x480_rectified_v1.yaml",
            "calibration/d435_nominal_640x480_rectified_v1.yaml",
        ):
            with self.subTest(name=name):
                self.assertEqual((package_config / name).resolve(), (runtime_config / name).resolve())

    def test_lower_layers_do_not_import_higher_layers(self):
        forbidden_by_layer = {
            ROOT / "sim_runtime": ("mrs_arena", "isaaclab_arena"),
            ROOT / "isaaclab_ext/src": ("mrs_arena", "isaaclab_arena"),
        }
        for layer_root, forbidden_imports in forbidden_by_layer.items():
            for source in layer_root.rglob("*.py"):
                text = source.read_text(encoding="utf-8")
                for forbidden_import in forbidden_imports:
                    with self.subTest(source=source, forbidden=forbidden_import):
                        self.assertNotIn(forbidden_import, text)

    def test_robot_usd_is_not_duplicated_in_learning_or_benchmark_layers(self):
        canonical_robot_assets = tuple(
            path
            for path in (ROOT / "sim_runtime/assets/robots").rglob("*")
            if path.is_file() and path.suffix.lower() in {".usd", ".usda"}
        )
        self.assertEqual([path.name for path in canonical_robot_assets], ["openflex_robot.usda"])
        generated_runtime_dirs = {".cache", ".venv"}
        for layer_name in ("isaaclab_ext", "arena_benchmark"):
            layer_root = ROOT / layer_name
            copied_robot_assets = [
                path
                for path in layer_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".usd", ".usda"}
                and "third_party" not in path.parts
                and not generated_runtime_dirs.intersection(path.relative_to(layer_root).parts)
            ]
            self.assertEqual(copied_robot_assets, [], msg=f"duplicate USD under {layer_name}")


if __name__ == "__main__":
    unittest.main()
