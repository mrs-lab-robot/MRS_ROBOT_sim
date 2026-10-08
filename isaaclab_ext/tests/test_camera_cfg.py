from pathlib import Path
from types import SimpleNamespace
import unittest

from mrs_robot_lab.assets.asset_resolver import AssetResolver
from mrs_robot_lab.sensors.camera_cfg import add_capture_cameras, load_camera_mounts


ROOT = Path(__file__).resolve().parents[2]


class CameraInterfaceConfigTest(unittest.TestCase):
    def test_camera_interfaces_derive_paths_and_intrinsics_from_runtime_sensor_config(self):
        mounts = load_camera_mounts(AssetResolver(ROOT), robot_prim_path="/World/Robot")

        self.assertEqual(
            tuple(mount.name for mount in mounts),
            ("base_d435", "head_d435", "left_wrist_d405", "right_wrist_d405"),
        )
        self.assertTrue(all(mount.prim_path.startswith("/World/Robot/Geometry/") for mount in mounts))
        self.assertEqual(
            {mount.name: (mount.width, mount.height) for mount in mounts},
            {
                "base_d435": (424, 240),
                "head_d435": (640, 480),
                "left_wrist_d405": (480, 270),
                "right_wrist_d405": (480, 270),
            },
        )
        self.assertEqual(mounts[0].optical_frame, "d435_color_optical_frame")
        self.assertEqual(
            mounts[0].parent_prim_path,
            "/World/Robot/Geometry/base_link",
        )
        self.assertEqual(
            mounts[0].prim_path,
            "/World/Robot/Geometry/base_link/base_d435_LabCamera",
        )
        self.assertEqual(mounts[0].translation_m, (0.36, 0.0, 0.055))
        self.assertEqual(mounts[0].quaternion_wxyz, (0.5, 0.5, -0.5, -0.5))

    def test_capture_mounts_follow_selected_sensor_ids_and_independent_rates(self):
        mounts = load_camera_mounts(
            AssetResolver(ROOT),
            robot_prim_path="/World/Robot",
            sensor_frequencies_hz={
                "base_camera": 15.0,
                "right_wrist_camera": 10.0,
            },
        )

        self.assertEqual(
            tuple(mount.name for mount in mounts),
            ("base_d435", "right_wrist_d405"),
        )
        self.assertAlmostEqual(mounts[0].update_period_s, 1.0 / 15.0)
        self.assertAlmostEqual(mounts[1].update_period_s, 1.0 / 10.0)

    def test_capture_mounts_reject_unknown_sensor_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown camera sensor"):
            load_camera_mounts(
                AssetResolver(ROOT),
                sensor_frequencies_hz={"front_camera": 30.0},
            )

    def test_capture_cameras_attach_selected_configs_to_the_scene_sensor_registry(self):
        scene = SimpleNamespace(env_regex_ns="/World/envs/env_.*", sensors={})

        names = add_capture_cameras(
            scene,
            {"base_camera": 15.0},
            AssetResolver(ROOT),
            sensor_factory=lambda camera_cfg: camera_cfg,
        )

        self.assertEqual(names, ("base_d435",))
        self.assertEqual(set(scene.sensors), {"base_d435"})
        self.assertEqual(
            scene.sensors["base_d435"].prim_path,
            "/World/envs/env_.*/Robot/Geometry/base_link/base_d435_LabCamera",
        )


if __name__ == "__main__":
    unittest.main()
