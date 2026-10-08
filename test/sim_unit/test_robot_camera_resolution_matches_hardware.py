from pathlib import Path
import unittest

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
HARDWARE_CAMERA_CONFIG = (
    REPOSITORY_ROOT
    / "openflex_ws/src/openflex_vla/config/cameras/cameras_config.yaml"
)
SIMULATION_CAMERA_CONFIG = (
    REPOSITORY_ROOT
    / "MRS_ROBOT_sim/sim_runtime/config/sensors/realsense/realsense_robot_mounts.yaml"
)


class RobotCameraResolutionMatchesHardwareTest(unittest.TestCase):
    def test_simulation_topics_use_each_hardware_raw_stream_resolution(self):
        hardware = yaml.safe_load(HARDWARE_CAMERA_CONFIG.read_text(encoding="utf-8"))
        simulation = yaml.safe_load(SIMULATION_CAMERA_CONFIG.read_text(encoding="utf-8"))

        simulation_by_topic = {
            camera["node_namespace"]: camera
            for camera in simulation["cameras"].values()
            if camera.get("node_namespace")
        }
        for hardware_camera in hardware["cameras"].values():
            topic_namespace = hardware_camera["launch_name"]
            with self.subTest(topic=topic_namespace):
                simulated_camera = simulation_by_topic[topic_namespace]
                raw_stream_size = (
                    hardware_camera.get("launch_width", hardware_camera["width"]),
                    hardware_camera.get("launch_height", hardware_camera["height"]),
                )
                self.assertEqual(
                    (simulated_camera["width"], simulated_camera["height"]),
                    raw_stream_size,
                    f"{topic_namespace} must preserve the true robot's raw ROS image size",
                )


if __name__ == "__main__":
    unittest.main()
