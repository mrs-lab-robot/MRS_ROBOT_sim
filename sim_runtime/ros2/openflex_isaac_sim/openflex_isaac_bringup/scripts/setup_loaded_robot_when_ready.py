#!/usr/bin/python3
"""Wait for Isaac REST API and attach ROS2 graphs to a loaded OpenFleX USD."""

from __future__ import annotations

from spawn_robot_when_ready import SpawnRobotWhenReady


class SetupLoadedRobotWhenReady(SpawnRobotWhenReady):
    def __init__(self) -> None:
        super().__init__(require_urdf_path=False, node_name="setup_loaded_robot_when_ready")
        if not self.has_parameter("robot_prim_path"):
            self.declare_parameter("robot_prim_path", "auto")
        if not self.has_parameter("articulation_prim_path"):
            self.declare_parameter("articulation_prim_path", "auto")
        if not self.has_parameter("command_topic"):
            self.declare_parameter("command_topic", "/openflex/joint_command")
        if not self.has_parameter("state_topic"):
            self.declare_parameter("state_topic", "/openflex/joint_states")
        if not self.has_parameter("enable_sensors"):
            self.declare_parameter("enable_sensors", True)
        if not self.has_parameter("camera_iso"):
            self.declare_parameter("camera_iso", 50.0)
        if not self.has_parameter("camera_width"):
            self.declare_parameter("camera_width", 640)
        if not self.has_parameter("camera_height"):
            self.declare_parameter("camera_height", 480)
        if not self.has_parameter("sensor_profile"):
            self.declare_parameter("sensor_profile", "data")
        if not self.has_parameter("lidar_profile"):
            self.declare_parameter("lidar_profile", "")

    def setup_loaded_robot(self) -> bool:
        payload = {
            "robot_prim_path": self.get_parameter("robot_prim_path").get_parameter_value().string_value,
            "articulation_prim_path": self.get_parameter("articulation_prim_path").get_parameter_value().string_value,
            "command_topic": self.get_parameter("command_topic").get_parameter_value().string_value,
            "state_topic": self.get_parameter("state_topic").get_parameter_value().string_value,
            "enable_sensors": self.get_parameter("enable_sensors").get_parameter_value().bool_value,
            "sensor_profile": self.get_parameter("sensor_profile").get_parameter_value().string_value,
            "lidar_profile": self.get_parameter("lidar_profile").get_parameter_value().string_value,
            "camera_iso": self.get_parameter("camera_iso").get_parameter_value().double_value,
            "camera_width": self.get_parameter("camera_width").get_parameter_value().integer_value,
            "camera_height": self.get_parameter("camera_height").get_parameter_value().integer_value,
        }

        self.get_logger().info("Attaching ROS2 graphs to loaded OpenFleX USD")
        try:
            result = self._request_json(
                "POST",
                "/openflex/setup_loaded_robot",
                payload=payload,
                timeout=90.0,
            )
        except Exception as exc:
            self.get_logger().error(f"Loaded USD setup request failed: {exc}")
            return False

        if result.get("success") is not True:
            self.get_logger().error(f"Failed to setup loaded OpenFleX USD: {result}")
            return False

        data = result.get("data")
        prim_path = "/World/OpenFlex"
        if isinstance(data, dict) and data.get("prim_path"):
            prim_path = str(data["prim_path"])
            self.get_logger().info(f"Loaded robot prim path: {prim_path}")
        if isinstance(data, dict) and data.get("articulation_prim_path"):
            self.get_logger().info(f"Articulation prim path: {data['articulation_prim_path']}")

        if self.auto_play:
            try:
                play_result = self._request_json("POST", "/simulation/play", timeout=self.play_timeout)
            except Exception as exc:
                self.get_logger().warn(
                    "Simulation play request did not return cleanly; "
                    f"checking timeline state before failing: {exc}"
                )
                if not self.wait_for_timeline_playing(prim_path, timeout=max(10.0, self.play_timeout)):
                    self.get_logger().error(f"Loaded USD setup succeeded, but play request failed: {exc}")
                    return False
            else:
                if play_result.get("success") is not True:
                    self.get_logger().error(f"Loaded USD setup succeeded, but play failed: {play_result}")
                    return False
            self.get_logger().info("Isaac simulation timeline is playing")
        return True


def main(args: list[str] | None = None) -> None:
    import rclpy

    rclpy.init(args=args)
    node: SetupLoadedRobotWhenReady | None = None
    exit_code = 1
    try:
        node = SetupLoadedRobotWhenReady()
        if node.wait_for_health() and node.setup_loaded_robot():
            exit_code = 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
