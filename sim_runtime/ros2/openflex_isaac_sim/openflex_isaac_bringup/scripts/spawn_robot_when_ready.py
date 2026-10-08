#!/usr/bin/python3
"""Wait for Isaac REST, then request a direct USD reference and ROS setup."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import rclpy
from rclpy.node import Node


# The REST server's spawn endpoint may wait up to 180 seconds for a Kit-thread
# operation. Leave room for the HTTP server to return its own result/timeout.
SPAWN_REQUEST_TIMEOUT_SEC = 190.0


class SpawnRobotWhenReady(Node):
    def __init__(self, require_urdf_path: bool = True, node_name: str = "spawn_robot_when_ready") -> None:
        super().__init__(node_name)

        self.declare_parameter("urdf_path", "")
        self.declare_parameter("usd_path", "")
        self.declare_parameter("x", 0.0)
        self.declare_parameter("y", 0.0)
        self.declare_parameter("z", 0.0)
        self.declare_parameter("roll", 0.0)
        self.declare_parameter("pitch", 0.0)
        self.declare_parameter("yaw", 0.0)
        self.declare_parameter("fixed", False)
        self.declare_parameter("enable_sensors", True)
        self.declare_parameter("apply_appearance", True)
        self.declare_parameter("api_host", "127.0.0.1")
        self.declare_parameter("api_port", 8080)
        self.declare_parameter("wait_timeout", 240.0)
        self.declare_parameter("poll_period", 2.0)
        self.declare_parameter("auto_play", True)
        self.declare_parameter("play_timeout", 60.0)
        # Let sensor render products finish their first allocation before the
        # launch file releases the controller spawners.
        self.declare_parameter("post_play_settle_sec", 0.0)

        self.urdf_path = self.get_parameter("urdf_path").get_parameter_value().string_value
        self.usd_path = self.get_parameter("usd_path").get_parameter_value().string_value
        if require_urdf_path and not self.urdf_path:
            raise ValueError("urdf_path parameter is required")

        self.api_host = self.get_parameter("api_host").get_parameter_value().string_value
        self.api_port = self.get_parameter("api_port").get_parameter_value().integer_value
        self.wait_timeout = self.get_parameter("wait_timeout").get_parameter_value().double_value
        self.poll_period = self.get_parameter("poll_period").get_parameter_value().double_value
        self.auto_play = self.get_parameter("auto_play").get_parameter_value().bool_value
        self.play_timeout = self.get_parameter("play_timeout").get_parameter_value().double_value
        self.post_play_settle_sec = max(
            0.0,
            self.get_parameter("post_play_settle_sec").get_parameter_value().double_value,
        )
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def base_url(self) -> str:
        return f"http://{self.api_host}:{self.api_port}"

    def _request_json(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, object] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with self.opener.open(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}

    def wait_for_health(self) -> bool:
        deadline = time.monotonic() + self.wait_timeout
        last_error = "not checked yet"
        last_log_time = 0.0

        while rclpy.ok() and time.monotonic() < deadline:
            try:
                result = self._request_json("GET", "/health", timeout=5.0)
                if result.get("success") is True:
                    self.get_logger().info(f"Isaac REST API is ready at {self.base_url}")
                    return True
                last_error = str(result)
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
                last_error = str(exc)

            now = time.monotonic()
            if last_log_time == 0.0 or now - last_log_time >= 10.0:
                # Isaac Sim needs tens of seconds to initialize Kit before
                # binding the REST socket.  Connection refused during that
                # window is an expected startup race, not a control failure.
                if "Connection refused" in last_error:
                    self.get_logger().debug(
                        f"Isaac REST API is still starting at {self.base_url}"
                    )
                else:
                    self.get_logger().warn(
                        f"Waiting for Isaac REST API at {self.base_url} "
                        f"(last error: {last_error})"
                    )
                last_log_time = now
            time.sleep(max(0.2, self.poll_period))

        self.get_logger().error(
            f"Timed out after {self.wait_timeout:.1f}s waiting for Isaac REST API "
            f"at {self.base_url}; last error: {last_error}"
        )
        return False

    def wait_for_timeline_playing(self, prim_path: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        quoted_prim_path = urllib.parse.quote(prim_path, safe="")
        last_error = "not checked yet"

        while rclpy.ok() and time.monotonic() < deadline:
            try:
                result = self._request_json(
                    "GET",
                    f"/openflex/debug/pose?prim_path={quoted_prim_path}",
                    timeout=5.0,
                )
                data = result.get("data")
                if isinstance(data, dict) and data.get("timeline_playing") is True:
                    return True
                last_error = str(result)
            except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
                last_error = str(exc)
            time.sleep(max(0.2, self.poll_period))

        self.get_logger().warn(f"Timeline did not report playing within {timeout:.1f}s: {last_error}")
        return False

    def spawn_robot(self) -> bool:
        payload = {
            "urdf_path": self.urdf_path,
            "x": self.get_parameter("x").get_parameter_value().double_value,
            "y": self.get_parameter("y").get_parameter_value().double_value,
            "z": self.get_parameter("z").get_parameter_value().double_value,
            "roll": self.get_parameter("roll").get_parameter_value().double_value,
            "pitch": self.get_parameter("pitch").get_parameter_value().double_value,
            "yaw": self.get_parameter("yaw").get_parameter_value().double_value,
            "fixed": self.get_parameter("fixed").get_parameter_value().bool_value,
            "enable_sensors": self.get_parameter("enable_sensors").get_parameter_value().bool_value,
            "apply_appearance": self.get_parameter("apply_appearance").get_parameter_value().bool_value,
        }

        self.get_logger().info(
            f"Loading robot USD {self.usd_path}; using URDF only for ROS 2 control metadata"
        )
        try:
            result = self._request_json(
                "POST",
                "/spawn_robot",
                payload=payload,
                timeout=SPAWN_REQUEST_TIMEOUT_SEC,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.get_logger().error(f"Isaac rejected spawn request: HTTP {exc.code}: {body}")
            return False
        except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Spawn request failed: {exc}")
            return False

        if result.get("success") is True:
            self.get_logger().info(f"Robot spawned successfully: {result.get('message')}")
            data = result.get("data")
            prim_path = "/openflex"
            if isinstance(data, dict) and data.get("prim_path"):
                prim_path = str(data["prim_path"])
                self.get_logger().info(f"Prim path: {data['prim_path']}")
            if self.auto_play:
                try:
                    play_result = self._request_json(
                        "POST", "/simulation/play", timeout=self.play_timeout
                    )
                except (urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError) as exc:
                    self.get_logger().warn(
                        "Simulation play request did not return cleanly; "
                        f"checking timeline state before failing: {exc}"
                    )
                    if not self.wait_for_timeline_playing(
                        prim_path, timeout=max(10.0, self.play_timeout)
                    ):
                        self.get_logger().error(
                            f"Robot spawned, but simulation play request failed: {exc}"
                        )
                        return False
                else:
                    if play_result.get("success") is not True:
                        self.get_logger().error(
                            f"Robot spawned, but simulation play failed: {play_result}"
                        )
                        return False
                self.get_logger().info("Isaac simulation timeline is playing")
                if self.post_play_settle_sec > 0.0:
                    self.get_logger().info(
                        "Waiting {:.1f} s for sensor render products to settle before "
                        "starting ROS 2 controllers".format(self.post_play_settle_sec)
                    )
                    time.sleep(self.post_play_settle_sec)
            return True

        self.get_logger().error(f"Failed to spawn robot: {result}")
        return False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: SpawnRobotWhenReady | None = None
    exit_code = 1
    try:
        node = SpawnRobotWhenReady()
        if node.wait_for_health() and node.spawn_robot():
            exit_code = 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
