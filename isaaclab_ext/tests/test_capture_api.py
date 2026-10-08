from __future__ import annotations

import json
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from mrs_robot_lab.recorders.capture_api import (
    COMMAND_PATHS,
    SESSION_STATUS_PATH,
    CaptureControlServer,
)


class CaptureControlServerTest(unittest.TestCase):
    def test_teleop_command_mailbox_keeps_latest_and_caches_joint_state(self) -> None:
        with patch("mrs_robot_lab.recorders.capture_api.ThreadingHTTPServer") as http_server:
            http_server.return_value.server_address = ("127.0.0.1", 24105)
            server = CaptureControlServer()
        try:
            server.update_teleop_state(
                ready=True,
                joint_state={"names": ["joint_a"], "position": [0.25], "velocity": [0.0]},
            )
            server.submit_teleop_command({"kind": "base_twist", "values": [0.1, 0.0, 0.0]})
            server.submit_teleop_command({"kind": "base_twist", "values": [0.2, 0.0, 0.0]})

            self.assertTrue(server.teleop_state()["ready"])
            self.assertEqual(server.teleop_state()["joint_state"]["position"], [0.25])
            self.assertEqual(
                server.take_latest_teleop_command()["values"], [0.2, 0.0, 0.0]
            )
            self.assertIsNone(server.take_latest_teleop_command())
        finally:
            server.close()

    def test_teleop_mailbox_rejects_malformed_or_non_finite_commands(self) -> None:
        with patch("mrs_robot_lab.recorders.capture_api.ThreadingHTTPServer") as http_server:
            http_server.return_value.server_address = ("127.0.0.1", 24105)
            server = CaptureControlServer()
        try:
            server.update_teleop_state(ready=True)
            with self.assertRaisesRegex(ValueError, "three finite"):
                server.submit_teleop_command({"kind": "base_twist", "values": [0.2, 0.1]})
            with self.assertRaisesRegex(ValueError, "finite"):
                server.submit_teleop_command(
                    {"kind": "dual_arm", "left": [float("nan")] + [0.0] * 7, "right": None}
                )
            with self.assertRaisesRegex(ValueError, "active"):
                server.submit_teleop_command({"kind": "estop", "active": "false"})
            self.assertIsNone(server.take_latest_teleop_command())
        finally:
            server.close()

    def test_public_api_uses_session_and_episode_paths(self) -> None:
        self.assertEqual(COMMAND_PATHS["/api/v1/episode/start"], "start")
        self.assertEqual(COMMAND_PATHS["/api/v1/episode/save"], "save")
        self.assertEqual(COMMAND_PATHS["/api/v1/episode/export"], "export")
        self.assertEqual(COMMAND_PATHS["/api/v1/episode/discard"], "discard")
        self.assertEqual(COMMAND_PATHS["/api/v1/environment/reset"], "reset")
        self.assertEqual(COMMAND_PATHS["/api/v1/session/stop"], "stop")
        self.assertEqual(SESSION_STATUS_PATH, "/api/v1/session/status")
        self.assertEqual(COMMAND_PATHS["/api/v1/recording/start"], "start")

    def test_loopback_api_reports_session_status_and_queues_episode_commands(self) -> None:
        try:
            server = CaptureControlServer(port=0)
        except PermissionError as error:
            self.skipTest(f"sandbox does not permit loopback sockets: {error}")
        server.start()
        try:
            server.update_status(state="env_ready", ready=True, recording=False, steps=0)
            with urlopen(f"{server.base_url}{SESSION_STATUS_PATH}", timeout=1) as response:
                status = json.loads(response.read().decode("utf-8"))
            self.assertTrue(status["ready"])
            self.assertFalse(status["recording"])
            self.assertEqual(status["state"], "env_ready")

            request = Request(
                f"{server.base_url}/api/v1/episode/start",
                data=b'{"metadata":{"input_source":"keyboard"}}',
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=1) as response:
                self.assertEqual(response.status, 202)
            queued = server.take_requests()
            self.assertEqual(len(queued), 1)
            self.assertEqual(queued[0].command, "start")
            self.assertEqual(queued[0].payload["metadata"]["input_source"], "keyboard")

            invalid = Request(
                f"{server.base_url}/api/v1/episode/reset",
                data=b"{}",
                method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(invalid, timeout=1)
            self.assertEqual(error.exception.code, 404)
            self.assertEqual(server.take_commands(), [])
        finally:
            server.close()

    def test_loopback_api_accepts_vr_commands_and_exposes_worker_joint_state(self) -> None:
        try:
            server = CaptureControlServer(port=0)
        except PermissionError as error:
            self.skipTest(f"sandbox does not permit loopback sockets: {error}")
        server.start()
        try:
            server.update_teleop_state(
                ready=True,
                joint_state={"names": ["joint_a"], "position": [0.25], "velocity": [0.0]},
            )
            with urlopen(f"{server.base_url}/api/v1/teleop/state", timeout=1) as response:
                state = json.loads(response.read().decode("utf-8"))
            self.assertTrue(state["ready"])
            self.assertEqual(state["joint_state"]["position"], [0.25])

            for values in ((0.1, 0.0, 0.0), (0.2, 0.0, 0.0)):
                request = Request(
                    f"{server.base_url}/api/v1/teleop/command",
                    data=json.dumps({"kind": "base_twist", "values": values}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=1) as response:
                    self.assertEqual(response.status, 202)

            latest = server.take_latest_teleop_command()
            self.assertEqual(latest["values"], [0.2, 0.0, 0.0])
            self.assertIsNone(server.take_latest_teleop_command())
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()
