"""Loopback-only control API for the unified Isaac Lab capture worker."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
from dataclasses import dataclass
from queue import Empty, Queue
import threading
import time


SESSION_STATUS_PATH = "/api/v1/session/status"
DEFAULT_CAPTURE_API_PORT = 24105
COMMAND_PATHS = {
    "/api/v1/session/stop": "stop",
    "/api/v1/episode/start": "start",
    "/api/v1/episode/save": "save",
    "/api/v1/episode/export": "export",
    "/api/v1/episode/discard": "discard",
    "/api/v1/environment/reset": "reset",
    # Compatibility aliases for the old GUI diagnostic recorder.
    "/api/v1/recording/start": "start",
    "/api/v1/recording/save": "save",
    "/api/v1/recording/discard": "discard",
}
_STATUS_PATHS = {SESSION_STATUS_PATH, "/api/v1/status"}
_TELEOP_COMMAND_PATH = "/api/v1/teleop/command"
_TELEOP_READY_PATH = "/api/v1/teleop/ready"
_TELEOP_STATE_PATH = "/api/v1/teleop/state"


@dataclass(frozen=True)
class CaptureRequest:
    """A validated GUI request queued for execution on the Kit-owning thread."""

    command: str
    path: str
    payload: dict


class CaptureControlServer:
    """Serve recorder status and enqueue explicit GUI capture actions on loopback."""

    def __init__(
        self, *, host: str = "127.0.0.1", port: int = DEFAULT_CAPTURE_API_PORT
    ) -> None:
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError as error:
            raise ValueError("capture control API must bind to a loopback IP") from error
        if not is_loopback:
            raise ValueError("capture control API must bind to a loopback IP")
        if not 0 <= int(port) <= 65535:
            raise ValueError("capture control API port must be between 0 and 65535")

        self._commands: Queue[CaptureRequest] = Queue()
        self._teleop_lock = threading.Lock()
        self._latest_teleop_command: dict | None = None
        self._teleop_state = {"ready": False, "message": "", "joint_state": {}}
        self._status_lock = threading.Lock()
        self._status = {
            "api_version": 1,
            "state": "idle",
            "ready": False,
            "recording": False,
            "steps": 0,
            "last_saved_path": "",
            "last_export_error": "",
            "last_quarantined_path": "",
            "last_qc_failure_reasons": [],
            "teleop_ready": False,
            "teleop_message": "",
            "message": "",
        }
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == _TELEOP_STATE_PATH:
                    self._reply(200, owner.teleop_state())
                    return
                if self.path not in _STATUS_PATHS:
                    self._reply(404, {"error": "not found"})
                    return
                self._reply(200, owner.status())

            def do_POST(self) -> None:
                if self.path in {_TELEOP_COMMAND_PATH, _TELEOP_READY_PATH}:
                    self._post_teleop()
                    return
                action = COMMAND_PATHS.get(self.path)
                if action is None:
                    self._reply(404, {"error": "not found"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._reply(400, {"error": "invalid content length"})
                    return
                if length < 0 or length > 4096:
                    self._reply(413, {"error": "request body too large"})
                    return
                body = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._reply(400, {"error": "request body must be JSON"})
                    return
                if not isinstance(payload, dict):
                    self._reply(400, {"error": "request body must be a JSON object"})
                    return
                owner._commands.put(CaptureRequest(action, self.path, payload))
                self._reply(202, {"accepted": True, "command": action})

            def _post_teleop(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._reply(400, {"error": "invalid content length"})
                    return
                if length < 0 or length > 4096:
                    self._reply(413, {"error": "request body too large"})
                    return
                try:
                    payload = json.loads((self.rfile.read(length) if length else b"{}").decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._reply(400, {"error": "request body must be JSON"})
                    return
                if not isinstance(payload, dict):
                    self._reply(400, {"error": "request body must be a JSON object"})
                    return
                try:
                    if self.path == _TELEOP_READY_PATH:
                        owner.update_teleop_state(
                            ready=payload.get("ready"),
                            message=payload.get("message", ""),
                        )
                    else:
                        owner.submit_teleop_command(payload)
                except (TypeError, ValueError, RuntimeError) as error:
                    self._reply(400, {"error": str(error)})
                    return
                self._reply(202, {"accepted": True})

            def _reply(self, status: int, payload: dict) -> None:
                encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args) -> None:
                return

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="arena-capture-control-api",
            daemon=True,
        )
        self._thread.start()

    def update_status(self, **values) -> None:
        with self._status_lock:
            self._status.update(values)

    def submit_teleop_command(self, payload: dict) -> None:
        """Keep only the latest operator command for the Kit-owning thread."""
        if not isinstance(payload, dict):
            raise TypeError("teleoperation command must be a JSON object")
        kind = payload.get("kind")
        if kind not in {"base_twist", "dual_arm", "estop"}:
            raise ValueError("teleoperation command kind must be base_twist, dual_arm, or estop")
        command = dict(payload)
        if kind == "base_twist":
            command["values"] = _finite_vector(payload.get("values"), 3, "base_twist requires three finite values")
        elif kind == "dual_arm":
            for side in ("left", "right"):
                values = payload.get(side)
                command[side] = (
                    None
                    if values is None
                    else _finite_vector(values, 8, f"{side} arm command requires eight finite values")
                )
        else:
            if not isinstance(payload.get("active"), bool):
                raise ValueError("estop command active must be boolean")
        source_seq = payload.get("source_seq")
        if source_seq is not None and (
            isinstance(source_seq, bool)
            or not isinstance(source_seq, int)
            or source_seq < 0
        ):
            raise ValueError("source_seq must be a non-negative integer")
        with self._teleop_lock:
            if not self._teleop_state["ready"]:
                raise RuntimeError("ROS 2 teleoperation sidecar is not ready")
            command["received_monotonic_ns"] = time.monotonic_ns()
            self._latest_teleop_command = command

    def take_latest_teleop_command(self) -> dict | None:
        """Consume the newest pending operator command without queue buildup."""
        with self._teleop_lock:
            command = self._latest_teleop_command
            self._latest_teleop_command = None
            return dict(command) if command is not None else None

    def update_teleop_state(
        self,
        *,
        ready: bool | None = None,
        message: str | None = None,
        joint_state: dict | None = None,
    ) -> None:
        """Cache the latest simulation feedback for the ROS VR IK node."""
        if ready is not None and not isinstance(ready, bool):
            raise TypeError("teleoperation readiness must be boolean")
        if message is not None and not isinstance(message, str):
            raise TypeError("teleoperation status message must be a string")
        if joint_state is not None and not isinstance(joint_state, dict):
            raise TypeError("teleoperation joint state must be a JSON object")
        with self._teleop_lock:
            if ready is not None:
                self._teleop_state["ready"] = ready
            if message is not None:
                self._teleop_state["message"] = message
            if joint_state is not None:
                self._teleop_state["joint_state"] = dict(joint_state)
            if ready is False:
                self._latest_teleop_command = None

    def teleop_state(self) -> dict:
        with self._teleop_lock:
            return dict(self._teleop_state)

    def status(self) -> dict:
        with self._status_lock:
            return dict(self._status)

    def take_commands(self) -> list[str]:
        """Drain command names for legacy callers that do not need payloads."""
        result = []
        while True:
            try:
                result.append(self._commands.get_nowait().command)
            except Empty:
                return result

    def take_requests(self) -> list[CaptureRequest]:
        """Drain validated command payloads for the worker control loop."""
        result = []
        while True:
            try:
                result.append(self._commands.get_nowait())
            except Empty:
                return result

    def close(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._server.shutdown()
            self._thread.join(timeout=2.0)
        self._server.server_close()
        self._thread = None


def _finite_vector(value, expected_size: int, message: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_size:
        raise ValueError(message)
    result = []
    for component in value:
        if isinstance(component, bool):
            raise ValueError(message)
        try:
            number = float(component)
        except (TypeError, ValueError) as error:
            raise ValueError(message) from error
        if not math.isfinite(number):
            raise ValueError(message)
        result.append(number)
    return result
