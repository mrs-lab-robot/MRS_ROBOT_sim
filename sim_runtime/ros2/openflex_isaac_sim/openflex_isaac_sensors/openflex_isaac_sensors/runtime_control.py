"""Runtime sensor lifecycle API, with all Kit/USD work on the app thread."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import re
import threading
import time
from typing import Callable
from urllib.parse import unquote, urlparse


class RuntimeSensorControlError(RuntimeError):
    """A sensor lifecycle request could not be completed."""


@dataclass
class RuntimeSensorResource:
    """A created sensor and the Kit-thread readiness/cleanup operations it owns."""

    destroy: Callable[[], None]
    ready: Callable[[], bool] = lambda: True
    destroy_ready: Callable[[], bool] = lambda: True


@dataclass
class _Sensor:
    label: str
    topic: str
    create: Callable[[], RuntimeSensorResource | Callable[[], None]]
    state: str = "inactive"
    error: str = ""
    resource: RuntimeSensorResource | None = None


@dataclass
class _Request:
    sensor_id: str
    action: str
    previous_state: str
    deadline: float
    done: threading.Event
    error: BaseException | None = None


@dataclass
class _Creating:
    request: _Request
    sensor: _Sensor
    resource: RuntimeSensorResource


@dataclass
class _Destroying:
    request: _Request
    sensor: _Sensor
    resource: RuntimeSensorResource
    final_state: str = "inactive"
    final_error: str = ""


class RuntimeSensorManager:
    """Keep sensor resources in a registry and serialize lifecycle changes."""

    _SENSOR_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
    _ACTIONS = {"create", "destroy"}

    def __init__(self, *, request_timeout_s: float = 30.0) -> None:
        self._owner_thread = threading.get_ident()
        self._request_timeout_s = max(0.1, float(request_timeout_s))
        self._sensors: dict[str, _Sensor] = {}
        self._pending: queue.Queue[_Request] = queue.Queue()
        self._creating: dict[str, _Creating] = {}
        self._destroying: dict[str, _Destroying] = {}
        self._ready = False
        self._error = ""
        self._lock = threading.RLock()

    def register(
        self,
        sensor_id: str,
        *,
        label: str,
        topic: str,
        create: Callable[[], RuntimeSensorResource | Callable[[], None]],
    ) -> None:
        if not self._SENSOR_ID.fullmatch(sensor_id):
            raise ValueError(f"invalid sensor id: {sensor_id!r}")
        if not label.strip() or not topic.startswith("/"):
            raise ValueError("sensor label and absolute ROS 2 topic are required")
        if not callable(create):
            raise TypeError("create must be callable")
        with self._lock:
            if self._ready:
                raise RuntimeSensorControlError("sensor registry is already ready")
            if sensor_id in self._sensors:
                raise ValueError(f"sensor already registered: {sensor_id}")
            self._sensors[sensor_id] = _Sensor(label.strip(), topic.strip(), create)

    def mark_ready(self, *, error: str = "") -> None:
        with self._lock:
            self._ready = True
            self._error = str(error).strip()

    def bind_application_thread(self) -> None:
        """Bind to Kit's update thread before exposing the registry."""
        with self._lock:
            if self._owner_thread == threading.get_ident():
                return
            if self._ready or not self._pending.empty():
                raise RuntimeSensorControlError(
                    "application thread cannot change after sensor control is ready"
                )
            self._owner_thread = threading.get_ident()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "ready": self._ready,
                "error": self._error,
                "sensors": {
                    sensor_id: {
                        "label": sensor.label,
                        "topic": sensor.topic,
                        "state": sensor.state,
                        "active": sensor.state == "active",
                        "resource_present": sensor.resource is not None,
                        "error": sensor.error,
                    }
                    for sensor_id, sensor in sorted(self._sensors.items())
                },
            }

    def request(self, sensor_id: str, action: str) -> dict:
        if not isinstance(action, str) or action not in self._ACTIONS:
            raise ValueError("action must be 'create' or 'destroy'")
        with self._lock:
            if not self._ready:
                raise RuntimeSensorControlError("robot sensor registry is not ready")
            sensor = self._sensors.get(sensor_id)
            if sensor is None:
                raise KeyError(sensor_id)

            if action == "create" and sensor.state == "active":
                return self.snapshot()
            if action == "destroy" and sensor.state == "inactive":
                return self.snapshot()
            if action == "create" and sensor.state == "error" and sensor.resource is not None:
                raise RuntimeSensorControlError(
                    f"sensor {sensor_id} still owns resources; destroy it before retrying create"
                )
            if sensor.state in {"creating", "destroying"}:
                raise RuntimeSensorControlError(
                    f"sensor {sensor_id} is already {sensor.state}"
                )
            if action == "destroy" and sensor.state == "error" and sensor.resource is None:
                sensor.state = "inactive"
                sensor.error = ""
                return self.snapshot()

            previous_state = sensor.state
            sensor.state = "creating" if action == "create" else "destroying"
            sensor.error = ""

        request = _Request(
            sensor_id=sensor_id,
            action=action,
            previous_state=previous_state,
            deadline=time.monotonic() + self._request_timeout_s,
            done=threading.Event(),
        )
        self._pending.put(request)
        if not request.done.wait(self._request_timeout_s):
            raise TimeoutError(f"timed out waiting to {action} sensor {sensor_id}")
        if request.error is not None:
            raise RuntimeSensorControlError(str(request.error)) from request.error
        return self.snapshot()

    def process_pending(self) -> int:
        """Create/destroy Kit resources; call only from the app update thread."""
        if threading.get_ident() != self._owner_thread:
            raise RuntimeSensorControlError(
                "sensor lifecycle changes must run on the Isaac application thread"
            )
        processed = self._poll_creating()
        processed += self._poll_destroying()
        while True:
            try:
                request = self._pending.get_nowait()
            except queue.Empty:
                return processed
            processed += 1
            with self._lock:
                sensor = self._sensors.get(request.sensor_id)
            defer_completion = False
            try:
                if time.monotonic() > request.deadline:
                    raise TimeoutError("sensor request expired before Kit applied it")
                if sensor is None:
                    raise KeyError(request.sensor_id)
                if request.action == "create":
                    created = sensor.create()
                    if isinstance(created, RuntimeSensorResource):
                        resource = created
                    elif callable(created):
                        resource = RuntimeSensorResource(destroy=created)
                    else:
                        raise TypeError(
                            "sensor factory must return a resource or cleanup callback"
                        )
                    with self._lock:
                        self._creating[request.sensor_id] = _Creating(
                            request=request, sensor=sensor, resource=resource
                        )
                        sensor.resource = resource
                    processed += self._poll_creating(only=request.sensor_id)
                else:
                    resource = sensor.resource
                    if resource is not None:
                        resource.destroy()
                        if not resource.destroy_ready():
                            with self._lock:
                                self._destroying[request.sensor_id] = _Destroying(
                                    request=request,
                                    sensor=sensor,
                                    resource=resource,
                                )
                            defer_completion = True
                    if not defer_completion:
                        with self._lock:
                            sensor.resource = None
                            sensor.state = "inactive"
                            sensor.error = ""
            except BaseException as exc:
                request.error = exc
                with self._lock:
                    if sensor is not None:
                        creating = self._creating.pop(request.sensor_id, None)
                        if creating is not None:
                            try:
                                creating.resource.destroy()
                            except BaseException as cleanup_error:
                                exc = RuntimeSensorControlError(
                                    f"{exc}; sensor cleanup also failed: {cleanup_error}"
                                )
                                request.error = exc
                            sensor.resource = None
                        sensor.state = "error"
                        sensor.error = str(exc)
            finally:
                if (request.action == "destroy" and not defer_completion) or request.error is not None:
                    request.done.set()

    def _poll_creating(self, *, only: str | None = None) -> int:
        """Advance asynchronous resource warm-up on each Kit update tick."""
        with self._lock:
            pending = tuple(
                (sensor_id, creating)
                for sensor_id, creating in self._creating.items()
                if only is None or sensor_id == only
            )
        processed = 0
        for sensor_id, creating in pending:
            request, sensor, resource = (
                creating.request,
                creating.sensor,
                creating.resource,
            )
            try:
                if time.monotonic() > request.deadline:
                    raise TimeoutError("sensor initialization exceeded its time limit")
                if resource.ready():
                    with self._lock:
                        self._creating.pop(sensor_id, None)
                        sensor.state = "active"
                        sensor.error = ""
                    request.done.set()
                    processed += 1
            except BaseException as exc:
                cleanup_pending = False
                cleanup_failed = False
                try:
                    resource.destroy()
                    cleanup_pending = not resource.destroy_ready()
                except BaseException as cleanup_error:
                    cleanup_failed = True
                    exc = RuntimeSensorControlError(
                        f"{exc}; sensor cleanup also failed: {cleanup_error}"
                    )
                with self._lock:
                    self._creating.pop(sensor_id, None)
                    sensor.state = "error"
                    sensor.error = str(exc)
                    if cleanup_pending:
                        sensor.resource = resource
                        self._destroying[sensor_id] = _Destroying(
                            request=request,
                            sensor=sensor,
                            resource=resource,
                            final_state="error",
                            final_error=str(exc),
                        )
                    elif not cleanup_failed:
                        sensor.resource = None
                request.error = exc
                request.done.set()
                processed += 1
        return processed

    def _poll_destroying(self) -> int:
        """Wait for Kit-owned resources to finish asynchronous teardown."""
        with self._lock:
            pending = tuple(self._destroying.items())
        processed = 0
        for sensor_id, destroying in pending:
            request, sensor, resource = (
                destroying.request,
                destroying.sensor,
                destroying.resource,
            )
            try:
                if time.monotonic() > request.deadline:
                    raise TimeoutError("sensor cleanup exceeded its time limit")
                if resource.destroy_ready():
                    with self._lock:
                        self._destroying.pop(sensor_id, None)
                        sensor.resource = None
                        sensor.state = destroying.final_state
                        sensor.error = destroying.final_error
                    request.done.set()
                    processed += 1
            except BaseException as exc:
                with self._lock:
                    self._destroying.pop(sensor_id, None)
                    sensor.state = "error"
                    sensor.error = (
                        f"{destroying.final_error}; cleanup also failed: {exc}"
                        if destroying.final_error
                        else str(exc)
                    )
                request.error = exc
                request.done.set()
                processed += 1
        return processed


def install_main_loop_sensor_pump(
    simulation_loop: object,
    manager: RuntimeSensorManager,
) -> None:
    """Poll lifecycle requests from the runner's per-frame main-thread hook.

    The Isaac REST runner calls ``process_commands`` once before every Kit
    update. Chaining sensor work there keeps USD and OmniGraph operations on
    the Kit thread and advances asynchronous readiness on every frame.
    """
    process_commands = getattr(simulation_loop, "process_commands", None)
    if not callable(process_commands):
        raise TypeError("simulation loop must provide a callable process_commands method")

    def process_commands_with_sensor_pump(*args, **kwargs):
        result = process_commands(*args, **kwargs)
        manager.bind_application_thread()
        manager.process_pending()
        return result

    setattr(simulation_loop, "process_commands", process_commands_with_sensor_pump)


class RuntimeSensorControlServer:
    """Loopback-only HTTP API consumed by the GUI through local/SSH commands."""

    def __init__(
        self,
        manager: RuntimeSensorManager,
        *,
        host: str = "127.0.0.1",
        port: int = 8086,
    ) -> None:
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("sensor control API must bind to loopback")
        self.manager = manager
        self.host = host
        self.port = int(port)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            return f"http://{self.host}:{self.port}"
        address, port = self._server.server_address[:2]
        return f"http://{address}:{port}"

    def start(self) -> None:
        if self._server is not None:
            return
        manager = self.manager

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if urlparse(self.path).path != "/v1/sensors":
                    self._json(404, {"error": "not found"})
                    return
                self._json(200, manager.snapshot())

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                prefix = "/v1/sensors/"
                if not path.startswith(prefix):
                    self._json(404, {"error": "not found"})
                    return
                sensor_id = unquote(path[len(prefix):])
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if size < 1 or size > 1024:
                        raise ValueError("request body must be between 1 and 1024 bytes")
                    payload = json.loads(self.rfile.read(size).decode("utf-8"))
                    if not isinstance(payload, dict) or set(payload) != {"action"}:
                        raise ValueError("body must contain only the action field")
                    result = manager.request(sensor_id, payload["action"])
                except KeyError:
                    self._json(404, {"error": f"unknown sensor: {sensor_id}"})
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    self._json(400, {"error": str(exc)})
                except TimeoutError as exc:
                    self._json(504, {"error": str(exc)})
                except RuntimeSensorControlError as exc:
                    self._json(409, {"error": str(exc)})
                else:
                    self._json(200, result)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="isaac-sensor-control-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
