#!/usr/bin/env python3
"""复测历史独立 MID360 链路，并区分 raw PointCloud2 与 Livox CustomMsg。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "isaac_sim_core" / "assets" / "environments" / "mid360_empty_stage.usda"
RAW_ROOT = ROOT / "reports" / "sensors" / "raw"
REPORT_PATH = ROOT / "reports" / "sensors" / "sensor_validation.md"
REPORT_MARKER = "<!-- MID360_STANDALONE_BATCHES -->"
RAW_TOPIC = "/openflex/livox_frame/lidar"
CUSTOM_TOPIC = "/livox/lidar"
BRIDGE_SCRIPT = (
    ROOT
    / "ros2_pkgs"
    / "openflex_isaac_sim"
    / "openflex_isaac_bringup"
    / "scripts"
    / "isaacsim_compat_bridge.py"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=("performance", "full"), default="performance")
    # The historical passing path is LidarSensor-owned. Keep graph_owned only
    # as an explicit compatibility experiment for older Isaac installations.
    parser.add_argument("--transport", choices=("graph_owned", "direct"), default="direct")
    parser.add_argument("--stage", type=Path, default=STAGE)
    # The outer harness creates its own evidence batch.  The worker receives
    # an explicit path from that harness, so only the worker requires it.
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--physics-hz", type=float, default=90.0)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--warmup-steps", type=int, default=90)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--kit-threads", type=int, default=16)
    parser.add_argument("--ros-domain-id", type=int, default=44)
    parser.add_argument("--without-compat-bridge", action="store_true")
    parser.add_argument("--executor", default=os.environ.get("ISAACSIM_TEST_EXECUTOR", "manual"))
    parser.add_argument("--append-report", action="store_true")
    return parser


def _profile(profile: str) -> tuple[str, int]:
    return (
        ("MID360_PERFORMANCE", 30_000)
        if profile == "performance"
        else ("MID360_APPROX", 300_000)
    )


def _pace_until(target_ns: int) -> None:
    """Use the historical online timing model: overload shows as RTF < 1."""
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 1_500_000:
            time.sleep((remaining_ns - 750_000) / 1e9)
        else:
            time.sleep(0)


def _worker(args: argparse.Namespace) -> int:
    """Isaac-Python child: build exactly the independent historical graph."""
    if args.output is None:
        raise ValueError("--worker requires --output")
    if not args.stage.is_file():
        raise FileNotFoundError(f"standalone MID360 stage does not exist: {args.stage}")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.physics_hz <= 0.0 or args.render_hz <= 0.0:
        raise ValueError("physics_hz and render_hz must be positive")
    if args.warmup_steps < 0 or args.steps < 0:
        raise ValueError("warmup_steps and steps must be non-negative")
    worker_state_path = args.output / "worker_state.json"
    worker_state_path.write_text(
        json.dumps({"state": "entered", "argv": sys.argv}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sensor_package = (
        ROOT
        / "ros2_pkgs"
        / "openflex_isaac_sim"
        / "openflex_isaac_sensors"
    )
    if str(sensor_package) not in sys.path:
        sys.path.insert(0, str(sensor_package))

    from isaacsim import SimulationApp

    def _json_value(value: object) -> object:
        """Convert common USD/OmniGraph values into durable diagnostic data."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [_json_value(item) for item in value]
        for attribute in ("pathString", "GetString"):
            try:
                converted = getattr(value, attribute)
                converted = converted() if callable(converted) else converted
                return str(converted)
            except Exception:
                pass
        return str(value)

    def _write_graph_diagnostics(graph_path: str, og_module: object, output: Path) -> None:
        diagnostics: dict[str, object] = {"graph_path": graph_path, "nodes": []}
        try:
            graph = og_module.get_graph_by_path(graph_path)
            if graph is None:
                diagnostics["error"] = "graph_not_found"
            else:
                for node in graph.get_nodes():
                    node_record: dict[str, object] = {
                        "path": _json_value(node.get_prim_path()),
                        "type": node.get_type_name(),
                    }
                    attributes: dict[str, object] = {}
                    for name in (
                        "inputs:cameraPrim",
                        "inputs:renderProductPath",
                        "inputs:selectedMetadata",
                        "outputs:renderProductPath",
                        "outputs:execOut",
                    ):
                        try:
                            attribute = node.get_attribute(name)
                            attributes[name] = _json_value(attribute.get())
                        except Exception as exc:
                            attributes[name] = f"<error: {exc}>"
                    node_record["attributes"] = attributes
                    try:
                        if node.get_type_name() == "isaacsim.core.nodes.IsaacCreateRenderProduct":
                            from isaacsim.core.nodes.ogn.OgnIsaacCreateRenderProductDatabase import (
                                OgnIsaacCreateRenderProductDatabase,
                            )

                            state = OgnIsaacCreateRenderProductDatabase.per_instance_internal_state(node)
                            node_record["internal_state"] = {
                                "initialized": bool(getattr(state, "initialized", False)),
                                "render_product_path": _json_value(getattr(state, "render_product_path", None)),
                                "camera_path": _json_value(getattr(state, "camera_path", None)),
                            }
                        elif node.get_type_name() == "isaacsim.ros2.bridge.ROS2RtxLidarHelper":
                            from isaacsim.ros2.nodes.ogn.OgnROS2RtxLidarHelperDatabase import (
                                OgnROS2RtxLidarHelperDatabase,
                            )

                            state = OgnROS2RtxLidarHelperDatabase.per_instance_internal_state(node)
                            node_record["internal_state"] = {
                                "initialized": bool(getattr(state, "initialized", False)),
                                "render_product_path": _json_value(getattr(state, "render_product_path", None)),
                                "writer_count": len(getattr(state, "_writers", []) or []),
                            }
                    except Exception as exc:
                        node_record["internal_state_error"] = str(exc)
                    diagnostics["nodes"].append(node_record)
        except Exception as exc:
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        (output / "graph_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    simulation_app = SimulationApp(
        {
            "headless": True,
            "disable_viewport_updates": True,
            "extra_args": [
                "--/app/runLoops/main/rateLimitEnabled=false",
                "--/app/runLoops/main/manualModeEnabled=true",
                "--/rtx/hydra/supportMultiTickRate=true",
                # The historical MID360 standalone path was validated on a
                # single renderer device.  These must be Kit startup options;
                # changing the settings after SimulationApp construction is
                # too late (the command line would still enable multi-GPU).
                "--/renderer/multiGpu/enabled=false",
                "--/exts/omni.replicator.srtx/enabled=false",
            ],
            "limit_cpu_threads": args.kit_threads,
        }
    )
    try:
        worker_state_path.write_text(
            json.dumps({"state": "simulation_app_started"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # The old standalone launch used the REST runner, which enabled this
        # extension before it built any ROS graph. A bare SimulationApp does
        # not load it implicitly.
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.ros2.bridge")
        for _ in range(4):
            simulation_app.update()
        import carb.settings
        import omni.graph.core as og
        import omni.timeline
        import omni.usd
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

        # Reproduce the historical sensor-library startup contract.  The
        # helper path is backed by the Replicator writer and must be selected
        # before the first ROS2RtxLidarHelper/RenderProduct is created.  The
        # old library also validated the standalone sensor on one GPU; keeping
        # that setting explicit avoids inheriting a user's multi-GPU profile.
        settings = carb.settings.get_settings()
        settings.set("/exts/omni.replicator.srtx/enabled", False)
        settings.set("/renderer/multiGpu/enabled", False)

        from openflex_isaac_sensors.mid360 import create_standalone_mid360

        stage_utils.open_stage(str(args.stage))
        for _ in range(3):
            simulation_app.update()
        stage = omni.usd.get_context().get_stage()
        if stage is None or not stage.GetPrimAtPath("/World").IsValid():
            raise RuntimeError("failed to load standalone MID360 stage")

        # ``start_sim_with_rest_api.py`` historically supplied this clock.
        # Keep it in the isolated worker so RTF comes from the same sim time
        # stream that timestamps PointCloud2.
        keys = og.Controller.Keys
        from omni.graph.core import GraphPipelineStage

        clock_graph, _, _, _ = og.Controller.edit(
            {
                "graph_path": "/Graph/ROS_Clock",
                "evaluator_name": "execution",
                "pipeline_stage": GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: [
                    ("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.CONNECT: [
                    ("OnPhysicsStep.outputs:step", "PublishClock.inputs:execIn"),
                    ("Context.outputs:context", "PublishClock.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                keys.SET_VALUES: [("ReadSimTime.inputs:resetOnStop", False)],
            },
        )
        og.Controller.evaluate_sync(clock_graph)

        profile_name, _max_points = _profile(args.profile)
        created = create_standalone_mid360(
            stage, profile=profile_name, transport=args.transport
        )
        worker_state_path.write_text(
            json.dumps({"state": "sensor_created", "created": created}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        RenderingManager.set_dt(1.0 / args.render_hz)
        dt = 1.0 / args.physics_hz
        SimulationManager.setup_simulation(dt=dt, device="cpu")
        physics_step = 0
        wall_times_ns: list[int] = []
        step_times_s: list[float] = []

        def _on_physics_step(_step_dt: float, _context: object | None = None) -> None:
            nonlocal physics_step
            physics_step += 1
            step_times_s.append(physics_step * dt)
            wall_times_ns.append(time.perf_counter_ns())

        callback_id = SimulationManager.register_callback(
            _on_physics_step, IsaacEvents.POST_PHYSICS_STEP
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(3):
            simulation_app.update()
        for _ in range(args.warmup_steps):
            simulation_app.update()

        _write_graph_diagnostics(created["graph_path"], og, args.output)

        (args.output / "ready.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ready",
                    "profile": profile_name,
                    "created": created,
                    "warmup_steps": args.warmup_steps,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Shader compilation and render-product bootstrap belong to warm-up,
        # not to the benchmark interval.
        wall_times_ns.clear()
        step_times_s.clear()
        measured_start_step = physics_step
        target_steps = measured_start_step + (args.steps or math.ceil((args.duration + 8.0) * args.physics_hz))
        app_update_ns: list[int] = []
        anchor_ns = time.perf_counter_ns()
        while simulation_app.is_running() and physics_step < target_steps:
            begin_ns = time.perf_counter_ns()
            simulation_app.update()
            app_update_ns.append(time.perf_counter_ns() - begin_ns)
            elapsed_steps = physics_step - measured_start_step
            if elapsed_steps > 0:
                _pace_until(anchor_ns + int(round(elapsed_steps * dt * 1e9)))

        timeline.stop()
        SimulationManager.deregister_callback(callback_id)
        wall_duration_s = (
            (wall_times_ns[-1] - wall_times_ns[0]) / 1e9 if len(wall_times_ns) >= 2 else None
        )
        sim_duration_s = step_times_s[-1] - step_times_s[0] if len(step_times_s) >= 2 else None
        gaps_s = [
            (later - earlier) / 1e9 for earlier, later in zip(wall_times_ns, wall_times_ns[1:])
        ]
        sorted_gaps = sorted(gaps_s)
        p99_gap_s = (
            sorted_gaps[min(len(sorted_gaps) - 1, round((len(sorted_gaps) - 1) * 0.99))]
            if sorted_gaps
            else None
        )
        metrics = {
            "schema_version": 1,
            "status": "complete",
            "profile": profile_name,
            "stage": str(args.stage),
            "physics_hz_configured": args.physics_hz,
            "render_hz_configured": args.render_hz,
            "warmup_steps": args.warmup_steps,
            "physics_steps": len(wall_times_ns),
            "physics_wall_hz": ((len(wall_times_ns) - 1) / wall_duration_s if wall_duration_s else None),
            "physics_rtf": (sim_duration_s / wall_duration_s if sim_duration_s and wall_duration_s else None),
            "physics_wall_gap_p99_s": p99_gap_s,
            "physics_wall_gap_max_s": max(gaps_s) if gaps_s else None,
            "app_update_wall_mean_s": (sum(app_update_ns) / len(app_update_ns) / 1e9 if app_update_ns else None),
            "app_update_wall_max_s": (max(app_update_ns) / 1e9 if app_update_ns else None),
            "created": created,
            "kit_runtime": {
                "rate_limit_enabled": settings.get_as_bool("/app/runLoops/main/rateLimitEnabled"),
                "manual_mode_enabled": settings.get_as_bool("/app/runLoops/main/manualModeEnabled"),
                "srtx_enabled": settings.get_as_bool("/exts/omni.replicator.srtx/enabled"),
                "renderer_multi_gpu_enabled": settings.get_as_bool("/renderer/multiGpu/enabled"),
                "tasking_thread_count": settings.get_as_int("/plugins/carb.tasking.plugin/threadCount"),
                "tbb_max_thread_count": settings.get_as_int("/plugins/omni.tbb.globalcontrol/maxThreadCount"),
            },
        }
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Isaac Sim 6.0 can block indefinitely in SimulationApp.close() after
        # a ROS2RtxLidarHelper graph has published. Metrics are already
        # durable and the worker owns no shared resources, so exit the child
        # process directly; the OS releases its Kit/GPU/DDS resources.
        os._exit(0)
    except BaseException as exc:
        (args.output / "worker_failure.json").write_text(
            json.dumps(
                {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        simulation_app.close()


@dataclass
class StreamSamples:
    wall: list[float] = field(default_factory=list)
    sim: list[float] = field(default_factory=list)
    nonempty_messages: int = 0
    point_counts: list[int] = field(default_factory=list)
    backward_count: int = 0
    duplicate_count: int = 0

    def add(self, sim_time: float, points: int) -> None:
        if self.sim and sim_time < self.sim[-1]:
            self.backward_count += 1
            return
        if self.sim and sim_time == self.sim[-1]:
            self.duplicate_count += 1
            return
        self.wall.append(time.monotonic())
        self.sim.append(sim_time)
        self.point_counts.append(points)
        if points > 0:
            self.nonempty_messages += 1

    def reset(self) -> None:
        self.wall.clear()
        self.sim.clear()
        self.point_counts.clear()
        self.nonempty_messages = 0
        self.backward_count = 0
        self.duplicate_count = 0

    def metrics(self) -> dict[str, Any]:
        if len(self.wall) < 2 or len(self.sim) < 2:
            return {
                "messages": len(self.wall),
                "wall_receive_hz": None,
                "sim_timestamp_hz": None,
                "nonempty_messages": self.nonempty_messages,
                "point_count_mean": None,
                "point_count_min": min(self.point_counts) if self.point_counts else None,
                "point_count_max": max(self.point_counts) if self.point_counts else None,
                "backward_count": self.backward_count,
                "duplicate_count": self.duplicate_count,
            }
        wall_duration = self.wall[-1] - self.wall[0]
        sim_duration = self.sim[-1] - self.sim[0]
        return {
            "messages": len(self.wall),
            "wall_receive_hz": ((len(self.wall) - 1) / wall_duration if wall_duration > 0.0 else None),
            "sim_timestamp_hz": ((len(self.sim) - 1) / sim_duration if sim_duration > 0.0 else None),
            "nonempty_messages": self.nonempty_messages,
            "point_count_mean": sum(self.point_counts) / len(self.point_counts),
            "point_count_min": min(self.point_counts),
            "point_count_max": max(self.point_counts),
            "backward_count": self.backward_count,
            "duplicate_count": self.duplicate_count,
        }


def _default_isaac_path() -> Path:
    configured = os.environ.get("ISAACSIM_PATH", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.home() / "isaacsim-6.0", Path.home() / "isaacsim", Path("/isaac-sim")))
    for candidate in candidates:
        if (candidate / "python.sh").is_file():
            return candidate
    raise FileNotFoundError("Isaac Sim python.sh was not found; set ISAACSIM_PATH")


def _isaac_ros2_environment(base: dict[str, str], isaac_path: Path) -> dict[str, str]:
    """Build the environment required by Isaac Sim's bundled ROS 2 bridge.

    Isaac Sim 6 embeds Python 3.12 and its own ROS 2 Humble C++ libraries.
    The parent shell is normally a ROS 2 Humble Python 3.10 environment, so
    passing its ``PYTHONPATH`` through makes Isaac try to import an ABI-
    incompatible system ``rclpy``.  More importantly, without the bundled
    Humble library directory first in ``LD_LIBRARY_PATH`` the bridge cannot
    load its internal RMW implementation and publishers remain silent even
    though the OmniGraph node reports ``initialized=true``.
    """
    environment = dict(base)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["ROS_DISTRO"] = "humble"
    environment["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    environment.pop("ROS_DISCOVERY_SERVER", None)

    candidates = [
        isaac_path / "exts" / "isaacsim.ros2.core" / "humble" / "lib",
        isaac_path / "exts" / "isaacsim.ros2.core-*" / "humble" / "lib",
        Path.home() / ".local" / "share" / "ov" / "data" / "Kit" / "Isaac-Sim Full" / "5.1" / "exts" / "3",
    ]
    ros2_lib: Path | None = None
    for candidate in candidates:
        if candidate.is_dir():
            ros2_lib = candidate
            break
    if ros2_lib is None:
        matches = sorted(
            path / "humble" / "lib"
            for path in isaac_path.glob("exts/isaacsim.ros2.core-*")
            if (path / "humble" / "lib").is_dir()
        )
        if matches:
            ros2_lib = matches[0]
    if ros2_lib is None:
        raise FileNotFoundError(
            f"Isaac Sim bundled ROS 2 Humble libraries were not found below {isaac_path}"
        )

    existing = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(ros2_lib)] + [entry for entry in existing if entry and entry != str(ros2_lib)]
    )

    fastdds_candidates = [
        Path(environment["FASTRTPS_DEFAULT_PROFILES_FILE"])
        if environment.get("FASTRTPS_DEFAULT_PROFILES_FILE")
        else None,
        ROOT.parent / "isaac_ros2_utils" / "isaac_ros2_scripts" / "config" / "fastdds.xml",
    ]
    for fastdds in fastdds_candidates:
        if fastdds is not None and fastdds.is_file():
            environment["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(fastdds)
            break
    return environment


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=30.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _existing_isaac_processes(domain_id: int) -> list[str]:
    """Return unrelated Isaac processes so a benchmark cannot hide contention."""
    result = subprocess.run(
        ["ps", "-eo", "pid=,comm=,args="], check=False, text=True, capture_output=True
    )
    this_pid = str(os.getpid())
    matches = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 3:
            continue
        pid, command, arguments = fields
        if pid == this_pid:
            continue
        arguments_lower = arguments.lower()
        is_kit_python = command == "python" and "/kit/python" in arguments_lower
        is_robot_launcher = "start_robot_control_sim.py" in arguments
        is_isaac_launcher = command in {"isaac-sim.sh", "python.sh"} and "isaacsim" in arguments_lower
        if not (is_kit_python or is_robot_launcher or is_isaac_launcher):
            continue
        matches.append(line.strip())
    return matches


def _system_run(args: argparse.Namespace) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import PointCloud2

    custom_available = True
    try:
        from livox_ros_driver2.msg import CustomMsg
    except ImportError:
        custom_available = False
        CustomMsg = None  # type: ignore[assignment,misc]

    class Audit(Node):
        def __init__(self) -> None:
            super().__init__("mid360_standalone_performance_audit")
            self.raw = StreamSamples()
            self.custom = StreamSamples()
            # ``rclpy.node.Node`` exposes a ``clock`` property in some ROS 2
            # distributions.  Do not shadow/use that name for our samples:
            # the old harness failed at ``node.clock.reset()`` with
            # ``'ROSClock' object is not callable`` before entering the
            # measurement window.
            self.clock_samples = StreamSamples()
            # Isaac RTX sensor writers may expose either RELIABLE or
            # BEST_EFFORT depending on the selected bridge/runtime. The ROS 2
            # sensor-data profile can match both, while a RELIABLE subscriber
            # cannot match a BEST_EFFORT publisher and would produce a false
            # "publisher exists but no samples" failure.
            self.create_subscription(PointCloud2, RAW_TOPIC, self._raw, qos_profile_sensor_data)
            self.create_subscription(Clock, "/clock", self._clock_cb, qos_profile_sensor_data)
            if custom_available and not args.without_compat_bridge:
                self.create_subscription(CustomMsg, CUSTOM_TOPIC, self._custom, qos_profile_sensor_data)

        def _raw(self, message: PointCloud2) -> None:
            stamp = message.header.stamp
            self.raw.add(float(stamp.sec) + float(stamp.nanosec) / 1e9, int(message.width) * max(1, int(message.height)))

        def _custom(self, message: Any) -> None:
            stamp = message.header.stamp
            self.custom.add(float(stamp.sec) + float(stamp.nanosec) / 1e9, int(message.point_num))

        def _clock_cb(self, message: Clock) -> None:
            stamp = message.clock
            self.clock_samples.add(float(stamp.sec) + float(stamp.nanosec) / 1e9, 1)

    profile_name, max_points = _profile(args.profile)
    batch_dir = args.output
    worker_dir = batch_dir / "isaac"
    worker_dir.mkdir(parents=True, exist_ok=False)
    worker_log = batch_dir / "isaac.launch.log"
    bridge_log = batch_dir / "compat_bridge.log"
    worker_steps = args.steps or math.ceil((args.duration + 8.0) * args.physics_hz)
    isaac_path = _default_isaac_path()
    command = [
        str(isaac_path / "python.sh"),
        str(Path(__file__).resolve()),
        "--worker",
        "--profile", args.profile,
        "--transport", args.transport,
        "--stage", str(args.stage),
        "--output", str(worker_dir),
        "--duration", str(args.duration),
        "--physics-hz", str(args.physics_hz),
        "--render-hz", str(args.render_hz),
        "--warmup-steps", str(args.warmup_steps),
        "--steps", str(worker_steps),
        "--kit-threads", str(args.kit_threads),
    ]
    environment = os.environ.copy()
    environment["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    environment["ROS_LOCALHOST_ONLY"] = "1"
    environment = _isaac_ros2_environment(environment, isaac_path)
    # Isaac Sim 6 must not inherit the system ROS 2 Python path, but the
    # compatibility bridge is a normal system-Python ROS 2 process and does
    # need that path to import ``rclpy`` and ``livox_ros_driver2``.  Keep two
    # deliberately separate environments instead of passing the Isaac
    # environment to both children.
    bridge_environment = os.environ.copy()
    bridge_environment["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    bridge_environment["ROS_LOCALHOST_ONLY"] = "1"
    bridge_environment["ROS_DISTRO"] = "humble"
    bridge_environment["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    result: dict[str, Any] = {
        "status": "TEST_ERROR",
        "profile": profile_name,
        "transport": args.transport,
        "profile_point_limit": max_points,
        "raw_topic": RAW_TOPIC,
        "custom_topic": CUSTOM_TOPIC,
        "compat_bridge_enabled": not args.without_compat_bridge,
        "custom_msg_type_available": custom_available,
        "concurrent_isaac_processes_before_start": _existing_isaac_processes(args.ros_domain_id),
        "worker_command": command,
        "worker_log": str(worker_log.relative_to(ROOT)),
        "bridge_log": str(bridge_log.relative_to(ROOT)),
        "worker_metrics": str((worker_dir / "metrics.json").relative_to(ROOT)),
    }
    if result["concurrent_isaac_processes_before_start"]:
        result["status"] = "TEST_ERROR_CONCURRENT_ISAAC"
        result["failure"] = (
            "another Isaac Sim process is already using the benchmark GPU; "
            "the benchmark was not started"
        )
        return result
    worker: subprocess.Popen[str] | None = None
    bridge: subprocess.Popen[str] | None = None
    node: Node | None = None
    rclpy.init()
    try:
        node = Audit()
        if not args.without_compat_bridge:
            if not custom_available:
                result["failure"] = "livox_ros_driver2 CustomMsg is unavailable; cannot run requested bridge test"
                return result
            bridge_command = [
                sys.executable,
                str(BRIDGE_SCRIPT),
                "--ros-args",
                "-p", f"sim_lidar_topic:={RAW_TOPIC}",
                "-p", f"livox_lidar_mode:=custom",
                "-p", f"livox_lidar_topic:={CUSTOM_TOPIC}",
                "-p", "livox_line_count:=4",
                "-p", f"livox_max_points:={max_points}",
                "-p", "sensor_only_mode:=true",
                "-p", "publish_livox_imu:=false",
                "-p", "publish_battery_state:=false",
                "-p", "publish_scan_from_pointcloud:=false",
                "-p", "publish_lift_status:=false",
                "-p", "publish_mapping_stub_services:=false",
                "-p", "relay_cmd_vel_safe:=false",
                "-p", "sensor_queue_depth:=1",
            ]
            result["bridge_command"] = bridge_command
            bridge = subprocess.Popen(
                bridge_command,
                cwd=ROOT,
                stdout=bridge_log.open("w", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                env=bridge_environment,
                text=True,
                start_new_session=True,
            )

        worker = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=worker_log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            start_new_session=True,
        )
        ready_path = worker_dir / "ready.json"
        startup_deadline = time.monotonic() + args.startup_timeout
        custom_required = not args.without_compat_bridge and custom_available
        while rclpy.ok() and time.monotonic() < startup_deadline and worker.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if _read_json(ready_path) and node.raw.wall and (not custom_required or node.custom.wall):
                break
        if not _read_json(ready_path):
            result["failure"] = "Isaac worker did not enter ready state before startup timeout"
            return result
        if not node.raw.wall:
            result["failure"] = "standalone MID360 did not publish raw PointCloud2 before startup timeout"
            return result
        if custom_required and not node.custom.wall:
            result["failure"] = "compat bridge did not publish Livox CustomMsg before startup timeout"
            return result

        node.raw.reset()
        node.custom.reset()
        node.clock_samples.reset()
        deadline = time.monotonic() + args.duration
        while rclpy.ok() and time.monotonic() < deadline and worker.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        result["raw_pointcloud2"] = node.raw.metrics()
        result["custom_msg"] = node.custom.metrics() if custom_required else None
        result["clock"] = node.clock_samples.metrics()
        result["measurement_wall_seconds"] = args.duration
        if worker.poll() is not None:
            result["failure"] = f"Isaac worker exited during the measurement window: {worker.returncode}"
            return result

        natural_exit_deadline = time.monotonic() + max(30.0, args.duration)
        while rclpy.ok() and time.monotonic() < natural_exit_deadline and worker.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        if worker.poll() is None:
            result["failure"] = "Isaac worker did not exit after its requested fixed-step run"
            return result
        if worker.returncode != 0:
            result["failure"] = f"Isaac worker exited with {worker.returncode}"
            return result
        result["status"] = "COMPLETE"
        return result
    except Exception as exc:  # noqa: BLE001
        result["failure"] = str(exc)
        result["failure_traceback"] = traceback.format_exc()
        return result
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _stop_process(worker)
        _stop_process(bridge)
        result["worker_returncode"] = worker.returncode if worker else None
        result["bridge_returncode"] = bridge.returncode if bridge else None
        result["isaac_internal"] = _read_json(worker_dir / "metrics.json")
        if result.get("status") == "COMPLETE" and result["isaac_internal"] is None:
            result["status"] = "TEST_ERROR"
            result["failure"] = "Isaac worker did not persist internal metrics.json"


def _fmt(value: Any, digits: int = 3) -> str:
    return "未采集" if value is None else f"{float(value):.{digits}f}"


def _stream_pass(metrics: dict[str, Any] | None) -> bool:
    return bool(
        metrics
        and float(metrics.get("sim_timestamp_hz") or 0.0) >= 9.95
        and int(metrics.get("nonempty_messages") or 0) > 0
        and int(metrics.get("backward_count") or 0) == 0
    )


def _markdown(batch_id: str, executed_at: str, args: argparse.Namespace, result: dict[str, Any]) -> str:
    raw = result.get("raw_pointcloud2") or {}
    custom = result.get("custom_msg") or {}
    internal = result.get("isaac_internal") or {}
    raw_ok = _stream_pass(raw)
    custom_required = not args.without_compat_bridge
    custom_ok = _stream_pass(custom) if custom_required else True
    concurrent = result.get("concurrent_isaac_processes_before_start") or []
    functional = "通过" if result.get("status") == "COMPLETE" and raw_ok and custom_ok else "未通过"
    realtime = "通过" if functional == "通过" and float(internal.get("physics_rtf") or 0.0) >= 1.0 and not concurrent else "未通过"
    return "\n".join(
        (
            f"### 批次：{batch_id}",
            "",
            "| 字段 | 内容 |",
            "| --- | --- |",
            f"| 执行时间 | {executed_at} |",
            f"| 执行者 | {args.executor} |",
            "| 版本 | Isaac Sim 6.0；当前仓库的历史独立 MID360 等效链路 |",
            "| 测试范围 | 静态 `mid360_empty_stage.usda`；不加载机器人、相机、ROS 2 控制或 RViz |",
            f"| 配置 | `{result.get('profile')}`；`MID360_APPROX`；physics/render `{args.physics_hz:g}/{args.render_hz:g} Hz`；headless；`ROS_DOMAIN_ID={args.ros_domain_id}`；兼容桥={'开启' if custom_required else '关闭'} |",
            f"| 并发 Isaac 负载 | {'无' if not concurrent else '有（本批次只作并发诊断，不作独立实时基线）'} |",
            f"| 功能结论 | {functional} |",
            f"| 实时结论 | {realtime} |",
            "",
            "| 指标 | 通过线 | 实测值 | 判定 |",
            "| --- | --- | ---: | --- |",
            f"| raw PointCloud2 仿真时间频率 | ≥`10 Hz` 且非空 | {_fmt(raw.get('sim_timestamp_hz'))} Hz；非空={raw.get('nonempty_messages', '未采集')} | {'通过' if raw_ok else '未通过'} |",
            f"| raw PointCloud2 墙钟频率 | 仅记录 | {_fmt(raw.get('wall_receive_hz'))} Hz | 仅记录 |",
            f"| raw 点数/帧（平均 / 范围） | performance 约`30k`；full 约`52.8k`，随命中率波动 | {_fmt(raw.get('point_count_mean'), 1)} / {raw.get('point_count_min', '未采集')}..{raw.get('point_count_max', '未采集')} | 仅记录 |",
            f"| CustomMsg 仿真时间频率 | {'≥`10 Hz` 且非空' if custom_required else '未启用'} | {_fmt(custom.get('sim_timestamp_hz')) if custom_required else '不适用'} Hz；非空={custom.get('nonempty_messages', '不适用') if custom_required else '不适用'} | {'通过' if custom_ok else '未通过'} |",
            f"| CustomMsg 墙钟频率 | 仅记录 | {_fmt(custom.get('wall_receive_hz')) if custom_required else '不适用'} Hz | 仅记录 |",
            f"| physics wall Hz | 仅记录 | {_fmt(internal.get('physics_wall_hz'))} Hz | 仅记录 |",
            f"| RTF | 独立实时基线≥`1.0` | {_fmt(internal.get('physics_rtf'))} | {'通过' if realtime == '通过' else '未通过'} |",
            "",
            "结论：raw PointCloud2 与 `/livox/lidar` CustomMsg 是不同链路，频率不可互相替代。"
            + ("检测到其他 Isaac 进程，本批次不得作为独立性能基线。" if concurrent else "本批次可与历史独立链路做同口径比较。"),
            "",
            f"原始证据：`raw/{batch_id}/result.json`、`raw/{batch_id}/isaac.launch.log`、`raw/{batch_id}/compat_bridge.log`、`raw/{batch_id}/isaac/metrics.json`。",
            "",
        )
    )


def _append_report(markdown: str) -> None:
    content = REPORT_PATH.read_text(encoding="utf-8")
    if REPORT_MARKER not in content:
        raise RuntimeError(f"report marker is missing: {REPORT_MARKER}")
    REPORT_PATH.write_text(content.replace(REPORT_MARKER, markdown + REPORT_MARKER), encoding="utf-8")


def _batch_id(executor: str) -> str:
    safe = "".join(char if char.isalnum() or char in "_-" else "-" for char in executor).strip("-")
    return f"{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-独立MID360-{safe or 'manual'}"


def _harness(args: argparse.Namespace) -> int:
    if not args.stage.is_file():
        raise FileNotFoundError(f"standalone MID360 stage does not exist: {args.stage}")
    if args.duration <= 0.0 or args.startup_timeout <= 0.0:
        raise ValueError("duration and startup_timeout must be positive")
    if args.physics_hz != 90.0 or args.render_hz != 60.0:
        raise ValueError("历史独立复测固定使用 physics/render 90/60 Hz")
    batch_id = _batch_id(args.executor)
    batch_dir = RAW_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    args.output = batch_dir
    executed_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    result = _system_run(args)
    result["configured"] = {
        "physics_hz": args.physics_hz,
        "render_hz": args.render_hz,
        "warmup_steps": args.warmup_steps,
        "duration_s": args.duration,
        "kit_threads": args.kit_threads,
        "ros_domain_id": args.ros_domain_id,
    }
    (batch_dir / "result.json").write_text(
        json.dumps(
            {"batch_id": batch_id, "executed_at": executed_at, "executor": args.executor, "result": result},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.append_report:
        _append_report(_markdown(batch_id, executed_at, args, result))
    print(batch_dir / "result.json")
    return 0 if result.get("status") == "COMPLETE" else 1


def main() -> int:
    args = _parser().parse_args()
    return _worker(args) if args.worker else _harness(args)


if __name__ == "__main__":
    raise SystemExit(main())
