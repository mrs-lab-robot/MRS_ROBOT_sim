#!/usr/bin/env python3
"""Run D435/D435i and D405 without loading a robot."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import json
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "config" / "realsense_standalone.yaml")
parser.add_argument(
    "--camera",
    choices=("none", "d435", "d435i", "d405", "both", "quad"),
    default="both",
    help="Camera group; quad creates two D435 and two D405 rigs from the quad config",
)
parser.add_argument("--physics-hz", type=float, default=None)
parser.add_argument(
    "--render-hz",
    type=float,
    default=None,
    help="Rendering cadence; can be lower than physics-hz for 30 Hz cameras",
)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument(
    "--warmup-steps",
    type=int,
    default=90,
    help="Unmeasured updates used to compile/warm RTX and ROS2 pipelines before online timing",
)
parser.add_argument(
    "--pace-spin-us",
    type=float,
    default=250.0,
    help="Busy-wait budget at the end of each paced period; 0 keeps sleep/yield-only pacing",
)
parser.add_argument(
    "--pace-sleep-guard-us",
    type=float,
    default=3000.0,
    help="Time reserved before each online deadline for scheduler wake-up and final spinning",
)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument(
    "--resolution-scale",
    type=float,
    default=1.0,
    help="Scale configured camera resolution for a rendering-cost A/B test",
)
parser.add_argument("--ros2", action="store_true", help="Attach ROS2 bridge helpers to existing RenderProducts")
parser.add_argument(
    "--local-capture",
    action="store_true",
    help="Also read RGB-D with local annotators; disabled by default for ROS2-only online runs",
)
parser.add_argument(
    "--gpu-dynamics",
    action="store_true",
    help="Run PhysX on CUDA and enable PhysX GPU Dynamics",
)
parser.add_argument(
    "--pacing",
    choices=("paced_online", "unpaced_capacity"),
    default=None,
    help="Use real-time loop pacing or run as fast as possible for capacity measurements",
)
parser.add_argument(
    "--no-render-manager-dt",
    action="store_true",
    help="A/B test: do not force RenderingManager to the physics dt",
)
parser.add_argument("--no-ros2-tf", action="store_true", help="A/B test without the standalone TF graph")
parser.add_argument(
    "--no-ros2-camera-info",
    action="store_true",
    help="A/B test without CameraInfo helpers; RGB/depth remain enabled",
)
parser.add_argument("--no-ros2-rgb", action="store_true", help="A/B test without RGB image topics")
parser.add_argument("--no-ros2-depth", action="store_true", help="A/B test without depth image topics")
parser.add_argument(
    "--ros2-rgb-h264",
    action="store_true",
    help="Publish RGB as sensor_msgs/CompressedImage via the official H.264 helper",
)
parser.add_argument(
    "--ros2-exec-gate",
    action="store_true",
    help="A/B test with outer 30 Hz execution gates (off by default)",
)
parser.add_argument(
    "--ros2-queue-size",
    type=int,
    default=None,
    help="Override the ROS2 helper queue depth for an output-pressure A/B test",
)
parser.add_argument(
    "--ros2-qos-profile",
    choices=("sensor_data", "reliable"),
    default="reliable",
    help="ROS2 image QoS; reliable is an A/B option for loss-sensitive recording",
)
parser.add_argument(
    "--ros2-publish-queue-thread",
    choices=("on", "off"),
    default="off",
    help="Select Isaac Sim ROS2 image publisher scheduling; default avoids the shared global image queue",
)
parser.add_argument(
    "--ros2-publish-queue-sleep-us",
    type=int,
    default=None,
    help="Override the Isaac Sim ROS2 image queue thread sleep interval",
)
parser.add_argument(
    "--ros2-publish-multithreading",
    choices=("on", "off"),
    default=None,
    help="Enable or disable per-image ROS2 bridge tasking; off uses the zero-delay synchronous path",
)
parser.add_argument(
    "--async-rendering",
    action="store_true",
    help="Enable Kit async rendering for an online scheduling A/B test",
)
parser.add_argument(
    "--minimal-rendering",
    action="store_true",
    help="A/B test with MinimalRendering textured diffuse mode",
)
parser.add_argument(
    "--renderer",
    choices=("RealTimePathTracing", "RaytracedLighting", "MinimalRendering"),
    default=None,
    help="Kit renderer mode for a rendering-cost A/B test",
)
parser.add_argument(
    "--anti-aliasing",
    type=int,
    choices=(0, 1, 2, 3, 4),
    default=None,
    help="RTX anti-aliasing mode: 0 off, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA",
)
parser.add_argument(
    "--srtx",
    action="store_true",
    help="Use Isaac Sim SRTX callbacks for ROS2 image transport",
)
parser.add_argument(
    "--no-per-sensor-tick-tlas",
    action="store_true",
    help="A/B test: disable per-sensor TLAS rebuilds used by RTX multi-tick rendering",
)
parser.add_argument(
    "--kit-threads",
    type=int,
    default=16,
    help="Kit tasking/TBB worker count for a reproducible performance A/B test",
)
parser.add_argument("--output", type=Path, default=Path("reports/realsense/raw/standalone_smoke"))
args, _ = parser.parse_known_args()

if args.kit_threads < 1:
    parser.error("--kit-threads must be positive")
if args.warmup_steps < 0:
    parser.error("--warmup-steps must be non-negative")
if args.pace_spin_us < 0.0 or args.pace_spin_us > 5000.0:
    parser.error("--pace-spin-us must be between 0 and 5000")
if args.pace_sleep_guard_us < 0.0 or args.pace_sleep_guard_us > 5000.0:
    parser.error("--pace-sleep-guard-us must be between 0 and 5000")
if args.ros2_queue_size is not None and args.ros2_queue_size < 1:
    parser.error("--ros2-queue-size must be positive")
if args.ros2_publish_queue_sleep_us is not None and args.ros2_publish_queue_sleep_us < 0:
    parser.error("--ros2-publish-queue-sleep-us must be non-negative")


def _pace_until(target_wall_ns: int, spin_us: float, sleep_guard_us: float) -> None:
    """Pace an online loop without adding a scheduler-sized period error."""

    spin_ns = int(round(spin_us * 1000.0))
    sleep_guard_ns = int(round(sleep_guard_us * 1000.0))
    while True:
        remaining_ns = target_wall_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > spin_ns + sleep_guard_ns:
            time.sleep((remaining_ns - spin_ns - sleep_guard_ns) / 1e9)
        elif remaining_ns > spin_ns:
            # Give the OS a chance to schedule ROS/DDS workers while retaining
            # a bounded final interval for the monotonic deadline.
            time.sleep(0)
        # The final interval intentionally spins.  This is only used by the
        # online profile and avoids time.sleep() overshooting every physics
        # deadline on a loaded desktop kernel.

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"PyYAML is required: {exc}") from exc

from isaacsim import SimulationApp

kit_extra_args = [
    "--/app/runLoops/main/rateLimitEnabled=false",
    "--/app/runLoops/main/manualModeEnabled=true",
    # Camera prim tick rates are honored by the RTX multi-tick renderer. Keep
    # this explicit because extension defaults can differ between 6.0 builds.
    "--/rtx/hydra/supportMultiTickRate=true",
]
if args.ros2_publish_queue_thread is not None:
    kit_extra_args.append(
        "--/exts/isaacsim.ros2.bridge/publish_with_queue_thread="
        + ("true" if args.ros2_publish_queue_thread == "on" else "false")
    )
if args.ros2_publish_queue_sleep_us is not None:
    kit_extra_args.append(
        f"--/exts/isaacsim.ros2.bridge/publish_queue_thread_sleep_us={args.ros2_publish_queue_sleep_us}"
    )
if args.ros2_publish_multithreading is not None:
    kit_extra_args.append(
        "--/exts/isaacsim.ros2.bridge/publish_multithreading_disabled="
        + ("false" if args.ros2_publish_multithreading == "on" else "true")
    )
if args.no_per_sensor_tick_tlas:
    kit_extra_args.append("--/rtx/rendering/perSensorTickTlas=false")
if args.srtx:
    kit_extra_args.append("--/exts/omni.replicator.srtx/enabled=true")
if args.async_rendering:
    kit_extra_args.extend([
        "--/app/asyncRendering=true",
        "--/app/asyncRenderingLowLatency=true",
    ])
if args.minimal_rendering:
    kit_extra_args.extend([
        "--/rtx/rendermode=MinimalRendering",
        "--/rtx/minimal/mode=2",
        "--/rtx/post/aa/op=0",
    ])

renderer_config = {"renderer": args.renderer} if args.renderer else {}
if args.anti_aliasing is not None:
    renderer_config["anti_aliasing"] = args.anti_aliasing

simulation_app = SimulationApp({
    "headless": args.headless,
    # The standalone performance profile has no viewport consumer. Keeping
    # its updates disabled avoids measuring an unrelated default viewport.
    "disable_viewport_updates": bool(args.headless),
    # Do not let Kit's application loop cap the measurement.  The online
    # profile is paced explicitly below; capacity runs must expose headroom.
    # Keep manual mode enabled so RenderingManager.set_dt() owns the fixed
    # simulation/render timestep instead of the wall-clock app limiter.
    "extra_args": kit_extra_args,
    # SimulationApp appends the tasking/TBB settings after extra_args. Use its
    # supported limit so the requested value is the final value on the Kit
    # command line and is also applied to PXR/OpenBLAS worker pools.
    "limit_cpu_threads": args.kit_threads,
    **renderer_config,
})

import numpy as np
import carb.settings
import omni.timeline
import omni.usd
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

from openflex_isaac_sensors.frame_packet import FramePacket
from openflex_isaac_sensors.integration import attach_realsense_calibrations
from openflex_isaac_sensors.rig import RealSenseRig
from openflex_isaac_sensors.sinks import AsyncJsonlSink, IsaacSimRos2Bridge
from openflex_isaac_sensors.transport import BoundedFrameQueue


def _kit_runtime_settings() -> dict[str, object]:
    """Return the Kit loop settings that affect frequency measurements."""

    settings = carb.settings.get_settings()
    return {
        "rate_limit_enabled": settings.get_as_bool("/app/runLoops/main/rateLimitEnabled"),
        "rate_limit_frequency_hz": settings.get_as_float("/app/runLoops/main/rateLimitFrequency"),
        "manual_mode_enabled": settings.get_as_bool("/app/runLoops/main/manualModeEnabled"),
        "fixed_time_stepping": settings.get_as_bool("/app/player/useFixedTimeStepping"),
        "fast_mode": settings.get_as_bool("/app/player/useFastMode"),
        "tasking_thread_count": settings.get_as_int("/plugins/carb.tasking.plugin/threadCount"),
        "tbb_max_thread_count": settings.get_as_int("/plugins/omni.tbb.globalcontrol/maxThreadCount"),
        "srtx_enabled": settings.get_as_bool("/exts/omni.replicator.srtx/enabled"),
        "rtx_multi_tick_enabled": settings.get_as_bool("/rtx/hydra/supportMultiTickRate"),
        "rtx_per_sensor_tick_tlas": settings.get_as_bool("/rtx/rendering/perSensorTickTlas"),
        "renderer_multi_gpu_enabled": settings.get_as_bool("/renderer/multiGpu/enabled"),
        "ros2_publish_with_queue_thread": settings.get_as_bool(
            "/exts/isaacsim.ros2.bridge/publish_with_queue_thread"
        ),
        "ros2_publish_queue_thread_sleep_us": settings.get_as_int(
            "/exts/isaacsim.ros2.bridge/publish_queue_thread_sleep_us"
        ),
        "ros2_publish_multithreading_disabled": settings.get_as_bool(
            "/exts/isaacsim.ros2.bridge/publish_multithreading_disabled"
        ),
    }


def _load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _selected_names(selection: str) -> list[str]:
    if selection == "none":
        return []
    if selection == "both":
        return ["d435_standalone", "d405_standalone"]
    if selection == "quad":
        return [
            "d435_front_standalone",
            "d435_rear_standalone",
            "d405_left_standalone",
            "d405_right_standalone",
        ]
    return [f"{selection}_standalone"]


def main() -> int:
    config = _load_config(args.config)
    attach_realsense_calibrations(config, args.config.expanduser().resolve().parent / "calibration")
    runtime = config.get("runtime", {}) or {}
    physics_hz = float(args.physics_hz or runtime.get("physics_hz", 90.0))
    dt = 1.0 / physics_hz
    render_hz = float(args.render_hz or runtime.get("render_hz", physics_hz))
    pacing_mode = str(args.pacing or runtime.get("pacing", "paced_online"))
    local_capture_enabled = not args.ros2 or args.local_capture
    if physics_hz <= 0.0 or render_hz <= 0.0:
        raise ValueError("physics and render frequencies must be positive")
    if args.resolution_scale <= 0.0 or args.resolution_scale > 1.0:
        raise ValueError("--resolution-scale must be in (0, 1]")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    stage_utils.set_stage_units(meters_per_unit=1.0)
    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/StandaloneSensors", "Xform")
    GroundPlane("/World/StandaloneGround", sizes=20.0)
    DistantLight("/World/StandaloneLight").set_intensities(1200)
    Cube("/World/StandaloneTarget", sizes=0.5, positions=np.array([0.0, 0.0, 0.25]))

    # Keep the online profile at one app/physics tick. Set this before creating
    # RenderProducts so Fabric and the rendering loop share the same dt.
    if pacing_mode == "paced_online" and not args.no_render_manager_dt:
        RenderingManager.set_dt(1.0 / render_hz)
    # Record the final values after RenderingManager has applied its fixed
    # timestep.  The explicit SimulationApp extra_args above still establish
    # the startup policy before any scene or sensor is created.
    kit_settings = _kit_runtime_settings()

    selected = set(_selected_names(args.camera))
    rigs: list[RealSenseRig] = []
    sinks: dict[str, AsyncJsonlSink] = {}
    ros_bridges: dict[str, IsaacSimRos2Bridge] = {}
    shared_ros_bridge = IsaacSimRos2Bridge("/World/StandaloneSensors/ROS2") if args.ros2 else None
    queues: dict[str, BoundedFrameQueue[FramePacket]] = {}
    for name, camera_config in (config.get("cameras", {}) or {}).items():
        if name not in selected:
            continue
        if not isinstance(camera_config, dict):
            raise ValueError(f"camera config must be a mapping: {name}")
        camera_config = dict(camera_config)
        if args.resolution_scale != 1.0:
            camera_config["width"] = max(1, round(int(camera_config.get("width", 640)) * args.resolution_scale))
            camera_config["height"] = max(1, round(int(camera_config.get("height", 480)) * args.resolution_scale))
        camera_config["quality_level"] = runtime.get("quality_level", "L0_aligned_fast")
        camera_config["enabled"] = True
        rig = RealSenseRig(name, camera_config)
        rig.create(
            stage,
            str(camera_config["parent_prim"]),
            camera_config.get("local_pose"),
            camera_config,
            create_render_product=not args.ros2,
        )
        if args.ros2:
            rig.create_python_render_product(
                publish_rgb=not args.no_ros2_rgb,
                publish_depth=not args.no_ros2_depth,
            )
        rigs.append(rig)
        if local_capture_enabled:
            sinks[name] = AsyncJsonlSink(
                output / f"{name}.jsonl",
                maxsize=int((config.get("queues", {}) or {}).get("output_size_per_camera", 8)),
            )
            queues[name] = BoundedFrameQueue(
                int((config.get("queues", {}) or {}).get("latest_size_per_camera", 2))
            )
        if args.ros2:
            ros_bridges[name] = shared_ros_bridge

    physics_device = "cuda:0" if args.gpu_dynamics else "cpu"
    SimulationManager.setup_simulation(dt=dt, device=physics_device)
    physics_gpu_dynamics_enabled = SimulationManager.is_gpu_dynamics_enabled()
    physics_device_actual = str(SimulationManager.get_device())
    physics_step = 0
    app_updates = 0
    app_update_wall_times_ns: list[int] = []
    physics_steps_per_update: list[int] = []
    physics_sim_times: list[float] = []
    physics_wall_times_ns: list[int] = []

    def on_physics_step(step_dt: float, context: object | None = None) -> None:
        nonlocal physics_step
        physics_step += 1
        sim_time = physics_step * dt
        physics_sim_times.append(sim_time)
        physics_wall_times_ns.append(time.perf_counter_ns())
        for rig in rigs:
            rig.record_physics(sim_time)

    callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)
    if args.ros2:
        ros_camera_specs = []
        for rig in rigs:
            ros_camera_specs.append({
                "camera_key": rig.name,
                "render_product_path": rig.render_product_path or "",
                "frame_id": rig.config.optical_frame,
                "node_namespace": rig.name,
                "rgb_topic": "color/image_raw",
                "depth_topic": "depth/image_rect_raw",
                "rgb_type": "rgb_h264" if args.ros2_rgb_h264 else "rgb",
                "camera_info_topic": "color/camera_info",
                "publish_camera_info": not args.no_ros2_camera_info,
                "publish_rgb": not args.no_ros2_rgb,
                "publish_depth": not args.no_ros2_depth,
                "frame_skip_count": 0,
                "camera_prim_path": rig.camera_prim_path or "",
                "tick_rate_hz": rig.config.tick_rate_hz,
                "queue_size": int(
                    args.ros2_queue_size
                    if args.ros2_queue_size is not None
                    else (config.get("queues", {}) or {}).get("ros2_queue_size", 5)
                ),
                "qos_profile": "Custom" if args.ros2_qos_profile == "reliable" else "Sensor Data",
                "exec_gate_step": (
                    max(1, round(physics_hz / rig.config.tick_rate_hz))
                    if args.ros2_exec_gate
                    else 0
                ),
            })
        if ros_camera_specs:
            shared_ros_bridge.attach_cameras(ros_camera_specs)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    if args.ros2:
        bootstrap_steps = max(
            1,
            max((int(spec.get("exec_gate_step", 0)) for spec in ros_camera_specs), default=1),
        )
        for _ in range(bootstrap_steps):
            simulation_app.update()
        shared_ros_bridge.configure_camera_gates(ros_camera_specs)
        # Create static TF after the ROS bridge has completed its bootstrap
        # ticks. The raw TF helper uses RunOneSimulationFrame; creating it
        # before bridge startup could consume that one shot before the ROS
        # publisher/context was ready.
        if not args.no_ros2_tf:
            for rig in rigs:
                shared_ros_bridge.attach_tf_tree(
                    rig.camera_prim_path or "",
                    topic_name="tf_static",
                    node_namespace=rig.name,
                    static_publisher=True,
                    mount_translation=rig.local_pose.translation_m,
                    mount_quaternion_wxyz=rig.local_pose.quaternion_wxyz,
                    mount_frame=f"{rig.name}_mount",
                    camera_frame=f"{rig.name}_camera",
                    optical_frame=rig.config.optical_frame,
                    imu_frame=(rig.config.imu_frame if rig.config.imu_enabled else None),
                )

    # RTX shader compilation and first RenderProduct allocation can block one
    # app update for hundreds of milliseconds. Complete that work before the
    # control/data-collection timing interval, while retaining the number of
    # warm-up steps in the report.
    for _ in range(args.warmup_steps):
        simulation_app.update()

    physics_sim_times.clear()
    physics_wall_times_ns.clear()
    physics_steps_per_update.clear()
    for rig in rigs:
        rig.reset_measurement()
    measured_start_physics_step = physics_step

    # ROS2-only runs let the helper own the RenderProduct pipeline. Local
    # annotator reads are enabled for offline FramePacket capture only because
    # get_data() can synchronize the GPU and duplicate image work.
    if local_capture_enabled:
        for rig in rigs:
            rig.start()

    # RenderingManager.set_dt() selects the simulation/render timestep, but it
    # does not make a standalone loop real-time. Anchor a monotonic wall clock
    # after startup so paced runs have RTF ~= 1 when the workload has headroom;
    # overloaded runs remain observable as RTF < 1 instead of being hidden by
    # catch-up sleeping.
    pacing_anchor_ns = time.perf_counter_ns() - int(round(physics_step * dt * 1e9))

    try:
        target_physics_steps = measured_start_physics_step + args.steps
        while simulation_app.is_running() and physics_step < target_physics_steps:
            previous_physics_step = physics_step
            app_update_start_ns = time.perf_counter_ns()
            simulation_app.update()
            app_update_wall_times_ns.append(time.perf_counter_ns() - app_update_start_ns)
            app_updates += 1
            physics_steps_per_update.append(physics_step - previous_physics_step)
            sim_time = physics_step * dt
            if local_capture_enabled:
                for rig in rigs:
                    for packet in rig.poll_due_frames(sim_time):
                        queues[rig.name].put(packet)
                for name, queue in queues.items():
                    for packet in queue.drain():
                        sinks[name].write(packet)
            if pacing_mode == "paced_online" and physics_step > 0:
                target_wall_ns = pacing_anchor_ns + int(round(physics_step * dt * 1e9))
                _pace_until(target_wall_ns, args.pace_spin_us, args.pace_sleep_guard_us)
    finally:
        timeline.stop()
        SimulationManager.deregister_callback(callback_id)
        for sink in sinks.values():
            sink.close()
        wall_duration_s = (
            (physics_wall_times_ns[-1] - physics_wall_times_ns[0]) / 1e9
            if len(physics_wall_times_ns) >= 2
            else None
        )
        sim_duration_s = (
            physics_sim_times[-1] - physics_sim_times[0]
            if len(physics_sim_times) >= 2
            else None
        )
        measured_physics_steps = len(physics_sim_times)
        physics_wall_gaps_s = [
            (later - earlier) / 1e9
            for earlier, later in zip(physics_wall_times_ns, physics_wall_times_ns[1:])
        ]
        ordered_gaps = sorted(physics_wall_gaps_s)
        p99_gap_s = (
            ordered_gaps[min(len(ordered_gaps) - 1, round((len(ordered_gaps) - 1) * 0.99))]
            if ordered_gaps
            else None
        )
        ordered_update_times_s = sorted(value / 1e9 for value in app_update_wall_times_ns)
        update_p99_s = (
            ordered_update_times_s[
                min(len(ordered_update_times_s) - 1, round((len(ordered_update_times_s) - 1) * 0.99))
            ]
            if ordered_update_times_s
            else None
        )
        metrics = {
            "schema_version": 1,
            "status": "complete",
            "physics_hz_configured": physics_hz,
            "physics_dt_s": dt,
            "physics_device": physics_device_actual,
            "physics_gpu_dynamics_enabled": physics_gpu_dynamics_enabled,
            "pacing_mode": pacing_mode,
            "pace_spin_us": args.pace_spin_us if pacing_mode == "paced_online" else 0.0,
            "pace_sleep_guard_us": args.pace_sleep_guard_us if pacing_mode == "paced_online" else 0.0,
            "warmup_steps": args.warmup_steps,
            "kit_runtime": kit_settings,
            "rendering_hz_configured": render_hz if pacing_mode == "paced_online" else None,
            "render_manager_dt_enabled": pacing_mode == "paced_online" and not args.no_render_manager_dt,
            "async_rendering_requested": args.async_rendering,
            "minimal_rendering_requested": args.minimal_rendering,
            "renderer_requested": args.renderer,
            "anti_aliasing_requested": args.anti_aliasing,
            "srtx_requested": args.srtx,
            "per_sensor_tick_tlas_requested": not args.no_per_sensor_tick_tlas,
            "ros2_tf_enabled": args.ros2 and not args.no_ros2_tf,
            "ros2_camera_info_enabled": args.ros2 and not args.no_ros2_camera_info,
            "ros2_rgb_enabled": args.ros2 and not args.no_ros2_rgb,
            "ros2_depth_enabled": args.ros2 and not args.no_ros2_depth,
            "ros2_rgb_h264": args.ros2 and not args.no_ros2_rgb and args.ros2_rgb_h264,
            "ros2_publish_queue_thread_requested": args.ros2_publish_queue_thread,
            "ros2_publish_queue_sleep_us_requested": args.ros2_publish_queue_sleep_us,
            "ros2_publish_multithreading_requested": args.ros2_publish_multithreading,
            "ros2_qos_profile_requested": args.ros2_qos_profile,
            "ros2_exec_gate_enabled": args.ros2 and args.ros2_exec_gate,
            "local_capture_enabled": local_capture_enabled,
            "physics_steps": measured_physics_steps,
            "app_updates": app_updates,
            "physics_steps_per_app_update_max": max(physics_steps_per_update) if physics_steps_per_update else 0,
            "physics_steps_per_app_update_p99": (
                sorted(physics_steps_per_update)[
                    min(len(physics_steps_per_update) - 1, round((len(physics_steps_per_update) - 1) * 0.99))
                ]
                if physics_steps_per_update
                else 0
            ),
            "resolution_scale": args.resolution_scale,
            "kit_threads_requested": args.kit_threads,
            "physics_sim_duration_s": sim_duration_s,
            "physics_wall_duration_s": wall_duration_s,
            "physics_wall_hz": (
                (measured_physics_steps - 1) / wall_duration_s
                if wall_duration_s and measured_physics_steps >= 2
                else None
            ),
            "physics_rtf": (
                sim_duration_s / wall_duration_s
                if sim_duration_s is not None and wall_duration_s
                else None
            ),
            "physics_wall_gap_p99_s": p99_gap_s,
            "physics_wall_gap_max_s": max(physics_wall_gaps_s) if physics_wall_gaps_s else None,
            "app_update_wall_mean_s": (
                sum(app_update_wall_times_ns) / len(app_update_wall_times_ns) / 1e9
                if app_update_wall_times_ns
                else None
            ),
            "app_update_wall_p99_s": update_p99_s,
            "app_update_wall_max_s": max(app_update_wall_times_ns) / 1e9 if app_update_wall_times_ns else None,
            "cameras": {
                rig.name: {
                    **rig.diagnostics(),
                    "queue": queues[rig.name].stats() if local_capture_enabled else None,
                    "async_output_queue": sinks[rig.name].stats() if local_capture_enabled else None,
                }
                for rig in rigs
            },
            "render_products": [rig.diagnostics().get("render_product") for rig in rigs],
            "render_product_count_expected": len(rigs),
            "ros2_gate_config": shared_ros_bridge.gate_config if shared_ros_bridge else {},
            "ros2_qos_config": shared_ros_bridge.qos_config if shared_ros_bridge else {},
        }
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        with (output / "timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("physics_step", "sim_time_s", "wall_time_ns", "wall_gap_ns"))
            for index, (sim_time, wall_time_ns) in enumerate(zip(physics_sim_times, physics_wall_times_ns), 1):
                previous = physics_wall_times_ns[index - 2] if index >= 2 else ""
                gap_ns = wall_time_ns - previous if previous != "" else ""
                writer.writerow((index, sim_time, wall_time_ns, gap_ns))
        for rig in rigs:
            rig.close()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
