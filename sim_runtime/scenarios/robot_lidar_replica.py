#!/usr/bin/env python3
"""Run a robot-mounted MID360 in a sensor-only Isaac Sim process.

This process owns only the MID360 render product and its ROS 2 output.  It
does not publish ``/clock`` and does not participate in robot control.  The
split is intentional: a point-cloud RTX product must not share the camera
replica's render/update loop when the camera and lidar rates are measured
independently.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[2]
SENSOR_PACKAGE = ROOT / "ros2_pkgs" / "simulation_bridge" / "sensor_pkg"
if str(SENSOR_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SENSOR_PACKAGE))


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--stage",
    type=Path,
    default=ROOT / "isaac_sim_core" / "assets" / "environments" / "robot_camera_lidar_stage.usda",
)
parser.add_argument("--robot-prim", default="/World/OpenFlex")
parser.add_argument("--snapshot-host", default="127.0.0.1")
parser.add_argument("--snapshot-port", type=int, default=24102)
parser.add_argument("--snapshot-startup-timeout", type=float, default=30.0)
parser.add_argument("--kinematic-sync-hz", type=float, default=30.0)
parser.add_argument(
    "--dynamic-transform-mode",
    choices=("sensor", "environment"),
    default="environment",
    help="move the RTX sensor or equivalently move the lightweight environment by the inverse pose",
)
parser.add_argument("--physics-hz", type=float, default=90.0)
parser.add_argument("--render-hz", type=float, default=60.0)
parser.add_argument("--warmup-steps", type=int, default=90)
parser.add_argument("--kit-threads", type=int, default=16)
parser.add_argument("--lidar-profile", default="MID360_PERFORMANCE")
parser.add_argument("--lidar-transport", choices=("helper", "native"), default="native")
parser.add_argument("--lidar-tick-rate-hz", type=float, default=10.0)
parser.add_argument("--lidar-topic", default="/openflex/livox_frame/lidar")
parser.add_argument("--no-lidar-object-id-map", action="store_true")
parser.add_argument("--publish-with-queue-thread", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()

if not args.stage.is_file():
    parser.error(f"stage does not exist: {args.stage}")
if args.physics_hz <= 0.0 or args.render_hz <= 0.0 or args.lidar_tick_rate_hz <= 0.0:
    parser.error("physics, render and lidar tick rates must be positive")
if not 1 <= args.snapshot_port <= 65535:
    parser.error("snapshot port must be between 1 and 65535")
if args.snapshot_startup_timeout <= 0.0 or args.kinematic_sync_hz <= 0.0:
    parser.error("snapshot timeout and kinematic sync rate must be positive")

kit_extra_args = [
    "--/app/runLoops/main/rateLimitEnabled=false",
    "--/app/runLoops/main/manualModeEnabled=true",
    "--/rtx/hydra/supportMultiTickRate=true",
    # RTX lidar with multi-tick rendering needs Motion BVH to be selected
    # before SimulationApp creates the Hydra engine.  Setting only the base
    # flag after startup is too late: the engine stays non-motion-aware,
    # produces the Isaac warning and can drop native PointCloud2 callbacks.
    # Keep the engine list aligned with Isaac Sim's RTX/ROS2 extension test
    # configuration, as this worker may create more than one render product.
    "--/renderer/raytracingMotion/enabled=true",
    "--/renderer/raytracingMotion/enableHydraEngineMasking=true",
    "--/renderer/raytracingMotion/enabledForHydraEngines=0,1,2,3",
    "--/renderer/multiGpu/enabled=false",
    "--/exts/omni.replicator.srtx/enabled=false",
    "--/exts/isaacsim.ros2.bridge/publish_with_queue_thread="
    + ("true" if args.publish_with_queue_thread else "false"),
]
simulation_app = SimulationApp({
    "headless": True,
    "disable_viewport_updates": True,
    "extra_args": kit_extra_args,
    "limit_cpu_threads": args.kit_threads,
})

import carb.settings
import omni.kit.app
import omni.timeline
import omni.usd
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager
from isaacsim.core.prims import XFormPrim
import numpy as np

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
for _ in range(4):
    omni.kit.app.get_app().update()
carb.settings.get_settings().set("/exts/omni.replicator.srtx/enabled", False)
carb.settings.get_settings().set("/renderer/multiGpu/enabled", False)

from isaacsim_sensors.integration import create_robot_sensor_suite
from isaacsim_sensors.mid360 import create_standalone_mid360


def _pace_until(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 1_000_000:
            time.sleep((remaining_ns - 500_000) / 1e9)
        else:
            time.sleep(0)


def _drain_snapshot(receiver: socket.socket) -> dict[str, object] | None:
    latest = None
    while True:
        try:
            payload, _source = receiver.recvfrom(65535)
        except BlockingIOError:
            return latest
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict) and parsed.get("schema_version") == 1:
            latest = parsed


def _wait_for_snapshot(receiver: socket.socket) -> dict[str, object]:
    deadline = time.monotonic() + args.snapshot_startup_timeout
    while time.monotonic() < deadline:
        snapshot = _drain_snapshot(receiver)
        if snapshot is not None and isinstance(snapshot.get("odom"), dict):
            return snapshot
        time.sleep(0.005)
    raise RuntimeError("timed out waiting for authority odometry snapshot")


def _quat_multiply_wxyz(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _rotate_vector_wxyz(quaternion: list[float], vector: list[float]) -> list[float]:
    conjugate = [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]]
    rotated = _quat_multiply_wxyz(
        _quat_multiply_wxyz(quaternion, [0.0, *vector]), conjugate
    )
    return rotated[1:]


def _apply_standalone_pose(
    snapshot: dict[str, object],
    transform_root: XFormPrim,
    mount_translation: list[float],
    mount_orientation_wxyz: list[float],
    previous_position: list[float] | None = None,
    previous_orientation: list[float] | None = None,
    transform_mode: str = "sensor",
) -> tuple[list[float], list[float], bool] | None:
    odom = snapshot.get("odom")
    if not isinstance(odom, dict):
        return None
    position = odom.get("position") or []
    orientation_xyzw = odom.get("orientation_xyzw") or []
    if len(position) != 3 or len(orientation_xyzw) != 4:
        return None
    base_orientation = [
        float(orientation_xyzw[3]),
        float(orientation_xyzw[0]),
        float(orientation_xyzw[1]),
        float(orientation_xyzw[2]),
    ]
    norm = math.sqrt(sum(value * value for value in base_orientation))
    if norm <= 1e-12:
        return None
    base_orientation = [value / norm for value in base_orientation]
    offset = _rotate_vector_wxyz(base_orientation, mount_translation)
    world_position = [
        float(position[0]) + offset[0],
        float(position[1]) + offset[1],
        0.25 + offset[2],
    ]
    world_orientation = _quat_multiply_wxyz(base_orientation, mount_orientation_wxyz)
    unchanged = (
        previous_position is not None
        and previous_orientation is not None
        and math.dist(previous_position, world_position) < 1.0e-5
        and min(
            math.dist(previous_orientation, world_orientation),
            math.dist(previous_orientation, [-value for value in world_orientation]),
        ) < 1.0e-6
    )
    if unchanged:
        return world_position, world_orientation, False
    if transform_mode == "environment":
        applied_orientation = [
            world_orientation[0],
            -world_orientation[1],
            -world_orientation[2],
            -world_orientation[3],
        ]
        applied_position = _rotate_vector_wxyz(
            applied_orientation, [-value for value in world_position]
        )
    else:
        applied_position = world_position
        applied_orientation = world_orientation
    transform_root.set_world_poses(
        positions=np.asarray([applied_position], dtype=np.float32),
        orientations=np.asarray([applied_orientation], dtype=np.float32),
    )
    return world_position, world_orientation, True


def main() -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind((args.snapshot_host, args.snapshot_port))
    receiver.setblocking(False)
    initial_snapshot = _wait_for_snapshot(receiver)
    stage_utils.open_stage(str(args.stage.resolve()))
    app = omni.kit.app.get_app()
    for _ in range(5):
        app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None or not stage.GetPrimAtPath("/World").IsValid():
        raise RuntimeError("loaded stage does not contain /World")

    robot_present = stage.GetPrimAtPath(args.robot_prim).IsValid()
    standalone_transform_root: XFormPrim | None = None
    mount_translation = [0.30, 0.0, 0.12]
    mount_orientation_wxyz = [0.6830127, -0.1830127, 0.1830127, -0.6830127]
    if robot_present:
        suite = create_robot_sensor_suite(
            stage,
            args.robot_prim,
            realsense_asset_dir=ROOT,
            mid360_asset_dir=ROOT,
            sensor_profile="lidar",
            lidar_profile=args.lidar_profile,
            lidar_transport=args.lidar_transport,
            lidar_object_id_map=not args.no_lidar_object_id_map,
            lidar_tick_rate_hz=args.lidar_tick_rate_hz,
            publish_camera_info=False,
        )
        if int(suite.get("camera_count", 0)) != 0 or not suite.get("lidar_prim_path"):
            raise RuntimeError(f"robot-mounted MID360 setup failed: {suite}")
    else:
        # The split performance process has no robot-state channel. Loading a
        # full robot here would therefore add RTX/USD cost without providing
        # dynamic mounting. Use the historical standalone scene instead; the
        # camera process remains the dynamic robot visualizer.
        if args.lidar_transport != "native":
            raise RuntimeError("standalone split MID360 requires the native/direct-compatible transport")
        created = create_standalone_mid360(
            stage,
            profile=args.lidar_profile,
            topic=args.lidar_topic,
            transport="direct",
            tick_rate_hz=args.lidar_tick_rate_hz,
            kinematic_root=False,
            create_imu=False,
        )
        suite = {
            "success": True,
            "created": list(created.values()),
            "camera_count": 0,
            "lidar_prim_path": created.get("lidar_path", ""),
            "sensor_profile": "lidar",
            "lidar_profile": args.lidar_profile,
            "lidar_transport": "direct",
            "standalone_stage": True,
            "lidar_tick_rate_hz": float(args.lidar_tick_rate_hz),
        }
        transform_path = (
            "/World/MID360TestEnvironment"
            if args.dynamic_transform_mode == "environment"
            else "/World/MID360"
        )
        standalone_transform_root = XFormPrim(transform_path, name="dynamic_mid360_transform_root")
        initial_pose = _apply_standalone_pose(
            initial_snapshot,
            standalone_transform_root,
            mount_translation,
            mount_orientation_wxyz,
            transform_mode=args.dynamic_transform_mode,
        )
        if initial_pose is None:
            raise RuntimeError("initial authority snapshot did not contain a usable odometry pose")

    RenderingManager.set_dt(1.0 / args.render_hz)

    dt = 1.0 / args.physics_hz
    # Match the historical standalone MID360 driver exactly.  The RTX lidar
    # helper is scheduled from Kit's app update loop; driving it through a
    # separate SimulationContext can produce a nominal 90 Hz physics stream
    # while dropping one in ten ROS point-cloud callbacks.
    SimulationManager.setup_simulation(dt=dt, device="cpu")
    physics_steps = 0
    physics_wall_times: list[int] = []

    def _on_physics_step(_step_dt: float, _context: object | None = None) -> None:
        nonlocal physics_steps
        physics_steps += 1
        physics_wall_times.append(time.perf_counter_ns())

    callback_id = SimulationManager.register_callback(
        _on_physics_step, IsaacEvents.POST_PHYSICS_STEP
    )
    timeline = omni.timeline.get_timeline_interface()
    initial_stamp_ns = int(initial_snapshot.get("joint_stamp_ns") or 0)
    if initial_stamp_ns > 0:
        timeline.set_current_time(initial_stamp_ns / 1e9)
    timeline.play()
    for _ in range(3):
        app.update()

    stopping = False

    def _request_stop(_signal_number: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    wall_times: list[int] = []
    update_times: list[int] = []
    snapshot_ages_s: list[float] = []
    snapshot_messages_received = 0
    snapshot_apply_updates = 0
    first_snapshot_joint_stamp_ns: int | None = None
    latest_snapshot_joint_stamp_ns: int | None = None
    first_sensor_position: list[float] | None = None
    latest_sensor_position: list[float] | None = None
    latest_sensor_orientation: list[float] | None = None
    latest_snapshot: dict[str, object] = initial_snapshot
    next_pose_apply_wall_s = 0.0
    try:
        for _ in range(args.warmup_steps):
            app.update()
        (args.output / "ready.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ready",
                    "sensor_profile": "lidar",
                    "ready_wall_ns": time.monotonic_ns(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        physics_wall_times.clear()
        anchor_ns = time.perf_counter_ns()
        measured_start_step = physics_steps
        step = 0
        while simulation_app.is_running() and not stopping:
            snapshot = _drain_snapshot(receiver)
            if snapshot is not None:
                latest_snapshot = snapshot
                snapshot_messages_received += 1
                stamp_ns = int(snapshot.get("joint_stamp_ns") or 0)
                if stamp_ns > 0:
                    if first_snapshot_joint_stamp_ns is None:
                        first_snapshot_joint_stamp_ns = stamp_ns
                    latest_snapshot_joint_stamp_ns = stamp_ns
                relay_wall_ns = int(snapshot.get("relay_wall_ns") or 0)
                if relay_wall_ns > 0:
                    snapshot_ages_s.append(
                        max(0.0, (time.monotonic_ns() - relay_wall_ns) / 1e9)
                    )
            now_s = time.monotonic()
            if standalone_transform_root is not None and now_s >= next_pose_apply_wall_s:
                pose = _apply_standalone_pose(
                    latest_snapshot,
                    standalone_transform_root,
                    mount_translation,
                    mount_orientation_wxyz,
                    latest_sensor_position,
                    latest_sensor_orientation,
                    args.dynamic_transform_mode,
                )
                if pose is not None:
                    latest_sensor_position, latest_sensor_orientation, pose_applied = pose
                    if first_sensor_position is None:
                        first_sensor_position = list(latest_sensor_position)
                    if pose_applied:
                        snapshot_apply_updates += 1
                next_pose_apply_wall_s = now_s + 1.0 / args.kinematic_sync_hz
            begin_ns = time.perf_counter_ns()
            app.update()
            update_times.append(time.perf_counter_ns() - begin_ns)
            if physics_steps > measured_start_step:
                wall_times.append(time.perf_counter_ns())
                step = physics_steps - measured_start_step
                _pace_until(anchor_ns + int(round(step * dt * 1e9)))
    finally:
        receiver.close()
        timeline.stop()
        SimulationManager.deregister_callback(callback_id)
        gaps_s = [(later - earlier) / 1e9 for earlier, later in zip(wall_times, wall_times[1:])]
        wall_duration_s = (wall_times[-1] - wall_times[0]) / 1e9 if len(wall_times) >= 2 else None
        sim_duration_s = (len(wall_times) - 1) / args.physics_hz if len(wall_times) >= 2 else None
        metrics = {
            "schema_version": 1,
            "status": "complete",
            "role": "robot_mounted_lidar_sensor_replica",
            "physics_hz_configured": args.physics_hz,
            "rendering_hz_configured": args.render_hz,
            "physics_steps": len(wall_times),
            "physics_wall_hz": ((len(wall_times) - 1) / wall_duration_s if wall_duration_s else None),
            "physics_rtf": (sim_duration_s / wall_duration_s if sim_duration_s and wall_duration_s else None),
            "physics_wall_gap_p99_s": _percentile(gaps_s, 0.99),
            "physics_wall_gap_max_s": max(gaps_s) if gaps_s else None,
            "app_update_wall_p99_s": _percentile([value / 1e9 for value in update_times], 0.99),
            "app_update_wall_max_s": max(update_times) / 1e9 if update_times else None,
            "snapshot_messages_received": snapshot_messages_received,
            "snapshot_apply_updates": snapshot_apply_updates,
            "snapshot_age_p99_s": _percentile(snapshot_ages_s, 0.99),
            "snapshot_age_max_s": max(snapshot_ages_s) if snapshot_ages_s else None,
            "first_snapshot_joint_stamp_ns": first_snapshot_joint_stamp_ns,
            "latest_snapshot_joint_stamp_ns": latest_snapshot_joint_stamp_ns,
            "first_sensor_position": first_sensor_position,
            "latest_sensor_position": latest_sensor_position,
            "sensor_displacement_m": (
                math.dist(first_sensor_position, latest_sensor_position)
                if first_sensor_position is not None and latest_sensor_position is not None
                else None
            ),
            "kinematic_sync_hz_configured": args.kinematic_sync_hz,
            "dynamic_transform_mode": args.dynamic_transform_mode,
            "sensor_suite": suite,
            "kit_runtime": {
                "rate_limit_enabled": carb.settings.get_settings().get_as_bool("/app/runLoops/main/rateLimitEnabled"),
                "manual_mode_enabled": carb.settings.get_settings().get_as_bool("/app/runLoops/main/manualModeEnabled"),
                "tasking_thread_count": carb.settings.get_settings().get_as_int("/plugins/carb.tasking.plugin/threadCount"),
                "tbb_max_thread_count": carb.settings.get_settings().get_as_int("/plugins/omni.tbb.globalcontrol/maxThreadCount"),
                "rtx_multi_tick_enabled": carb.settings.get_settings().get_as_bool("/rtx/hydra/supportMultiTickRate"),
                "motion_bvh_enabled": carb.settings.get_settings().get_as_bool(
                    "/renderer/raytracingMotion/enabled"
                ),
                "motion_bvh_hydra_engine_masking_enabled": carb.settings.get_settings().get_as_bool(
                    "/renderer/raytracingMotion/enableHydraEngineMasking"
                ),
                "motion_bvh_hydra_engines": carb.settings.get_settings().get(
                    "/renderer/raytracingMotion/enabledForHydraEngines"
                ),
                "ros2_publish_with_queue_thread": carb.settings.get_settings().get_as_bool(
                    "/exts/isaacsim.ros2.bridge/publish_with_queue_thread"
                ),
            },
        }
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
