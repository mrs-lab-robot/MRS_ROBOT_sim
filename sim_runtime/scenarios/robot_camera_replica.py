#!/usr/bin/env python3
"""Render a non-authoritative robot camera replica driven by local UDP snapshots.

The process deliberately owns no ROS control graph and publishes no ``/clock``.
Its only outputs are camera topics. Physics and control remain in the separate
authority Isaac process.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SENSOR_PACKAGE = ROOT / "ros2_pkgs" / "simulation_bridge" / "sensor_pkg"
if str(SENSOR_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SENSOR_PACKAGE))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", type=Path, default=ROOT / "isaac_sim_core" / "assets" / "environments" / "robot_control_stage.usda")
parser.add_argument("--sensor-config-root", type=Path, default=ROOT / "isaac_sim_core" / "config" / "sensor_params" / "realsense")
parser.add_argument("--robot-prim", default="/World/OpenFlex")
parser.add_argument("--snapshot-host", default="127.0.0.1")
parser.add_argument("--snapshot-port", type=int, default=24101)
parser.add_argument("--physics-hz", type=float, default=30.0)
parser.add_argument("--render-hz", type=float, default=30.0)
parser.add_argument("--steps", type=int, default=0, help="0 runs until SIGINT/SIGTERM")
parser.add_argument("--warmup-steps", type=int, default=90)
parser.add_argument("--pace-spin-us", type=float, default=750.0)
parser.add_argument("--pace-sleep-guard-us", type=float, default=0.0)
parser.add_argument("--kit-threads", type=int, default=16)
parser.add_argument("--srtx", action="store_true", help="use Isaac Sim native SRTX ROS2 camera transport")
parser.add_argument("--publish-with-queue-thread", action="store_true", help="publish ROS2 camera messages from the bridge queue thread")
parser.add_argument(
    "--minimal-rendering",
    action="store_true",
    help="use the Isaac Sim MinimalRendering textured-diffuse path for this visual-only replica",
)
parser.add_argument(
    "--no-per-sensor-tick-tlas",
    action="store_true",
    help="avoid a separate TLAS rebuild for each camera tick during this visual-only probe",
)
parser.add_argument(
    "--disable-replica-collisions",
    action="store_true",
    help="remove collision schemas from the non-authoritative replica; joints and visuals remain enabled",
)
parser.add_argument(
    "--hide-replica-guide-meshes",
    action="store_true",
    help="hide guide-purpose collision meshes in the visual replica before RTX rendering",
)
parser.add_argument(
    "--visual-lod-proxy",
    action="store_true",
    help="replace direct visual meshes with per-link bounding-box proxies in this experiment replica",
)
parser.add_argument(
    "--visual-kinematic-replica",
    action="store_true",
    help="run the replica as a visual-only USD kinematic chain without PhysX bodies or joints",
)
parser.add_argument(
    "--kinematic-sync-hz",
    type=float,
    default=0.0,
    help=(
        "maximum visual-kinematic state-application rate; 0 applies each fresh "
        "snapshot at the replica step rate"
    ),
)
parser.add_argument(
    "--anti-aliasing",
    type=int,
    choices=(0, 1, 2, 3, 4),
    default=None,
    help="RTX anti-aliasing mode: 0 off, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA",
)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
parser.add_argument("--no-camera-info", action="store_true", help="do not publish per-frame CameraInfo in this capture probe")
parser.add_argument("--camera-tick-rate-hz", type=float, default=30.0, help="simulation-time sample cadence for all replica cameras")
parser.add_argument(
    "--sensor-profile",
    choices=("rgb_depth", "data"),
    default="rgb_depth",
    help="rgb_depth publishes four cameras; data additionally creates the MID360",
)
parser.add_argument(
    "--lidar-profile",
    default="MID360_PERFORMANCE",
    help="MID360 profile used only when --sensor-profile=data",
)
parser.add_argument(
    "--lidar-transport",
    choices=("helper", "native"),
    default="helper",
    help="MID360 PointCloud2 transport used only when --sensor-profile=data",
)
parser.add_argument(
    "--no-lidar-object-id-map",
    action="store_true",
    help="disable the optional MID360 object-id map writer in combined camera + lidar runs",
)
parser.add_argument(
    "--lidar-tick-rate-hz",
    type=float,
    default=10.0,
    help="MID360 simulation-time output tick rate for robot sensor runs",
)
parser.add_argument("--output", type=Path, required=True)
args, _ = parser.parse_known_args()

if args.physics_hz <= 0.0 or args.render_hz <= 0.0:
    parser.error("physics and render frequencies must be positive")
if args.camera_width <= 0 or args.camera_height <= 0:
    parser.error("camera dimensions must be positive")
if args.camera_tick_rate_hz <= 0.0:
    parser.error("camera tick rate must be positive")
if args.kinematic_sync_hz < 0.0:
    parser.error("kinematic sync rate must be non-negative")
if args.lidar_tick_rate_hz <= 0.0:
    parser.error("lidar tick rate must be positive")
if not 1 <= args.snapshot_port <= 65535:
    parser.error("snapshot port must be between 1 and 65535")

from isaacsim import SimulationApp


kit_extra_args = [
    "--/app/runLoops/main/rateLimitEnabled=false",
    "--/app/runLoops/main/manualModeEnabled=true",
    "--/rtx/hydra/supportMultiTickRate=true",
    "--/exts/isaacsim.ros2.bridge/publish_with_queue_thread="
    + ("true" if args.publish_with_queue_thread else "false"),
]
if args.no_per_sensor_tick_tlas:
    kit_extra_args.append("--/rtx/rendering/perSensorTickTlas=false")
if args.minimal_rendering:
    kit_extra_args.extend((
        "--/rtx/rendermode=MinimalRendering",
        "--/rtx/minimal/mode=2",
        "--/rtx/post/aa/op=0",
    ))
renderer_config = {"anti_aliasing": args.anti_aliasing} if args.anti_aliasing is not None else {}
simulation_app = SimulationApp({
    "headless": True,
    "disable_viewport_updates": True,
    "extra_args": kit_extra_args,
    "limit_cpu_threads": args.kit_threads,
    **renderer_config,
})

import numpy as np
import carb.settings
import omni.timeline
import omni.usd
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation, XFormPrim
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim_sensors.integration import create_robot_sensor_suite


def _pace_until(target_ns: int) -> None:
    spin_ns = int(round(args.pace_spin_us * 1000.0))
    guard_ns = int(round(args.pace_sleep_guard_us * 1000.0))
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > spin_ns + guard_ns:
            time.sleep((remaining_ns - spin_ns - guard_ns) / 1e9)
        elif remaining_ns > spin_ns:
            time.sleep(0)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * quantile))]


def _find_articulation_root(stage: object, fallback: str) -> str:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(fallback)
    if root and root.IsValid() and root.HasAPI(UsdPhysics.ArticulationRootAPI):
        return fallback
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"no articulation root found below {fallback}")


def _drain_snapshot(receiver: socket.socket) -> dict[str, Any] | None:
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


def _apply_root_pose(snapshot: dict[str, Any], robot_root: XFormPrim) -> None:
    odom = snapshot.get("odom")
    if not isinstance(odom, dict):
        return
    position = odom.get("position") or []
    orientation = odom.get("orientation_xyzw") or []
    if len(position) != 3 or len(orientation) != 4:
        return
    # The authority spawns its root at z=0.25. ``/odom`` describes planar
    # motion relative to that spawn pose.
    root_position = np.asarray([[float(position[0]), float(position[1]), 0.25]], dtype=np.float32)
    xyzw = [float(value) for value in orientation]
    root_orientation = np.asarray([[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]], dtype=np.float32)
    robot_root.set_world_poses(positions=root_position, orientations=root_orientation)


def _apply_snapshot(
    snapshot: dict[str, Any], articulation: Articulation, dof_indices: dict[str, int], robot_root: XFormPrim
) -> int:
    names = snapshot.get("names") or []
    positions = snapshot.get("positions") or []
    selected = [
        (dof_indices[str(name)], float(position))
        for name, position in zip(names, positions)
        if str(name) in dof_indices
    ]
    if selected:
        indices = np.asarray([item[0] for item in selected], dtype=np.int32)
        values = np.asarray([[item[1] for item in selected]], dtype=np.float32)
        articulation.set_joint_positions(values, joint_indices=indices)
    _apply_root_pose(snapshot, robot_root)
    return len(selected)


def _disable_replica_collisions(stage: object) -> int:
    """Remove collision schemas from the visual-only replica.

    The authority process remains the sole owner of collision and contact
    simulation.  The replica only needs articulated link poses for camera
    rendering, so collision broadphase/narrowphase work is unnecessary here.
    """

    collision_schema_names = {
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "NewtonCollisionAPI",
        "NewtonMeshCollisionAPI",
    }
    removed = 0
    for prim in stage.Traverse():
        for schema_name in tuple(prim.GetAppliedSchemas()):
            if schema_name in collision_schema_names:
                try:
                    prim.RemoveAPI(schema_name)
                    removed += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[replica] could not remove {schema_name} from {prim.GetPath()}: {exc}")
    return removed


def _hide_replica_guide_meshes(stage: object) -> int:
    """Hide authored guide-purpose meshes from a purely visual replica.

    The robot USD uses ``guide`` purpose for collision meshes. These meshes
    are not part of the RGB-D visual contract; making that explicit prevents
    an RTX renderer from retaining them in a replica acceleration structure.
    """
    from pxr import UsdGeom

    hidden = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.GetPurposeAttr().Get() != UsdGeom.Tokens.guide:
            continue
        imageable.MakeInvisible()
        hidden += 1
    return hidden


def _install_visual_lod_proxies(stage: object, specs: list[dict[str, object]]) -> tuple[int, int]:
    """Install simple link-local boxes and hide their high-detail source meshes.

    This is deliberately an opt-in throughput probe.  It keeps the authored
    link hierarchy, joint chain, and camera mounts intact, while replacing
    direct mesh children of kinematic links with coarse boxes.  It must never
    be presented as a high-fidelity sensor-data profile.
    """
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    links = {str(spec["body1"]) for spec in specs}
    links.add(args.robot_prim.rstrip("/") + "/Geometry/base_link")
    hidden_meshes = 0
    proxies = 0
    for link_path in sorted(links):
        link = stage.GetPrimAtPath(link_path)
        if not link or not link.IsValid():
            continue
        for child in link.GetChildren():
            if not child.IsA(UsdGeom.Mesh):
                continue
            imageable = UsdGeom.Imageable(child)
            if imageable.GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
                continue
            bounds = cache.ComputeLocalBound(child).ComputeAlignedRange()
            if bounds.IsEmpty():
                continue
            minimum = bounds.GetMin()
            maximum = bounds.GetMax()
            dimensions = maximum - minimum
            if min(float(dimensions[0]), float(dimensions[1]), float(dimensions[2])) <= 1e-7:
                continue
            center = (minimum + maximum) * 0.5
            name = "ReplicaLod_" + "".join(
                value if value.isalnum() or value == "_" else "_" for value in str(child.GetName())
            )
            cube = UsdGeom.Cube.Define(stage, link_path + "/" + name)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
            cube.AddScaleOp().Set(Gf.Vec3d(
                float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
            ))
            imageable.MakeInvisible()
            hidden_meshes += 1
            proxies += 1
    return hidden_meshes, proxies


def _resolve_robot_path(path: str, robot_prim: str) -> str:
    source_root = "/openarmx_integrated"
    if path == source_root:
        return robot_prim
    if path.startswith(source_root + "/"):
        return robot_prim.rstrip("/") + path[len(source_root):]
    return path


def _matrix_from_joint_frame(position: object, orientation: object) -> object:
    from pxr import Gf

    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(
        Gf.Quatd(
            float(orientation.GetReal()),
            Gf.Vec3d(*[float(value) for value in orientation.GetImaginary()]),
        )
    )
    matrix.SetTranslateOnly(Gf.Vec3d(*[float(value) for value in position]))
    return matrix


def _axis_vector(axis: str) -> object:
    from pxr import Gf

    return {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }.get(axis.upper(), Gf.Vec3d(0.0, 0.0, 1.0))


def _axis_angle_matrix(axis: object, angle_rad: float) -> object:
    """Build a PXR rotation matrix without the slow Gf.Rotation overload."""
    from pxr import Gf

    half = 0.5 * float(angle_rad)
    sine = math.sin(half)
    cosine = math.cos(half)
    quaternion = Gf.Quatd(
        cosine,
        Gf.Vec3d(
            float(axis[0]) * sine,
            float(axis[1]) * sine,
            float(axis[2]) * sine,
        ),
    )
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(quaternion)
    return matrix


def _collect_kinematic_joint_specs(stage: object, robot_prim: str) -> list[dict[str, object]]:
    """Read the authored USD joint frames before disabling replica physics."""

    specs: list[dict[str, object]] = []
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        if not type_name.endswith("Joint") or not type_name.startswith("Physics"):
            continue
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        axis_attr = prim.GetAttribute("physics:axis")
        pos0_attr = prim.GetAttribute("physics:localPos0")
        pos1_attr = prim.GetAttribute("physics:localPos1")
        rot0_attr = prim.GetAttribute("physics:localRot0")
        rot1_attr = prim.GetAttribute("physics:localRot1")
        if not body0 or not body1 or not axis_attr or not pos0_attr or not pos1_attr or not rot0_attr or not rot1_attr:
            continue
        position0 = pos0_attr.Get()
        position1 = pos1_attr.Get()
        orientation0 = rot0_attr.Get()
        orientation1 = rot1_attr.Get()
        if position0 is None or position1 is None or orientation0 is None or orientation1 is None:
            continue
        specs.append({
            "name": str(prim.GetName()),
            # Referenced assets retain authored relationship targets below
            # /openarmx_integrated; resolve them to the composed replica root.
            "body0": _resolve_robot_path(str(body0[0]), robot_prim),
            "body1": _resolve_robot_path(str(body1[0]), robot_prim),
            "parent_frame": _matrix_from_joint_frame(position0, orientation0),
            "child_frame": _matrix_from_joint_frame(position1, orientation1),
            "axis": str(axis_attr.Get() or "Z"),
            "joint_type": type_name,
        })
    return specs


def _joint_type_counts(stage: object) -> dict[str, int]:
    """Return composed joint prim types for replica-stage diagnostics."""
    counts: dict[str, int] = {}
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        if "Joint" in type_name:
            counts[type_name] = counts.get(type_name, 0) + 1
    return dict(sorted(counts.items()))


def _disable_replica_physics(stage: object) -> int:
    """Disable all replica physics prims while retaining the visual USD tree."""

    physics_schema_names = {
        "PhysicsRigidBodyAPI",
        "PhysicsMassAPI",
        "PhysicsCollisionAPI",
        "PhysicsMeshCollisionAPI",
        "PhysicsArticulationRootAPI",
        "PhysxArticulationAPI",
        "PhysxJointAPI",
        "PhysicsDriveAPI:angular",
        "PhysicsDriveAPI:linear",
        "PhysicsJointStateAPI:angular",
        "PhysicsJointStateAPI:linear",
        "IsaacLinkAPI",
        "IsaacJointAPI",
        "NewtonArticulationRootAPI",
        "NewtonCollisionAPI",
        "NewtonMeshCollisionAPI",
        "NewtonMimicAPI",
        "PhysicsMassAPI",
    }
    disabled = 0
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        # SimulationContext still needs an active PhysicsScene in order to
        # advance the timeline.  The replica has no bodies or joints left, so
        # retaining this empty scene is harmless and avoids a silent
        # post-warmup stop of ``SimulationContext.step``.
        if type_name.startswith("Physics") and type_name.endswith("Joint"):
            prim.SetActive(False)
            disabled += 1
        for schema_name in tuple(prim.GetAppliedSchemas()):
            # Joint prims are already inactive; multi-apply drive/state APIs
            # are not required for a visual replica and their authored names
            # are not accepted by RemoveAPI as a single schema identifier in
            # all USD builds.  Skipping them avoids a large warning storm.
            if ":" in schema_name and schema_name.split(":", 1)[0] in {
                "PhysicsDriveAPI",
                "PhysicsJointStateAPI",
            }:
                continue
            if schema_name in physics_schema_names:
                try:
                    prim.RemoveAPI(schema_name)
                    disabled += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[replica] could not remove {schema_name} from {prim.GetPath()}: {exc}")
    return disabled


def _prepare_kinematic_transforms(stage: object, specs: list[dict[str, object]]) -> dict[str, object]:
    """Create one reusable transform op per visual link.

    Clearing and rebuilding an XformOpOrder on every state packet causes a
    full USD change-list/recomposition.  With a live relay this can block the
    first replica update for seconds.  Prepare the operation once and only
    set its value in the hot path.
    """
    from pxr import UsdGeom

    operations: dict[str, object] = {}
    for spec in specs:
        path = str(spec["body1"])
        if path in operations:
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        operations[path] = xformable.AddTransformOp()
    return operations


def _apply_kinematic_snapshot(
    stage: object,
    snapshot: dict[str, Any],
    specs: list[dict[str, object]],
    robot_prim: str,
    transform_ops: dict[str, object],
) -> int:
    from pxr import Gf, Sdf

    names = snapshot.get("names") or []
    positions = snapshot.get("positions") or []
    values = {str(name): float(position) for name, position in zip(names, positions)}
    applied = 0
    pending = list(specs)
    known_links = {robot_prim.rstrip("/") + "/Geometry/base_link"}
    pending_transforms: list[tuple[object, object, str]] = []
    while pending:
        progress = False
        remaining: list[dict[str, object]] = []
        for spec in pending:
            body0 = str(spec["body0"])
            body1 = str(spec["body1"])
            if body0 not in known_links:
                remaining.append(spec)
                continue
            angle = values.get(str(spec["name"]), 0.0)
            if str(spec["joint_type"]) == "PhysicsPrismaticJoint":
                motion = Gf.Matrix4d(1.0)
                motion.SetTranslateOnly(_axis_vector(str(spec["axis"])) * angle)
            else:
                motion = _axis_angle_matrix(_axis_vector(str(spec["axis"])), angle)
            local = spec["parent_frame"] * motion * spec["child_frame"].GetInverse()
            transform_op = transform_ops.get(body1)
            if transform_op is not None:
                pending_transforms.append((transform_op, local, body1))
                known_links.add(body1)
                applied += 1
                progress = True
        if not progress:
            break
        pending = remaining
    # Commit all link changes as one USD change block.  Setting each op
    # individually causes a full composition/render invalidation per link
    # and can stall the first live snapshot for several seconds.
    with Sdf.ChangeBlock():
        for transform_op, local, _body1 in pending_transforms:
            transform_op.Set(local)
    return applied


def _order_kinematic_specs(specs: list[dict[str, object]], robot_prim: str) -> list[dict[str, object]]:
    """Topologically order the USD joint chain once during startup."""

    ordered: list[dict[str, object]] = []
    pending = list(specs)
    known_links = {robot_prim.rstrip("/") + "/Geometry/base_link"}
    while pending:
        progress = False
        remaining: list[dict[str, object]] = []
        for spec in pending:
            if str(spec["body0"]) not in known_links:
                remaining.append(spec)
                continue
            ordered.append(spec)
            known_links.add(str(spec["body1"]))
            progress = True
        if not progress:
            # Keep unsupported/disconnected joints visible in metrics, but do
            # not let them block the connected camera-bearing robot chain.
            break
        pending = remaining
    return ordered


def main() -> int:
    if not args.stage.is_file():
        raise FileNotFoundError(f"replica stage does not exist: {args.stage}")
    config_file = args.sensor_config_root / "realsense_robot_mounts.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"camera configuration does not exist: {config_file}")
    args.output.mkdir(parents=True, exist_ok=True)

    stage_utils.open_stage(str(args.stage))
    for _ in range(3):
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None or not stage.GetPrimAtPath(args.robot_prim).IsValid():
        raise RuntimeError(f"robot replica prim does not exist: {args.robot_prim}")
    composed_joint_type_counts = _joint_type_counts(stage)
    kinematic_specs = (
        _collect_kinematic_joint_specs(stage, args.robot_prim)
        if args.visual_kinematic_replica
        else []
    )
    if args.visual_kinematic_replica:
        kinematic_specs = _order_kinematic_specs(kinematic_specs, args.robot_prim)
    physics_prims_disabled = (
        _disable_replica_physics(stage) if args.visual_kinematic_replica else 0
    )
    kinematic_transform_ops = (
        _prepare_kinematic_transforms(stage, kinematic_specs)
        if args.visual_kinematic_replica
        else {}
    )
    collision_schemas_removed = (
        _disable_replica_collisions(stage)
        if args.disable_replica_collisions and not args.visual_kinematic_replica
        else 0
    )
    guide_meshes_hidden = _hide_replica_guide_meshes(stage) if args.hide_replica_guide_meshes else 0
    lod_meshes_hidden, lod_proxies_created = (
        _install_visual_lod_proxies(stage, kinematic_specs)
        if args.visual_lod_proxy and args.visual_kinematic_replica
        else (0, 0)
    )
    RenderingManager.set_dt(1.0 / args.render_hz)
    suite = create_robot_sensor_suite(
        stage,
        args.robot_prim,
        # The component loader resolves the released config from the
        # repository layout.  Pass the repository root rather than the leaf
        # ``.../realsense`` directory so its compatibility lookup remains
        # valid in both source and installed layouts.
        realsense_asset_dir=ROOT,
        mid360_asset_dir=ROOT,
        sensor_profile=args.sensor_profile,
        lidar_profile=args.lidar_profile,
        lidar_transport=args.lidar_transport,
        lidar_object_id_map=not args.no_lidar_object_id_map,
        lidar_tick_rate_hz=args.lidar_tick_rate_hz,
        srtx_enabled=args.srtx,
        camera_resolution=(args.camera_width, args.camera_height),
        publish_camera_info=not args.no_camera_info,
        camera_tick_rate_hz=args.camera_tick_rate_hz,
    )
    if int(suite.get("camera_count", 0)) != 4:
        raise RuntimeError(f"expected four replica cameras, got: {suite}")

    simulation_context = SimulationContext(physics_dt=1.0 / args.physics_hz)
    simulation_context.initialize_physics()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(3):
        simulation_context.step(render=True)
    articulation_root = None
    articulation = None
    dof_indices: dict[str, int] = {}
    if not args.visual_kinematic_replica:
        articulation_root = _find_articulation_root(stage, args.robot_prim)
        articulation = Articulation(articulation_root)
        articulation.initialize()
        dof_indices = {name: articulation.get_dof_index(name) for name in articulation.dof_names}
    robot_root = XFormPrim(args.robot_prim, name="robot_camera_replica_root")

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    receiver.bind((args.snapshot_host, args.snapshot_port))
    receiver.setblocking(False)
    dt = 1.0 / args.physics_hz
    wall_times: list[int] = []
    app_update_times: list[int] = []
    snapshot_ages_s: list[float] = []
    snapshot_count = 0
    applied_updates = 0
    applied_dof_total = 0
    stale_steps = 0
    first_snapshot_joint_stamp_ns: int | None = None
    latest_snapshot_joint_stamp_ns: int | None = None
    latest_snapshot: dict[str, Any] | None = None
    snapshot_dirty = False
    kinematic_sync_period_s = (
        1.0 / args.kinematic_sync_hz
        if args.visual_kinematic_replica and args.kinematic_sync_hz > 0.0
        else 0.0
    )
    next_kinematic_sync_s = 0.0
    snapshot_rate_limited_steps = 0
    stopping = False

    def _request_stop(_signal_number, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        for _ in range(args.warmup_steps):
            simulation_context.step(render=True)
        # Camera topics can emit initialization frames while the scene is
        # still loading.  Publish an explicit readiness marker only after
        # warmup, so external performance harnesses do not start measuring
        # before this process has entered its real update loop.
        (args.output / "ready.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ready",
                    "warmup_steps": args.warmup_steps,
                    "kinematic_joint_count": len(kinematic_specs),
                    "ready_wall_ns": time.monotonic_ns(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        anchor_ns = time.perf_counter_ns()
        step = 0
        while (
            simulation_app.is_running()
            and not stopping
            and (args.steps <= 0 or step < args.steps)
        ):
            snapshot = _drain_snapshot(receiver)
            if snapshot is not None:
                latest_snapshot = snapshot
                snapshot_dirty = True
                snapshot_count += 1
                stamp_ns = int(snapshot.get("joint_stamp_ns", 0))
                if stamp_ns > 0:
                    if first_snapshot_joint_stamp_ns is None:
                        first_snapshot_joint_stamp_ns = stamp_ns
                    latest_snapshot_joint_stamp_ns = stamp_ns
            begin_ns = time.perf_counter_ns()
            simulation_elapsed_s = step * dt
            state_update_due = (
                not args.visual_kinematic_replica
                or kinematic_sync_period_s <= 0.0
                or simulation_elapsed_s + 1e-12 >= next_kinematic_sync_s
            )
            if latest_snapshot is not None and snapshot_dirty and state_update_due:
                relay_wall_ns = int(latest_snapshot.get("relay_wall_ns", 0))
                if relay_wall_ns > 0:
                    snapshot_ages_s.append(max(0.0, (time.monotonic_ns() - relay_wall_ns) / 1e9))
                if args.visual_kinematic_replica:
                    applied_dof_total += _apply_kinematic_snapshot(
                        stage,
                        latest_snapshot,
                        kinematic_specs,
                        args.robot_prim,
                        kinematic_transform_ops,
                    )
                    # Apply the root odometry pose separately; the kinematic
                    # chain itself is authored below the base link.
                    _apply_root_pose(latest_snapshot, robot_root)
                else:
                    applied_dof_total += _apply_snapshot(
                        latest_snapshot, articulation, dof_indices, robot_root
                    )
                applied_updates += 1
                snapshot_dirty = False
                if kinematic_sync_period_s > 0.0:
                    # Keep a simulation-time schedule rather than resetting it
                    # from wall time. At 40 Hz stepping and 30 Hz cameras this
                    # yields the expected 25/50 ms cadence without drifting to
                    # an accidental 20 Hz cadence.
                    while next_kinematic_sync_s <= simulation_elapsed_s + 1e-12:
                        next_kinematic_sync_s += kinematic_sync_period_s
            elif latest_snapshot is not None and snapshot_dirty:
                snapshot_rate_limited_steps += 1
            else:
                stale_steps += 1
            simulation_context.step(render=True)
            app_update_times.append(time.perf_counter_ns() - begin_ns)
            wall_times.append(time.perf_counter_ns())
            step += 1
            _pace_until(anchor_ns + int(round(step * dt * 1e9)))
    finally:
        receiver.close()
        timeline.stop()
        gaps_s = [(later - earlier) / 1e9 for earlier, later in zip(wall_times, wall_times[1:])]
        wall_duration_s = (wall_times[-1] - wall_times[0]) / 1e9 if len(wall_times) >= 2 else None
        sim_duration_s = (len(wall_times) - 1) * dt if len(wall_times) >= 2 else None
        metrics = {
            "schema_version": 1,
            "status": "complete",
            "role": "non_authoritative_robot_camera_replica",
            "physics_hz_configured": args.physics_hz,
            "rendering_hz_configured": args.render_hz,
            "physics_steps": len(wall_times),
            "physics_wall_hz": ((len(wall_times) - 1) / wall_duration_s if wall_duration_s else None),
            "physics_rtf": (sim_duration_s / wall_duration_s if sim_duration_s is not None and wall_duration_s else None),
            "physics_wall_gap_p99_s": _percentile(gaps_s, 0.99),
            "physics_wall_gap_max_s": max(gaps_s) if gaps_s else None,
            "app_update_wall_p99_s": _percentile([value / 1e9 for value in app_update_times], 0.99),
            "app_update_wall_max_s": max(app_update_times) / 1e9 if app_update_times else None,
            "snapshot_messages_received": snapshot_count,
            "snapshot_apply_updates": applied_updates,
            "snapshot_applied_dof_total": applied_dof_total,
            "snapshot_age_p99_s": _percentile(snapshot_ages_s, 0.99),
            "snapshot_age_max_s": max(snapshot_ages_s) if snapshot_ages_s else None,
            "snapshot_stale_steps": stale_steps,
            "snapshot_rate_limited_steps": snapshot_rate_limited_steps,
            "kinematic_sync_hz_configured": args.kinematic_sync_hz or None,
            "first_snapshot_joint_stamp_ns": first_snapshot_joint_stamp_ns,
            "latest_snapshot_joint_stamp_ns": latest_snapshot_joint_stamp_ns,
            "articulation_root": articulation_root,
            "articulation_dof_count": len(dof_indices),
            "kinematic_joint_count": len(kinematic_specs),
            "composed_joint_type_counts": composed_joint_type_counts,
            "kinematic_transform_count": len(kinematic_transform_ops),
            "physics_prims_disabled": physics_prims_disabled,
            "collision_schemas_removed": collision_schemas_removed,
            "guide_meshes_hidden": guide_meshes_hidden,
            "lod_meshes_hidden": lod_meshes_hidden,
            "lod_proxies_created": lod_proxies_created,
            "camera_suite": suite,
            "kit_runtime": {
                "rate_limit_enabled": carb.settings.get_settings().get_as_bool("/app/runLoops/main/rateLimitEnabled"),
                "manual_mode_enabled": carb.settings.get_settings().get_as_bool("/app/runLoops/main/manualModeEnabled"),
                "tasking_thread_count": carb.settings.get_settings().get_as_int("/plugins/carb.tasking.plugin/threadCount"),
                "tbb_max_thread_count": carb.settings.get_settings().get_as_int("/plugins/omni.tbb.globalcontrol/maxThreadCount"),
                "rtx_multi_tick_enabled": carb.settings.get_settings().get_as_bool("/rtx/hydra/supportMultiTickRate"),
                "rtx_per_sensor_tick_tlas": carb.settings.get_settings().get_as_bool(
                    "/rtx/rendering/perSensorTickTlas"
                ),
                "minimal_rendering_requested": args.minimal_rendering,
                "anti_aliasing_requested": args.anti_aliasing,
                "disable_replica_collisions": args.disable_replica_collisions,
                "hide_replica_guide_meshes": args.hide_replica_guide_meshes,
                "visual_lod_proxy": args.visual_lod_proxy,
                "srtx_enabled": carb.settings.get_settings().get_as_bool("/exts/omni.replicator.srtx/enabled"),
                "ros2_publish_with_queue_thread": carb.settings.get_settings().get_as_bool(
                    "/exts/isaacsim.ros2.bridge/publish_with_queue_thread"
                ),
            },
        }
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (args.output / "timeseries.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "wall_time_ns", "wall_gap_ns"))
            for index, value in enumerate(wall_times, 1):
                previous = wall_times[index - 2] if index >= 2 else ""
                writer.writerow((index, value, value - previous if previous != "" else ""))
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
