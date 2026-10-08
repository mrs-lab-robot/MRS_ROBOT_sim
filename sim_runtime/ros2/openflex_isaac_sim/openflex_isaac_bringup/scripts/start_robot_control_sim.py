#!/usr/bin/env python3
"""Run the Isaac REST loop with direct USD loading and ROS 2 robot control.

Isaac references the canonical OpenFlex USD; the generated URDF is consumed
only as joint/topic metadata by the existing ROS 2 control graph. Sensors are
exposed as factories and created or destroyed on demand after robot control
is ready.
"""

from __future__ import annotations

import math
import os
import queue
import runpy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


_RUNTIME_SENSOR_MANAGER = None
_RUNTIME_SENSOR_SERVER = None
_RUNTIME_SENSOR_RUNTIME = None

# Older empty_stage.usd revisions placed these navigation fixtures directly
# under /World without the GUI-managed MRS_ prefix. Remove them on the first
# profile switch so an already-running Isaac Sim session can leave that stage.
_LEGACY_STARTUP_SCENE_PRIMS = frozenset(
    {"TargetBoxFront", "TargetBoxSide", "TargetWall"}
)


def _isaac_install_root() -> Path | None:
    """Return the standalone root or pip package root for the active Kit Python."""
    configured = os.environ.get("ISAACSIM_PATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if (candidate / "python.sh").is_file() and (candidate / "exts").is_dir():
            return candidate

    executable = Path(sys.executable).resolve()
    for parent in executable.parents:
        if (parent / "python.sh").is_file() and (parent / "exts").is_dir():
            return parent

    try:
        import isaacsim

        package_root = Path(isaacsim.__file__).resolve().parent
        if (package_root / "exts").is_dir():
            return package_root
    except (ImportError, AttributeError, OSError):
        pass
    return None


def _sensor_python_paths() -> list[str]:
    """Keep only the consolidated sensor adapter on Isaac's Python path."""
    paths: list[str] = []
    configured = os.environ.get("OPENFLEX_ISAAC_SENSOR_ASSET_DIR", "").strip()
    roots = [Path(configured).expanduser()] if configured else []

    # The launch contract passes the consolidated repository root. Keep a
    # source-tree fallback so this script also works before the sensor package
    # has been installed into the colcon workspace.
    for root in roots:
        paths.extend(
            (
                str(root / "ros2_pkgs" / "openflex_isaac_sim" / "openflex_isaac_sensors"),
                str(root / "openflex_isaac_sensors"),
                str(root / "ros2" / "openflex_isaac_sensors"),
            )
        )

    # Prefer the installed consolidated package when available. Do not pick
    # the historical package with a similar purpose: importing both trees in
    # Isaac Sim can register duplicate protobuf/ROS bridge symbols.
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        prefix_path = Path(prefix)
        for candidate in (
            prefix_path / "local" / "lib" / "python3.10" / "dist-packages",
            prefix_path / "lib" / "python3.10" / "site-packages",
        ):
            if (candidate / "openflex_isaac_sensors").is_dir():
                paths.append(str(candidate))

    return list(dict.fromkeys(path for path in paths if Path(path).is_dir()))


def _clean_isaac_runtime_environment() -> None:
    """Match the historical stable launcher environment inside Isaac Sim."""
    for key in ("PYTHONHOME", "PYTHONUSERBASE"):
        os.environ.pop(key, None)

    sensor_paths = _sensor_python_paths()
    if sensor_paths:
        os.environ["PYTHONPATH"] = os.pathsep.join(sensor_paths)
        # Isaac's Python process is already running, so updating PYTHONPATH
        # alone does not update sys.path. Put the consolidated sensor package
        # ahead of ROS/system paths before the REST runner imports it.
        for path in reversed(sensor_paths):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
    else:
        os.environ.pop("PYTHONPATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"

    isaac_root = _isaac_install_root()
    if isaac_root is None:
        return
    bridge_paths = []
    for exts_root in (isaac_root / "exts" / "3", isaac_root / "exts"):
        bridge_roots = list(exts_root.glob("isaacsim.ros2.bridge-*"))
        unversioned_bridge = exts_root / "isaacsim.ros2.bridge"
        if unversioned_bridge.is_dir():
            bridge_roots.append(unversioned_bridge)
        for bridge_root in bridge_roots:
            library_path = bridge_root / "humble" / "lib"
            if library_path.is_dir():
                bridge_paths.append(str(library_path))
    if bridge_paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            list(dict.fromkeys(bridge_paths + [entry for entry in existing if entry]))
        )


def _patch_simulation_app_startup() -> None:
    """Apply the single-GPU startup contract before Kit is constructed.

    The generic REST runner creates ``SimulationApp`` itself.  Settings changed
    after construction are too late for renderer device selection, and the
    default Isaac Sim profile currently enables multi-GPU even on a one-GPU
    machine.  The validated MID360 path uses one renderer device, so inject
    those Kit command-line settings into the runner's SimulationApp config.
    """
    import isaacsim

    original = getattr(isaacsim, "SimulationApp", None)
    if original is None or getattr(original, "_isaacsim_robot_single_gpu", False):
        return

    def simulation_app_with_single_gpu(config=None, *args, **kwargs):
        if isinstance(config, dict):
            config = dict(config)
            if config.get("headless", False):
                # The legacy REST runner passes hide_ui=False even in
                # headless mode. Keep the UI hidden and avoid updating Kit's
                # unused default viewport; camera and lidar RenderProducts
                # remain enabled and are updated by the simulation.
                config["hide_ui"] = True
                config["disable_viewport_updates"] = True
            extra_args = list(config.get("extra_args", []))
            required_args = (
                "--/app/runLoops/main/rateLimitEnabled=false",
                "--/app/runLoops/main/manualModeEnabled=true",
                "--/rtx/hydra/supportMultiTickRate=true",
                "--/renderer/raytracingMotion/enabled=true",
                "--/renderer/multiGpu/enabled=false",
                "--/exts/omni.replicator.srtx/enabled=false",
            )
            for value in required_args:
                if value not in extra_args:
                    extra_args.append(value)
            config["extra_args"] = extra_args
        return original(config, *args, **kwargs)

    simulation_app_with_single_gpu._isaacsim_robot_single_gpu = True
    isaacsim.SimulationApp = simulation_app_with_single_gpu


def _patch_kit_app_lookup_before_startup() -> None:
    """Make Isaac 6's pre-start Kit lookup follow the legacy None contract."""
    import omni.kit.app

    original = omni.kit.app.get_app
    if getattr(original, "_openflex_none_before_startup", False):
        return

    def get_app_if_started():
        try:
            return original()
        except RuntimeError as error:
            if "Failed to acquire interface: omni::kit::IApp" in str(error):
                return None
            raise

    get_app_if_started._openflex_none_before_startup = True
    omni.kit.app.get_app = get_app_if_started


def _patch_tensor_view_creation() -> None:
    """Finish PhysX shape setup before Isaac creates tensor simulation views."""
    import omni.physics.tensors
    import omni.physx

    original = omni.physics.tensors.create_simulation_view
    if getattr(original, "_openflex_physx_presettle", False):
        return

    def create_simulation_view_after_presettle(frontend_name, *args, **kwargs):
        backend = kwargs.get("backend", "physx")
        if str(frontend_name).lower() == "warp" and backend == "physx":
            from isaacsim.core.simulation_manager import SimulationManager

            simulation = omni.physx.get_physx_simulation_interface()
            simulation.fetch_results()
            simulation.simulate(SimulationManager.get_physics_dt(), 0.0)
            simulation.fetch_results()
            print(
                "[robot-control-only] completed PhysX presettle before tensor view creation",
                flush=True,
            )
        return original(frontend_name, *args, **kwargs)

    create_simulation_view_after_presettle._openflex_physx_presettle = True
    omni.physics.tensors.create_simulation_view = create_simulation_view_after_presettle


def _ensure_clock_graph() -> None:
    import OmniGraphSchema
    import omni.graph.core as og
    from omni.graph.core import GraphPipelineStage
    import omni.timeline
    import omni.usd

    graph_path = "/Graph/ROS_Clock"
    stage = omni.usd.get_context().get_stage()
    existing = stage.GetPrimAtPath(graph_path) if stage is not None else None
    if existing is not None and existing.IsValid():
        if existing.IsA(OmniGraphSchema.OmniGraph):
            return
        raise RuntimeError(f"cannot create ROS clock graph; path is occupied: {graph_path}")

    omni.timeline.get_timeline_interface().stop()
    keys = og.Controller.Keys
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            # Isaac Sim 6's ROS 2 clock workflow is driven by playback ticks.
            # Physics-step events are not guaranteed to fire for a graph
            # created after the stage has already been opened and played.
            "pipeline_stage": GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [("ReadSimTime.inputs:resetOnStop", False)],
        },
    )
    try:
        og.Controller.evaluate_sync(graph)
    except Exception:
        pass
    print(f"[robot-control-only] created ROS clock graph at {graph_path}", flush=True)


def _isaac6_joint_control_graph_blueprint(
    articulation_prim_path: str,
    joint_states_topic: str,
    joint_command_topic: str,
):
    """Describe the Isaac Sim 6 ROS joint-state/control ActionGraph."""
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("ReadJointState", "isaacsim.sensors.physics.IsaacReadJointState"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
        ("ReadJointState.outputs:execOut", "PublishJointState.inputs:execIn"),
        ("ReadJointState.outputs:jointNames", "PublishJointState.inputs:jointNames"),
        ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
        ("ReadJointState.outputs:jointVelocities", "PublishJointState.inputs:jointVelocities"),
        ("ReadJointState.outputs:jointEfforts", "PublishJointState.inputs:jointEfforts"),
        ("ReadJointState.outputs:jointDofTypes", "PublishJointState.inputs:jointDofTypes"),
        ("ReadJointState.outputs:stageMetersPerUnit", "PublishJointState.inputs:stageMetersPerUnit"),
        ("ReadJointState.outputs:sensorTime", "PublishJointState.inputs:sensorTime"),
        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
        ("Context.outputs:context", "PublishJointState.inputs:context"),
        ("Context.outputs:context", "SubscribeJointState.inputs:context"),
        ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
        (
            "SubscribeJointState.outputs:positionCommand",
            "ArticulationController.inputs:positionCommand",
        ),
        (
            "SubscribeJointState.outputs:velocityCommand",
            "ArticulationController.inputs:velocityCommand",
        ),
        ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
    ]
    values = [
        ("ArticulationController.inputs:robotPath", articulation_prim_path),
        ("PublishJointState.inputs:topicName", joint_states_topic),
        ("SubscribeJointState.inputs:topicName", joint_command_topic),
    ]
    return create_nodes, connections, values


def _install_isaac6_joint_control_graph(urdf_path: str) -> None:
    """Replace the legacy ROS graph with the Isaac Sim 6 joint-state API."""
    import omni.graph.core as og
    import omni.usd
    import usdrt
    from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
    from omni.graph.core import GraphPipelineStage
    from pxr import Sdf

    urdf_root = ET.parse(urdf_path).getroot()
    robot_name = urdf_root.get("name", "").strip()
    if not robot_name:
        raise ValueError(f"URDF has no robot name: {urdf_path}")

    hardware_params = {
        parameter.get("name", ""): (parameter.text or "").strip()
        for parameter in urdf_root.findall(".//ros2_control/hardware/param")
    }
    joint_states_topic = hardware_params.get("joint_states_topic", "")
    joint_command_topic = hardware_params.get("joint_commands_topic", "")
    if not joint_states_topic or not joint_command_topic:
        raise ValueError("URDF topic_based_ros2_control is missing joint state/command topics")

    articulation_prim_path = get_articulation_root_api_prim_path(f"/{robot_name}")
    if not articulation_prim_path:
        raise RuntimeError(f"articulation root was not found below /{robot_name}")
    articulation_prim_path = str(articulation_prim_path)

    graph_path = f"/{robot_name}/ActionGraph"
    stage = omni.usd.get_context().get_stage()
    existing_graph = stage.GetPrimAtPath(graph_path) if stage is not None else None
    if existing_graph is not None and existing_graph.IsValid():
        stage.RemovePrim(Sdf.Path(graph_path))
        if stage.GetPrimAtPath(graph_path).IsValid():
            raise RuntimeError(f"could not replace legacy ROS control graph at {graph_path}")

    create_nodes, connections, values = _isaac6_joint_control_graph_blueprint(
        articulation_prim_path,
        joint_states_topic,
        joint_command_topic,
    )
    values.append(
        ("ReadJointState.inputs:prim", [usdrt.Sdf.Path(articulation_prim_path)])
    )
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.CONNECT: connections,
            og.Controller.Keys.SET_VALUES: values,
        },
    )
    og.Controller.evaluate_sync(graph)
    print(
        "[robot-control-only] installed Isaac Sim 6 joint control graph: "
        f"prim={articulation_prim_path} state={joint_states_topic} command={joint_command_topic}",
        flush=True,
    )


def _installed_sensor_share() -> Path:
    """Resolve the installed sensor config for the runtime sensor factories."""

    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if prefix:
            candidate = Path(prefix) / "share" / "openflex_isaac_sensors"
            if (candidate / "config" / "realsense_robot_mounts.yaml").is_file():
                return candidate
    raise RuntimeError("installed openflex_isaac_sensors config was not found in AMENT_PREFIX_PATH")


def _sensor_asset_root() -> Path:
    """Resolve the consolidated sensor-asset root inside Isaac Python.

    Component 5 passes this explicitly because Isaac Sim runs in its own
    Python environment and cannot reliably discover the colcon source tree.
    The workspace root is also accepted for the in-repository layout, where
    MID360 configuration lives under ``isaac_sim_core/config``.
    """
    configured = (
        os.environ.get("OPENFLEX_ISAAC_MID360_ASSET_DIR", "").strip()
        or os.environ.get("OPENFLEX_ISAAC_SENSOR_ASSET_DIR", "").strip()
    )
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser().resolve())
    for parent in Path(__file__).resolve().parents:
        candidates.extend((parent, parent / "src" / "openflex_isaac_sensor_assets"))
    for candidate in candidates:
        if (
            (candidate / "mid360" / "config" / "mid360_robot_mount.yaml").is_file()
            or (candidate / "isaac_sim_core" / "config" / "sensor_params" / "mid360" / "mid360_robot_mount.yaml").is_file()
        ):
            return candidate
    raise RuntimeError(
        "MID360 asset root was not found; set OPENFLEX_ISAAC_MID360_ASSET_DIR "
        "or OPENFLEX_ISAAC_SENSOR_ASSET_DIR"
    )


_PRIM_WATCH_LISTENERS: list[tuple[object, object]] = []


def _set_robot_viewport() -> None:
    """Set the interactive Isaac viewport to the validated 5.1-style view."""
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[2.0, 2.0, 1.5],
            target=[0.0, 0.0, 0.85],
            camera_prim_path="/OmniverseKit_Persp",
        )
        print("[robot-control-only] set interactive viewport to robot close view", flush=True)
    except Exception as error:
        # Headless runs and stripped-down Kit profiles do not expose a viewport.
        # Camera setup must never make the ROS2 simulation fail.
        print(
            "[robot-control-only] viewport setup skipped: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )


def _reference_robot_usd(
    stage,
    usd_path: str,
    *,
    robot_root_path: str = "/openflex",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> str:
    """Reference the canonical articulated robot USD into the live Isaac stage."""
    from pxr import Gf, UsdGeom

    asset_path = Path(usd_path).expanduser().resolve(strict=True)
    if not asset_path.is_file():
        raise FileNotFoundError(f"robot USD asset is not a file: {asset_path}")

    robot_prim = stage.DefinePrim(robot_root_path, "Xform")
    robot_prim.GetReferences().AddReference(str(asset_path))
    xform = UsdGeom.Xformable(robot_prim)
    asset_translation = (0.0, 0.0, 0.0)
    for operation in xform.GetOrderedXformOps():
        if (
            operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
            and not operation.IsInverseOp()
        ):
            value = operation.Get()
            asset_translation = tuple(float(value[index]) for index in range(3))
            break

    common_xform = UsdGeom.XformCommonAPI(robot_prim)
    common_xform.SetTranslate(
        Gf.Vec3d(
            asset_translation[0] + x,
            asset_translation[1] + y,
            asset_translation[2] + z,
        )
    )
    common_xform.SetRotate(
        Gf.Vec3f(*(math.degrees(angle) for angle in (roll, pitch, yaw))),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )
    return robot_root_path


def _apply_direct_usd_drive_profiles(
    stage,
    urdf_path: str,
    *,
    robot_root_path: str = "/openflex",
) -> list[str]:
    """Overlay control-URDF drive parameters on a directly referenced USD robot.

    The previous URDF importer copied ``isaac_drive_api`` settings into the
    resulting stage before playback. Direct USD loading deliberately skips that
    importer, so retain the same runtime-only dynamics overlay here. The
    canonical USD asset itself remains untouched.
    """
    from pxr import UsdPhysics

    urdf_file = Path(urdf_path).expanduser().resolve(strict=True)
    urdf_root = ET.parse(urdf_file).getroot()
    applied: list[str] = []
    for urdf_joint in urdf_root.findall("joint"):
        profile = urdf_joint.find("isaac_drive_api")
        if profile is None:
            continue

        name = urdf_joint.get("name", "")
        joint_type = urdf_joint.get("type", "")
        if joint_type in {"continuous", "revolute"}:
            drive_axis = "angular"
        elif joint_type == "prismatic":
            drive_axis = "linear"
        else:
            continue
        if not name:
            raise ValueError("isaac_drive_api is attached to an unnamed URDF joint")

        usd_joint = stage.GetPrimAtPath(f"{robot_root_path}/Physics/{name}")
        if not usd_joint.IsValid():
            raise ValueError(f"direct USD joint is missing: {usd_joint.GetPath()}")
        drive = UsdPhysics.DriveAPI.Get(usd_joint, drive_axis)
        if not drive.GetPrim().IsValid():
            raise ValueError(f"direct USD {drive_axis} drive is missing: {usd_joint.GetPath()}")

        attributes = profile.attrib
        if "damping" in attributes:
            drive.CreateDampingAttr().Set(float(attributes["damping"]))
        if "stiffness" in attributes:
            drive.CreateStiffnessAttr().Set(float(attributes["stiffness"]))
        if "max_force" in attributes:
            drive.CreateMaxForceAttr().Set(float(attributes["max_force"]))
        if "joint_friction" in attributes:
            friction = usd_joint.GetAttribute("physxJoint:jointFriction")
            if not friction.IsValid():
                from pxr import Sdf

                friction = usd_joint.CreateAttribute(
                    "physxJoint:jointFriction", Sdf.ValueTypeNames.Float
                )
            friction.Set(float(attributes["joint_friction"]))
        applied.append(name)
    return applied


def _nested_rigid_body_reset_paths(
    body_paths, *, articulation_root_path: str
) -> tuple[str, ...]:
    """Return unique articulation body paths other than the root link."""
    articulation_root_path = str(articulation_root_path)
    return tuple(
        sorted(
            {
                str(path)
                for path in body_paths
                if str(path) != articulation_root_path
            }
        )
    )


def _reset_xform_stack_preserving_world_pose(
    xformable, world_transform, *, precision
) -> None:
    """Reset inherited rigid-body transforms without moving the authored pose."""
    xformable.ClearXformOpOrder()
    transform_op = xformable.AddTransformOp(precision, "openflexWorldPose")
    transform_op.Set(world_transform)
    xformable.SetResetXformStack(True)


def _repair_nested_rigid_body_xforms(
    stage,
    urdf_path: str,
    *,
    robot_root_path: str = "/openflex",
) -> tuple[str, ...]:
    """Author runtime XformStack resets for linked rigid bodies in direct USD.

    The canonical USD nests each moving link below its parent link. PhysX
    rejects that hierarchy unless each child rigid body resets the inherited
    transform stack. Cache the current world poses first, then re-author each
    child link as a reset-stack transform so visuals and initial joint poses
    remain unchanged. The source USD asset is not modified.
    """
    from pxr import UsdGeom, UsdPhysics

    urdf_file = Path(urdf_path).expanduser().resolve(strict=True)
    urdf_root = ET.parse(urdf_file).getroot()
    body_paths: set[str] = set()
    articulation_roots: set[str] = set()

    for urdf_joint in urdf_root.findall("joint"):
        joint_name = urdf_joint.get("name", "").strip()
        if not joint_name:
            continue
        joint_prim = stage.GetPrimAtPath(f"{robot_root_path}/Physics/{joint_name}")
        if not joint_prim.IsValid():
            continue
        joint = UsdPhysics.Joint(joint_prim)
        targets = (
            joint.GetBody0Rel().GetTargets()
            + joint.GetBody1Rel().GetTargets()
        )
        if not targets:
            continue
        for target in targets:
            body_prim = stage.GetPrimAtPath(target)
            if not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(
                    f"USD joint {joint_name} targets a missing/non-rigid body: {target}"
                )
            body_path = str(body_prim.GetPath())
            body_paths.add(body_path)
            if body_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_roots.add(body_path)

    if len(articulation_roots) != 1:
        raise RuntimeError(
            "expected one direct-USD articulation root, found: "
            + ", ".join(sorted(articulation_roots))
        )
    articulation_root_path = next(iter(articulation_roots))
    reset_paths = _nested_rigid_body_reset_paths(
        body_paths, articulation_root_path=articulation_root_path
    )

    # Collect every world transform before authoring any reset; otherwise
    # resetting a parent first would change the cached pose of its descendants.
    xform_cache = UsdGeom.XformCache()
    world_poses = {
        path: xform_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        for path in reset_paths
    }
    changed_paths: list[str] = []
    for path in reset_paths:
        prim = stage.GetPrimAtPath(path)
        xformable = UsdGeom.Xformable(prim)
        if xformable.GetResetXformStack():
            continue
        _reset_xform_stack_preserving_world_pose(
            xformable,
            world_poses[path],
            precision=UsdGeom.XformOp.PrecisionDouble,
        )
        changed_paths.append(path)

    updated_cache = UsdGeom.XformCache()
    for path, expected in world_poses.items():
        actual = updated_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        max_delta = max(
            abs(float(expected[row][column]) - float(actual[row][column]))
            for row in range(4)
            for column in range(4)
        )
        if max_delta > 1e-6:
            raise RuntimeError(
                f"resetting the USD transform stack moved {path} by matrix delta {max_delta}"
            )

    print(
        "[robot-control-only] applied pose-preserving XformStack reset to "
        f"{len(changed_paths)} nested rigid bodies; articulation root={articulation_root_path}",
        flush=True,
    )
    return tuple(changed_paths)


def _reapply_usd_component(stage, params: dict, add_usd, update_usd):
    """Update an existing GUI-managed prim rather than stacking USD references.

    Episode resets only change xform attributes on existing ``MRS_`` prims.
    This preserves the referenced asset and avoids topology resyncs in the
    physics stage. Other REST clients retain the original additive behavior.
    """
    prim_name = str(params.get("prim_name", ""))
    if prim_name.startswith("MRS_"):
        if not prim_name.isidentifier():
            raise ValueError(f"invalid GUI-managed USD prim name: {prim_name!r}")
        prim_path = f"/World/{prim_name}"
        existing_prim = stage.GetPrimAtPath(prim_path)
        if existing_prim is not None and existing_prim.IsValid():
            return update_usd(existing_prim, params)
    result = add_usd(params)
    if (
        prim_name.startswith("MRS_")
        and isinstance(params, dict)
        and "scale" in params
        and isinstance(result, dict)
        and result.get("success")
    ):
        imported_prim = stage.GetPrimAtPath(f"/World/{prim_name}")
        if imported_prim is None or not imported_prim.IsValid():
            return {
                "success": False,
                "message": f"USD 加载成功后未找到 prim：/World/{prim_name}",
            }
        return update_usd(imported_prim, params)
    return result


def _set_usd_component_pose(prim, params: dict):
    """Set root translate/rotate and optional scale ops for imported assets."""
    import math

    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    rotate_op = None
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = operation
        elif operation.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rotate_op = operation
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if rotate_op is None:
        rotate_op = xformable.AddRotateXYZOp()
    translate_op.Set((
        float(params.get("x", 0.0)),
        float(params.get("y", 0.0)),
        float(params.get("z", 0.0)),
    ))
    rotate_op.Set(tuple(
        math.degrees(float(params.get(axis, 0.0)))
        for axis in ("roll", "pitch", "yaw")
    ))
    if "scale" in params:
        scale = params["scale"]
        if not isinstance(scale, (list, tuple)) or len(scale) != 3:
            raise ValueError("USD asset scale must contain exactly three positive values")
        if any(isinstance(value, bool) for value in scale):
            raise ValueError("USD asset scale must contain exactly three positive values")
        scale_values = tuple(float(value) for value in scale)
        if any(not math.isfinite(value) or value <= 0 for value in scale_values):
            raise ValueError("USD asset scale must contain exactly three positive values")
        scale_op = next(
            (
                operation
                for operation in xformable.GetOrderedXformOps()
                if operation.GetOpType() == UsdGeom.XformOp.TypeScale
            ),
            None,
        )
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        scale_op.Set(scale_values)
    return {
        "success": True,
        "data": {"prim_path": str(prim.GetPath()), "updated": True},
    }


def _validate_scene_asset_scale(asset: dict) -> None:
    """Validate and normalize an optional scale before changing the active stage."""
    if "scale" not in asset:
        return
    scale = asset["scale"]
    if not isinstance(scale, (list, tuple)) or len(scale) != 3:
        raise ValueError("环境资产 scale 必须包含三个正数")
    if any(isinstance(value, bool) for value in scale):
        raise ValueError("环境资产 scale 必须包含三个正数")
    try:
        normalized = [float(value) for value in scale]
    except (TypeError, ValueError) as exc:
        raise ValueError("环境资产 scale 必须包含三个正数") from exc
    if any(not math.isfinite(value) or value <= 0 for value in normalized):
        raise ValueError("环境资产 scale 必须包含三个正数")
    asset["scale"] = normalized


def _replace_scene_environment(stage, assets, add_usd, update_usd):
    """Transactionally replace GUI-owned environment prims.

    Every asset is first imported below a unique, temporary ``MRS_`` prim and
    receives its final pose there. Only after the whole new environment is
    ready are current GUI-owned prims renamed to a rollback namespace and the
    staged prims promoted to their public ``/World/MRS_*`` paths. On any error,
    the previous environment is restored and temporary content is removed.
    Robot, sensors, floor, lights, and unrelated world content are untouched.
    """
    import uuid

    if not isinstance(assets, list) or not assets:
        raise ValueError("环境必须包含至少一个 USD 资产")

    desired_names = set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise ValueError("环境资产配置必须是字典")
        prim_name = str(asset.get("prim_name", "")).strip()
        if not prim_name.startswith("MRS_") or not prim_name.isidentifier():
            raise ValueError(f"环境资产 prim_name 必须是 MRS_ 前缀标识符：{prim_name!r}")
        if prim_name in desired_names:
            raise ValueError(f"环境资产 prim_name 重复：{prim_name}")
        _validate_scene_asset_scale(asset)
        desired_names.add(prim_name)

    world = stage.GetPrimAtPath("/World")
    if world is None or not world.IsValid():
        raise RuntimeError("Isaac Sim Stage 缺少 /World，无法切换环境")

    token = uuid.uuid4().hex
    staged = []
    backups = []
    promoted = []

    def remove_if_present(path: str) -> None:
        prim = stage.GetPrimAtPath(path)
        if prim is not None and prim.IsValid():
            stage.RemovePrim(prim.GetPath())

    def rename_world_prim(source_name: str, target_name: str) -> None:
        source_path = f"/World/{source_name}"
        target_path = f"/World/{target_name}"
        stage_rename = getattr(stage, "RenamePrim", None)
        if callable(stage_rename):
            if not stage_rename(source_path, target_path):
                raise RuntimeError(f"无法切换环境prim名称：{source_path} -> {target_path}")
            return

        from pxr import Sdf

        edits = Sdf.BatchNamespaceEdit()
        edits.Add(Sdf.NamespaceEdit.Rename(Sdf.Path(source_path), target_name))
        layer = stage.GetEditTarget().GetLayer()
        if not layer.Apply(edits):
            raise RuntimeError(f"无法切换环境prim名称：{source_path} -> {target_path}")

    try:
        for index, asset in enumerate(assets):
            stage_name = f"MRS_Staging_{token}_{index}"
            staged.append((stage_name, asset["prim_name"]))
            staged_asset = dict(asset)
            staged_asset["prim_name"] = stage_name

            result = add_usd(staged_asset)
            if not isinstance(result, dict) or not result.get("success"):
                raise RuntimeError(
                    f"加载 USD 失败（{asset['prim_name']}）：{result}"
                )
            prim = stage.GetPrimAtPath(f"/World/{stage_name}")
            if prim is None or not prim.IsValid():
                raise RuntimeError(
                    f"USD 加载成功后未找到暂存 prim：/World/{stage_name}"
                )
            pose_result = update_usd(prim, asset)
            if not isinstance(pose_result, dict) or not pose_result.get("success"):
                raise RuntimeError(
                    f"设置环境资产位姿失败（{asset['prim_name']}）：{pose_result}"
                )

        old_prims = [
            prim for prim in world.GetChildren()
            if prim.GetName().startswith("MRS_")
            and prim.GetName() not in {name for name, _ in staged}
            or prim.GetName() in _LEGACY_STARTUP_SCENE_PRIMS
        ]
        for index, prim in enumerate(old_prims):
            old_name = prim.GetName()
            backup_name = f"MRS_Backup_{token}_{index}"
            rename_world_prim(old_name, backup_name)
            backups.append((old_name, backup_name))

        try:
            for stage_name, final_name in staged:
                rename_world_prim(stage_name, final_name)
                promoted.append((stage_name, final_name))
        except Exception:
            for stage_name, final_name in reversed(promoted):
                rename_world_prim(final_name, stage_name)
            for old_name, backup_name in reversed(backups):
                rename_world_prim(backup_name, old_name)
            raise

        removed_names = sorted(
            old_name for old_name, _ in backups
            if old_name not in desired_names or old_name in _LEGACY_STARTUP_SCENE_PRIMS
        )
        for _, backup_name in backups:
            remove_if_present(f"/World/{backup_name}")
    except Exception:
        # If promotion failed, the inner rollback has restored the old names.
        # If backup creation failed partway through, restore whatever moved.
        for stage_name, final_name in reversed(promoted):
            if stage.GetPrimAtPath(f"/World/{final_name}") is not None:
                try:
                    rename_world_prim(final_name, stage_name)
                except Exception:
                    pass
        for old_name, backup_name in reversed(backups):
            if stage.GetPrimAtPath(f"/World/{backup_name}") is not None:
                try:
                    rename_world_prim(backup_name, old_name)
                except Exception:
                    pass
        for stage_name, _ in staged:
            remove_if_present(f"/World/{stage_name}")
        raise

    return {
        "success": True,
        "data": {
            "loaded": sorted(desired_names),
            "removed": removed_names,
        },
    }


def _install_scene_environment_route(server, rest_api_module) -> None:
    """Expose an Isaac-thread queued endpoint for whole-profile scene swaps."""
    from fastapi import HTTPException

    @server.app.post("/set_environment")
    async def set_environment(payload: dict):
        assets = payload.get("assets") if isinstance(payload, dict) else None
        if not isinstance(assets, list) or not assets:
            raise HTTPException(status_code=422, detail="assets 必须是非空列表")
        command = rest_api_module.Command(
            cmd_type="replace_scene_environment",
            params={"assets": assets},
        )
        server.command_queue.put(command)
        try:
            result = command.result_queue.get(timeout=170.0)
        except queue.Empty as error:
            raise HTTPException(status_code=504, detail="等待场景切换超时") from error
        if not result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Isaac Sim 场景切换失败"),
            )
        return result


def _install_control_only_spawn(rest_api_module) -> None:
    def install_prim_resync_watch(stage, robot_root_path="/openflex"):
        """Log a Python stack whenever a robot link prim resyncs at runtime.

        Diagnostics for the physics-view invalidation seen in
        Isaac Sim 6 full-stack runs. Disable with OPENFLEX_PRIM_WATCH=0.
        """
        if os.environ.get("OPENFLEX_PRIM_WATCH", "0").strip().lower() in (
            "0",
            "false",
            "off",
        ):
            return
        import traceback

        from pxr import Tf, Usd

        def _on_objects_changed(notice, _stage):
            for path in notice.GetResyncedPaths():
                if str(path).startswith(robot_root_path + "/Geometry/"):
                    print(
                        f"[prim-watch] resynced {path}\n"
                        + "".join(traceback.format_stack(limit=20)),
                        flush=True,
                    )

        listener = Tf.Notice.Register(
            Usd.Notice.ObjectsChanged, _on_objects_changed, stage
        )
        # Tf.Notice retains Python callbacks weakly. Keep both the listener
        # token and callback alive for the full Kit session.
        _PRIM_WATCH_LISTENERS.append((listener, _on_objects_changed))

    def spawn_robot_control_only(self, params: dict) -> dict:
        import omni.usd

        # Idempotency guard: a stale spawn client can retry after the robot is
        # already referenced and its ROS control graph exists.
        existing_stage = omni.usd.get_context().get_stage()
        existing_root = existing_stage.GetPrimAtPath("/openflex")
        existing_graph = existing_stage.GetPrimAtPath("/openflex/ActionGraph")
        if existing_root.IsValid() and existing_graph.IsValid():
            print(
                "[robot-control-only] /openflex already spawned with a control graph; "
                "acknowledging duplicate spawn request without re-importing",
                flush=True,
            )
            return {
                "success": True,
                "data": {"prim_path": "/openflex", "already_spawned": True},
            }

        stage = omni.usd.get_context().get_stage()
        usd_path = os.environ.get("OPENFLEX_ROBOT_USD", "").strip()
        if not usd_path:
            raise ValueError("usd_path is required; Isaac Sim loads the robot from USD directly")
        prim_path = _reference_robot_usd(
            stage,
            usd_path,
            x=float(params.get("x", 0.0)),
            y=float(params.get("y", 0.0)),
            z=float(params.get("z", 0.0)),
            roll=float(params.get("roll", 0.0)),
            pitch=float(params.get("pitch", 0.0)),
            yaw=float(params.get("yaw", 0.0)),
        )
        applied_drive_profiles = _apply_direct_usd_drive_profiles(
            stage,
            params["urdf_path"],
            robot_root_path=prim_path,
        )
        _repair_nested_rigid_body_xforms(
            stage,
            params["urdf_path"],
            robot_root_path=prim_path,
        )
        print(
            "[robot-control-only] referenced robot USD directly: "
            f"asset={Path(usd_path).expanduser().resolve()} prim={prim_path}",
            flush=True,
        )
        print(
            "[robot-control-only] applied direct-USD drive profiles: "
            + ", ".join(applied_drive_profiles),
            flush=True,
        )
        sensor_catalog_error = ""
        lidar_profile = os.environ.get("OPENFLEX_MID360_LIDAR_PROFILE", "").strip() or None
        lidar_transport = os.environ.get("OPENFLEX_MID360_LIDAR_TRANSPORT", "helper").strip().lower()
        lidar_object_id_map = os.environ.get(
            "OPENFLEX_MID360_OBJECT_ID_MAP", "false"
        ).strip().lower() in ("1", "true", "yes", "on")

        # First stabilize physics without authoring sensor prims. The old
        # startup path pre-created every camera and the lidar before the
        # control graph, which made the sensor page a ROS-output gate rather
        # than a real lifecycle manager.
        import omni.timeline
        import omni.kit.app

        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        _patch_tensor_view_creation()

        # 确保 timeline 处于停止状态
        if timeline.is_playing():
            timeline.stop()
            for _ in range(3):
                app.update()

        stage = omni.usd.get_context().get_stage()
        install_prim_resync_watch(stage)

        # The first play settles collision cooking before articulation tensor
        # views are created. Runtime sensor initialization happens after this
        # stable control session exists; each sensor factory owns its cleanup.
        print("[robot-control-only] prewarming sensor-free physics", flush=True)
        timeline.play()
        for _ in range(24):
            app.update()
        print(
            "[robot-control-only] physics prewarm complete; "
            "keeping timeline active",
            flush=True,
        )

        # 步骤 3：在同一个稳定 physics session 中创建控制图。节点会在后续
        # physics step 初始化并复用现有 simulation view。
        print("[robot-control-only] creating robot control graph while playing", flush=True)
        import robot_controller

        robot_controller.main(urdf_path=params["urdf_path"])
        _install_isaac6_joint_control_graph(params["urdf_path"])
        print("[robot-control-only] robot control graph ready", flush=True)
        # The REST runner's normal loop performs the next Kit update as soon
        # as this request handler returns. Calling app.update() recursively
        # here can stall Kit while its command queue is being processed.

        if _RUNTIME_SENSOR_MANAGER is None:
            raise RuntimeError("runtime sensor control API was not initialized")
        if not _RUNTIME_SENSOR_MANAGER.snapshot()["ready"]:
            global _RUNTIME_SENSOR_RUNTIME
            try:
                from openflex_isaac_sensors.runtime_sensors import RobotSensorRuntime

                _RUNTIME_SENSOR_RUNTIME = RobotSensorRuntime(
                    stage_getter=lambda: omni.usd.get_context().get_stage(),
                    robot_prim_path="/openflex",
                    realsense_asset_dir=_installed_sensor_share(),
                    mid360_asset_dir=_sensor_asset_root(),
                    lidar_profile=lidar_profile,
                    lidar_transport=lidar_transport,
                    lidar_object_id_map=lidar_object_id_map,
                )
                _RUNTIME_SENSOR_RUNTIME.register(_RUNTIME_SENSOR_MANAGER)
            except Exception as error:
                sensor_catalog_error = f"传感器配置加载失败：{type(error).__name__}: {error}"
                print(f"[robot-control-only] {sensor_catalog_error}", flush=True)
            _RUNTIME_SENSOR_MANAGER.mark_ready(error=sensor_catalog_error)
        sensor_snapshot = _RUNTIME_SENSOR_MANAGER.snapshot()
        sensor_catalog_error = sensor_snapshot["error"]
        sensor_catalog = sensor_snapshot["sensors"]
        print(
            "[robot-control-only] runtime sensor lifecycle ready: "
            + (", ".join(sorted(sensor_catalog)) if sensor_catalog else "no sensors")
            + (f"; error={sensor_catalog_error}" if sensor_catalog_error else ""),
            flush=True,
        )

        _set_robot_viewport()

        print("[robot-control-only] ensuring ROS clock graph", flush=True)
        _ensure_clock_graph()
        return {
            "success": True,
            "data": {"prim_path": prim_path, "sensors": sensor_catalog},
        }

    api_class = rest_api_module.IsaacSimRestApi
    api_class._spawn_robot = spawn_robot_control_only

    # Also create the ROS clock graph for an empty-stage benchmark. The
    # graph is harmlessly idempotent when the robot spawn path calls it later,
    # and this lets the empty scene use exactly the same Isaac timing source.
    create_server = rest_api_module.create_server

    def create_server_with_clock(*args, **kwargs):
        global _RUNTIME_SENSOR_MANAGER
        global _RUNTIME_SENSOR_SERVER

        server = create_server(*args, **kwargs)
        _install_scene_environment_route(server, rest_api_module)
        if _RUNTIME_SENSOR_MANAGER is None:
            from openflex_isaac_sensors.runtime_control import (
                RuntimeSensorControlServer,
                RuntimeSensorManager,
                install_main_loop_sensor_pump,
            )

            _RUNTIME_SENSOR_MANAGER = RuntimeSensorManager()
            install_main_loop_sensor_pump(server, _RUNTIME_SENSOR_MANAGER)
            sensor_port = int(os.environ.get("OPENFLEX_SENSOR_CONTROL_PORT", "8086"))
            _RUNTIME_SENSOR_SERVER = RuntimeSensorControlServer(
                _RUNTIME_SENSOR_MANAGER,
                host="127.0.0.1",
                port=sensor_port,
            )
            _RUNTIME_SENSOR_SERVER.start()
            print(
                f"[robot-control-only] loopback sensor control API ready at "
                f"{_RUNTIME_SENSOR_SERVER.base_url}",
                flush=True,
            )
        if os.environ.get("OPENFLEX_EMPTY_SCENE", "").strip().lower() in {"1", "true", "yes", "on"}:
            import omni.timeline

            # Empty-scene benchmarks do not replace the stage after server
            # creation. Robot runs create this graph in spawn_robot_control_only
            # after the final USD stage has been opened, otherwise its
            # OnPhysicsStep callback can remain bound to the invalidated stage.
            _ensure_clock_graph()
            omni.timeline.get_timeline_interface().play()
            print("[robot-control-only] empty-scene benchmark playback started", flush=True)
        return server

    rest_api_module.create_server = create_server_with_clock

    # Keep the control-only contract explicit even if a downstream copy of
    # the REST class dispatches through _execute_command directly.
    original_execute = api_class._execute_command

    def execute_control_only(self, command):
        command_type = getattr(command, "cmd_type", None)
        spawn_type = getattr(rest_api_module.CommandType, "SPAWN_ROBOT", None)
        if command_type == spawn_type:
            return spawn_robot_control_only(self, command.params)
        add_usd_type = getattr(rest_api_module.CommandType, "ADD_USD", None)
        if command_type == add_usd_type:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            return _reapply_usd_component(
                stage, command.params, self._add_usd, _set_usd_component_pose
            )
        if command_type == "replace_scene_environment":
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            return _replace_scene_environment(
                stage,
                command.params.get("assets"),
                self._add_usd,
                _set_usd_component_pose,
            )
        return original_execute(self, command)

    api_class._execute_command = execute_control_only
    print(
        "[robot-control-only] REST spawn hooks installed: "
        f"module={getattr(rest_api_module, '__file__', '<unknown>')} "
        f"class={api_class.__module__}.{api_class.__name__}",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: start_robot_control_sim.py <isaac_ros2_scripts_share> "
            "<stage> [render_hz] [physics_hz] [real_hz] [headless] [api_port]"
        )
    scripts_share = Path(sys.argv[1]).expanduser().resolve()
    # ament_python installs this dependency's ``isaac_scripts/*.py`` files
    # directly below its package share directory (the source tree keeps the
    # extra directory). Support both layouts for source and install spaces.
    isaac_scripts = scripts_share / "isaac_scripts"
    if not (isaac_scripts / "start_sim_with_rest_api.py").is_file():
        isaac_scripts = scripts_share
    runner = isaac_scripts / "start_sim_with_rest_api.py"
    if not runner.is_file():
        raise SystemExit(f"Isaac REST runner does not exist: {runner}")

    _clean_isaac_runtime_environment()
    _patch_simulation_app_startup()
    _patch_kit_app_lookup_before_startup()
    sys.path.insert(0, str(isaac_scripts))

    import rest_api_server

    _install_control_only_spawn(rest_api_server)
    print(
        "[robot-control-only] launching REST runner with "
        f"runner={runner} rest_api={getattr(rest_api_server, '__file__', '<unknown>')}",
        flush=True,
    )
    sys.argv = [str(runner), *sys.argv[2:]]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
