"""MID360 runtime helpers for standalone and robot-mounted Isaac Sim use."""

from __future__ import annotations

import os
from typing import Mapping


SUPPORTED_PROFILES = {
    "MID360_APPROX",
    "MID360_FULL",
    "MID360_PERFORMANCE",
    "LIVOX_MID360",
    "LIVOX_MID360_PERFORMANCE",
}

# Isaac removes a Replicator render product when its Python handle is garbage
# collected. Keep all runtime objects alive for the lifetime of the Kit app.
_MOUNT_FOLLOWERS: list["_MountedSensorFollower"] = []
_KINEMATIC_SENSOR_ROOTS: list[object] = []
_LIDAR_RENDER_PRODUCT_HANDLES: list[object] = []
_LIDAR_SENSOR_HANDLES: list[object] = []
_IMU_SENSOR_HANDLES: dict[str, object] = {}


def _enable_stable_ids() -> None:
    """Enable stable RTX object IDs before creating a lidar product."""
    try:
        import carb

        carb.settings.get_settings().set("/rtx-transient/stableIds/enabled", True)
    except Exception as exc:  # pragma: no cover - depends on Isaac runtime
        print(f"[openflex_isaac_sensors] Stable ID setting unavailable: {exc}", flush=True)


def _resolve_mid360_config(requested_profile: str | None = None) -> tuple[str, dict[str, str] | None]:
    """Resolve the configured Isaac Sim model and Livox-compatible variant."""
    requested = (
        requested_profile
        if requested_profile is not None
        else os.environ.get("OPENFLEX_ISAAC_LIDAR_CONFIG", "Example_Rotary")
    ).strip()
    if requested.upper() in {"MID360_APPROX", "SICK_MULTISCAN165", "MULTISCAN165"}:
        return (
            "multiScan100",
            {"Product": "multiScan165", "Profile": "Profile01_20Hz_0p5deg"},
        )
    return requested or "Example_Rotary", None


def _set_lidar_attributes(stage: object, lidar_path: str, attributes: dict[str, object]) -> None:
    prim = stage.GetPrimAtPath(lidar_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"MID360 prim is invalid: {lidar_path}")
    if hasattr(prim, "HasAPI") and not prim.HasAPI("OmniSensorGenericLidarCoreAPI"):
        prim.ApplyAPI("OmniSensorGenericLidarCoreAPI")
    for name, value in attributes.items():
        attribute = prim.GetAttribute(name)
        if attribute.IsValid():
            attribute.Set(value)
        else:
            raise RuntimeError(f"MID360 profile attribute is unavailable: {name}")


def _configure_mid360_performance_profile(
    stage: object, lidar_path: str, *, tick_rate_hz: float = 10.0
) -> None:
    """Apply the MID360 range/return limits without changing profile scan cadence.

    SICK's selected ``Profile01_20Hz_0p5deg`` authors
    ``patternFiringRateHz=20``.  Writing ``20000`` here looks like a rate-unit
    conversion but disables GenericModelOutput on Isaac Sim 6.0.  Keep the
    variant-authored firing rate and apply only the performance limits.
    """
    del tick_rate_hz  # the selected sensor variant owns its scan geometry/rate
    _set_lidar_attributes(
        stage,
        lidar_path,
        {
            "omni:sensor:Core:maxReturns": 1,
            "omni:sensor:Core:nearRangeM": 0.1,
            "omni:sensor:Core:farRangeM": 40.0,
        },
    )


def _configure_mid360_full_profile(
    stage: object, lidar_path: str, *, tick_rate_hz: float = 10.0
) -> None:
    """Apply full-range limits without overriding the selected variant's scan rate."""
    del tick_rate_hz
    _set_lidar_attributes(
        stage,
        lidar_path,
        {
            "omni:sensor:Core:nearRangeM": 0.1,
            "omni:sensor:Core:farRangeM": 40.0,
        },
    )


def _create_lidar_render_product(lidar: object, sensor_name: str) -> tuple[str, object | None]:
    """Create the Isaac RTX render product and retain its Python handles."""
    # Isaac Sim 5.1's stable LidarRtx wrapper owns the render product itself.
    # Reuse it instead of trying the Isaac Sim 6 experimental LidarSensor API
    # or creating a second product from the OmniLidar prim.
    get_render_product_path = getattr(lidar, "get_render_product_path", None)
    if callable(get_render_product_path):
        render_product_path = str(get_render_product_path())
        if not render_product_path:
            raise RuntimeError("Isaac LidarRtx did not create a render product")
        attach_annotator = getattr(lidar, "attach_annotator", None)
        if not callable(attach_annotator):
            raise RuntimeError("Isaac LidarRtx does not expose attach_annotator()")
        attach_annotator("GenericModelOutput")
        _LIDAR_SENSOR_HANDLES.append(lidar)
        render_product = getattr(lidar, "render_product", None) or getattr(
            lidar, "_render_product", None
        )
        if render_product is not None:
            _LIDAR_RENDER_PRODUCT_HANDLES.append(render_product)
        return render_product_path, lidar

    try:
        from isaacsim.sensors.experimental.rtx import LidarSensor

        # The historical working path explicitly registers GenericModelOutput
        # before the ROS2 helper is evaluated. Without it, discovery can show a
        # publisher while no PointCloud2 samples arrive.
        sensor = LidarSensor(lidar, annotators=["generic-model-output"])
        _LIDAR_SENSOR_HANDLES.append(sensor)
        render_product = sensor.render_product
        _LIDAR_RENDER_PRODUCT_HANDLES.append(render_product)
        return str(render_product.GetPath()), sensor
    except ImportError:
        import omni.replicator.core as rep

        render_product = rep.create.render_product(
            camera=lidar.paths[0],
            resolution=(128, 128),
            name=sensor_name + "_lidar",
        )
        _LIDAR_RENDER_PRODUCT_HANDLES.append(render_product)
        return str(render_product.path), None


def _attach_native_pointcloud_writer(
    sensor: object,
    topic: str,
    frame_id: str,
    *,
    object_id_map: bool = True,
) -> None:
    """Attach Isaac Sim's native PointCloud2 writer for diagnostics."""
    sensor.attach_writer(
        "RtxLidarROS2PublishPointCloud",
        topicName=topic,
        frameId=frame_id,
        outputIntensity=True,
        outputTimestamp=True,
        outputEmitterId=True,
        outputChannelId=True,
        outputObjectId=bool(object_id_map),
    )


def _frame_skip_count() -> int:
    try:
        return max(0, int(os.environ.get("OPENFLEX_LIDAR_FRAME_SKIP", "0")))
    except ValueError:
        return 0


def _create_direct_helper_lidar_graph(
    *,
    graph_path: str,
    render_product_path: str,
    topic: str,
    frame_id: str,
    object_id_map: bool = True,
    frame_skip_count: int = 0,
) -> None:
    """Create the historical LidarSensor-owned helper graph.

    This deliberately mirrors the passing graph in the historical
    ``launch_sensor.py``: ``OnPlaybackTick`` drives the ROS 2 helper directly.
    The helper consumes the GenericModelOutput render product owned by
    ``LidarSensor``.  Inserting an additional ``RunOneSimulationFrame`` node
    changes scheduling semantics and, on Isaac Sim 6, can leave the ROS
    publisher discoverable while no PointCloud2 samples are emitted.
    """
    import omni.graph.core as og

    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("LidarEnableGate", "isaacsim.core.nodes.IsaacSimulationGate"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PointCloudConfig", "isaacsim.ros2.bridge.ROS2RtxLidarPointCloudConfig"),
        ("PointCloudPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "LidarEnableGate.inputs:execIn"),
        ("LidarEnableGate.outputs:execOut", "PointCloudPublish.inputs:execIn"),
        ("ROS2Context.outputs:context", "PointCloudPublish.inputs:context"),
        (
            "PointCloudConfig.outputs:selectedMetadata",
            "PointCloudPublish.inputs:selectedMetadata",
        ),
    ]
    set_values = [
        ("LidarEnableGate.inputs:step", 0),
        ("PointCloudPublish.inputs:renderProductPath", render_product_path),
        ("PointCloudPublish.inputs:topicName", topic),
        ("PointCloudPublish.inputs:frameId", frame_id),
        ("PointCloudPublish.inputs:nodeNamespace", ""),
        ("PointCloudPublish.inputs:resetSimulationTimeOnStop", False),
        ("PointCloudPublish.inputs:type", "point_cloud"),
        ("PointCloudPublish.inputs:fullScan", True),
        ("PointCloudPublish.inputs:frameSkipCount", max(0, int(frame_skip_count))),
        ("PointCloudConfig.inputs:outputIntensity", True),
        ("PointCloudConfig.inputs:outputTimestamp", True),
        ("PointCloudConfig.inputs:outputChannelId", True),
        ("PointCloudConfig.inputs:outputEmitterId", True),
        ("PointCloudConfig.inputs:outputObjectId", bool(object_id_map)),
    ]
    if object_id_map:
        set_values.extend(
            [
                ("PointCloudPublish.inputs:enableObjectIdMap", True),
                (
                    "PointCloudPublish.inputs:objectIdMapTopicName",
                    topic.rstrip("/") + "/object_id_map",
                ),
            ]
        )

    graph, _, _, _ = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: create_nodes,
            keys.CONNECT: connections,
            keys.SET_VALUES: set_values,
        },
    )
    og.Controller.evaluate_sync(graph)


def _create_graph_owned_lidar_graph(
    *,
    graph_path: str,
    lidar_path: str,
    topic: str,
    frame_id: str,
    object_id_map: bool = True,
    frame_skip_count: int = 0,
) -> None:
    """Fallback for Isaac versions without ``LidarSensor``."""
    import omni.graph.core as og

    try:
        import usdrt

        camera_prim = [usdrt.Sdf.Path(lidar_path)]
    except ImportError:
        from pxr import Sdf

        camera_prim = [Sdf.Path(lidar_path)]

    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("SimulationFrame", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
        ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("LidarEnableGate", "isaacsim.core.nodes.IsaacSimulationGate"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("PointCloudConfig", "isaacsim.ros2.bridge.ROS2RtxLidarPointCloudConfig"),
        ("PointCloudPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "SimulationFrame.inputs:execIn"),
        ("SimulationFrame.outputs:step", "RenderProduct.inputs:execIn"),
        ("RenderProduct.outputs:execOut", "LidarEnableGate.inputs:execIn"),
        ("LidarEnableGate.outputs:execOut", "PointCloudPublish.inputs:execIn"),
        ("RenderProduct.outputs:renderProductPath", "PointCloudPublish.inputs:renderProductPath"),
        ("ROS2Context.outputs:context", "PointCloudPublish.inputs:context"),
        (
            "PointCloudConfig.outputs:selectedMetadata",
            "PointCloudPublish.inputs:selectedMetadata",
        ),
    ]
    set_values = [
        ("LidarEnableGate.inputs:step", 0),
        ("RenderProduct.inputs:cameraPrim", camera_prim),
        ("RenderProduct.inputs:enabled", True),
        ("PointCloudPublish.inputs:topicName", topic),
        ("PointCloudPublish.inputs:frameId", frame_id),
        ("PointCloudPublish.inputs:nodeNamespace", ""),
        ("PointCloudPublish.inputs:resetSimulationTimeOnStop", False),
        ("PointCloudPublish.inputs:type", "point_cloud"),
        ("PointCloudPublish.inputs:fullScan", True),
        ("PointCloudPublish.inputs:frameSkipCount", max(0, int(frame_skip_count))),
        ("PointCloudConfig.inputs:outputIntensity", True),
        ("PointCloudConfig.inputs:outputTimestamp", True),
        ("PointCloudConfig.inputs:outputChannelId", True),
        ("PointCloudConfig.inputs:outputEmitterId", True),
        ("PointCloudConfig.inputs:outputObjectId", bool(object_id_map)),
    ]
    if object_id_map:
        set_values.extend(
            [
                ("PointCloudPublish.inputs:enableObjectIdMap", True),
                (
                    "PointCloudPublish.inputs:objectIdMapTopicName",
                    topic.rstrip("/") + "/object_id_map",
                ),
            ]
        )

    graph, _, _, _ = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: create_nodes,
            keys.CONNECT: connections,
            keys.SET_VALUES: set_values,
        },
    )
    og.Controller.evaluate_sync(graph)


def create_physics_imu_graph(stage: object, parent_path: str, frame_id: str, topic: str) -> str:
    """Create the physics IMU and ROS 2 publisher graph."""
    import omni.graph.core as og
    from omni.graph.core import GraphPipelineStage

    imu_path = f"{parent_path}/IMU"
    _make_imu_sensor(imu_path, frequency_hz=200.0)
    try:
        import usdrt

        imu_prim = usdrt.Sdf.Path(imu_path)
    except ImportError:
        from pxr import Sdf

        imu_prim = Sdf.Path(imu_path)

    keys = og.Controller.Keys
    graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": f"{parent_path}/IMU_ROS2_Graph",
            "pipeline_stage": GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnPhysicsStep", "isaacsim.core.nodes.OnPhysicsStep"),
                ("IMUEnableGate", "isaacsim.core.nodes.IsaacSimulationGate"),
                ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                ("PublishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
            ],
            keys.CONNECT: [
                ("OnPhysicsStep.outputs:step", "IMUEnableGate.inputs:execIn"),
                ("IMUEnableGate.outputs:execOut", "ReadIMU.inputs:execIn"),
                ("ReadIMU.outputs:execOut", "PublishIMU.inputs:execIn"),
                ("ReadIMU.outputs:linAcc", "PublishIMU.inputs:linearAcceleration"),
                ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
                ("ReadIMU.outputs:orientation", "PublishIMU.inputs:orientation"),
                ("ReadIMU.outputs:sensorTime", "PublishIMU.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [
                ("IMUEnableGate.inputs:step", 0),
                ("ReadIMU.inputs:imuPrim", [imu_prim]),
                ("ReadIMU.inputs:readGravity", True),
                ("PublishIMU.inputs:topicName", topic),
                ("PublishIMU.inputs:frameId", frame_id),
                ("PublishIMU.inputs:publishOrientation", True),
                ("PublishIMU.inputs:publishLinearAcceleration", True),
                ("PublishIMU.inputs:publishAngularVelocity", True),
            ],
        },
    )
    del graph
    return imu_path


def _make_imu_sensor(imu_path: str, *, frequency_hz: float) -> object:
    """Create an IMU through Isaac Sim 6 experimental or stable 5.1 APIs."""
    try:
        from isaacsim.sensors.experimental.physics import IMU, IMUSensor

        sensor = IMUSensor(
            IMU.create(
                imu_path,
                translations=[[0.0, 0.0, 0.0]],
                orientations=[[1.0, 0.0, 0.0, 0.0]],
            )
        )
    except ImportError:
        import numpy as np
        from isaacsim.sensors.physics import IMUSensor

        sensor = IMUSensor(
            prim_path=imu_path,
            name="openflex_imu",
            frequency=max(1, int(round(frequency_hz))),
            translation=np.zeros(3, dtype=np.float64),
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
    _IMU_SENSOR_HANDLES[imu_path] = sensor
    return sensor


def release_physics_imu_sensor(imu_path: str) -> None:
    """Release the Python wrapper before the runtime removes its USD prim."""
    sensor = _IMU_SENSOR_HANDLES.pop(imu_path, None)
    destroy = getattr(sensor, "destroy", None)
    if callable(destroy):
        destroy()


def _create_lidar(
    stage: object,
    lidar_path: str,
    *,
    tick_rate_hz: float = 10.0,
    sensor_config: str | None = None,
) -> object:
    """Create a LiDAR using the base model selected by the sensor YAML."""
    config, variant = _resolve_mid360_config(sensor_config)
    print(
        f"[openflex_isaac_sensors] Creating MID360 with config={config}, "
        f"variant={variant or 'default'}, tick_rate_hz={tick_rate_hz:g}",
        flush=True,
    )
    try:
        try:
            from isaacsim.sensors.experimental.rtx import Lidar
        except ImportError:
            from isaacsim.sensors.rtx import LidarRtx

            stable_config = config
            stable_kwargs: dict[str, object] = {}
            if isinstance(variant, Mapping):
                # 5.1's LidarRtx accepts a shipped product configuration but
                # does not expose the 6.0 Product/Profile mapping. Select the
                # closest installed product and retain our explicit profile
                # attributes below.
                stable_config = str(variant.get("Product") or config)
            elif variant:
                stable_kwargs["variant"] = variant
            lidar = LidarRtx(
                prim_path=lidar_path,
                name="openflex_mid360",
                config_file_name=stable_config,
                **stable_kwargs,
            )
        else:
            lidar = Lidar.create(
                path=lidar_path,
                config=config,
                variant=variant,
                accumulate_outputs=True,
                aux_output_level="FULL",
                tick_rate=float(tick_rate_hz),
            )
    except Exception as exc:
        raise RuntimeError(f"failed to create MID360 at {lidar_path}: {exc}") from exc
    actual_path = _lidar_prim_path(lidar)
    prim = stage.GetPrimAtPath(actual_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"failed to create MID360 at {actual_path}")
    return lidar


def _lidar_prim_path(lidar: object) -> str:
    """Normalize Isaac Sim 5.1 and 6.0 sensor wrapper path properties."""
    paths = getattr(lidar, "paths", None)
    if paths:
        return str(paths[0])
    path = getattr(lidar, "prim_path", None)
    if path:
        return str(path)
    raise RuntimeError(f"unsupported Isaac RTX lidar handle: {type(lidar).__name__}")


def _lidar_render_product(lidar: object) -> object | None:
    """Return the product handle used by either supported Lidar API."""
    return getattr(lidar, "render_product", None) or getattr(lidar, "_render_product", None)


def _configure_profile(
    stage: object, lidar_path: str, profile: str, *, tick_rate_hz: float = 10.0
) -> str:
    requested = profile.strip().upper()
    if requested not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported MID360 profile: {profile}")
    if requested in {"MID360_PERFORMANCE", "LIVOX_MID360_PERFORMANCE"}:
        _configure_mid360_performance_profile(stage, lidar_path, tick_rate_hz=tick_rate_hz)
    else:
        _configure_mid360_full_profile(stage, lidar_path, tick_rate_hz=tick_rate_hz)
    return requested


def create_standalone_mid360(
    stage: object,
    *,
    profile: str = "MID360_PERFORMANCE",
    root_path: str = "/World/MID360",
    frame_id: str = "livox_frame",
    topic: str = "/openflex/livox_frame/lidar",
    imu_topic: str = "/livox/imu",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    transport: str = "direct",
    tick_rate_hz: float = 10.0,
    sensor_config: str | None = None,
    kinematic_root: bool = True,
    create_imu: bool = True,
) -> dict[str, str]:
    """Create the independent MID360 chain used by the historical baseline."""
    import math
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    if transport not in {"graph_owned", "direct"}:
        raise ValueError("standalone MID360 transport must be 'graph_owned' or 'direct'")
    if not root_path.startswith("/"):
        raise ValueError("root_path must be an absolute USD path")
    if tick_rate_hz <= 0.0:
        raise ValueError("tick_rate_hz must be positive")

    _enable_stable_ids()
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    root_prim = root.GetPrim()
    if kinematic_root:
        if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(root_prim)
        rigid_body = UsdPhysics.RigidBodyAPI(root_prim)
        rigid_body.CreateKinematicEnabledAttr().Set(True)
        mass = UsdPhysics.MassAPI.Apply(root_prim)
        mass.CreateMassAttr().Set(0.001)
        mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))
    pose = UsdGeom.XformCommonAPI(root)
    pose.SetTranslate(Gf.Vec3d(*map(float, translation)))
    pose.SetRotate(
        tuple(float(value) * 180.0 / math.pi for value in rotation_rpy_rad),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    lidar_path = f"{root_path}/Lidar"
    lidar = _create_lidar(
        stage,
        lidar_path,
        tick_rate_hz=tick_rate_hz,
        sensor_config=sensor_config,
    )
    lidar_path = _lidar_prim_path(lidar)
    requested_profile = _configure_profile(
        stage, lidar_path, profile, tick_rate_hz=tick_rate_hz
    )
    graph_path = f"{root_path}/Lidar_ROS2_Graph"
    render_product_path = ""
    if transport == "direct":
        render_product_path, sensor = _create_lidar_render_product(lidar, frame_id)
        if sensor is None:
            raise RuntimeError("historical MID360 transport requires the Isaac Sim LidarSensor API")
        _create_direct_helper_lidar_graph(
            graph_path=graph_path,
            render_product_path=render_product_path,
            topic=topic,
            frame_id=frame_id,
            object_id_map=True,
            frame_skip_count=_frame_skip_count(),
        )
    else:
        _create_graph_owned_lidar_graph(
            graph_path=graph_path,
            lidar_path=lidar_path,
            topic=topic,
            frame_id=frame_id,
            object_id_map=True,
            frame_skip_count=_frame_skip_count(),
        )
    imu_path = create_physics_imu_graph(stage, root_path, frame_id, imu_topic) if create_imu else ""
    print(
        f"[openflex_isaac_sensors] Created standalone MID360 profile={requested_profile}, "
        f"graph={graph_path}, topic={topic}, transport={transport}",
        flush=True,
    )
    return {
        "root_path": root_path,
        "lidar_path": lidar_path,
        **({"imu_path": imu_path} if imu_path else {}),
        "graph_path": graph_path,
        "transport": transport,
        **({"render_product_path": render_product_path} if render_product_path else {}),
    }


def robot_sensor_root_path(robot_prim_path: str) -> str:
    """Return the stage-level root used by robot-mounted runtime sensors."""
    robot_name = robot_prim_path.rstrip("/").rsplit("/", 1)[-1] or "OpenFlex"
    clean_name = "".join(char if char.isalnum() or char == "_" else "_" for char in robot_name)
    return f"/World/{clean_name}_Sensors/MID360"


def robot_lidar_graph_path(robot_prim_path: str) -> str:
    """Return the LiDAR graph path below the independent sensor root."""
    return robot_sensor_root_path(robot_prim_path) + "/Lidar_ROS2_Graph"


class _MountedSensorFollower:
    """Optionally keep a stage-level sensor root aligned with a robot mount.

    Runtime USD transform writes are unsafe for an active RTX render product on
    the Isaac Sim 6 path used by this repository.  The default behavior is
    therefore a one-time pose snapshot, which is the safe behavior used by the
    validated robot+MID360 runs.  Set ``OPENFLEX_MID360_UNSAFE_FOLLOW=1`` only
    for an isolated experiment that explicitly needs runtime synchronization.
    """

    def __init__(self, stage: object, mount_path: str, root_path: str) -> None:
        from pxr import Gf, UsdGeom

        self._stage = stage
        self._mount_path = mount_path
        self._root_path = root_path
        mount_prim = stage.GetPrimAtPath(mount_path)
        if not mount_prim or not mount_prim.IsValid():
            raise RuntimeError(f"MID360 mount prim does not exist: {mount_path}")

        root = UsdGeom.Xform.Define(stage, root_path)
        root_prim = root.GetPrim()
        # This root is a presentation/runtime transform only.  It is deliberately
        # outside the robot articulation and must not be authored as a rigid body:
        # PhysX can otherwise pull the independently-created sensor tree into the
        # robot's closed articulation when the follower updates its pose.
        # Keep this guard for stages left behind by an earlier process so a rerun
        # does not retain the problematic APIs in the live session.
        for api_name in ("PhysicsRigidBodyAPI", "PhysicsMassAPI"):
            if root_prim.HasAPI(api_name):
                root_prim.RemoveAPI(api_name)

        xform = UsdGeom.Xformable(root_prim)
        self._transform_op = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTransform),
            None,
        ) or xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        self._last_transform = None

        self.sync()

        if os.environ.get("OPENFLEX_MID360_UNSAFE_FOLLOW", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            try:
                from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager

                self._physics_callback_id = SimulationManager.register_callback(
                    self._on_post_physics_step,
                    IsaacEvents.POST_PHYSICS_STEP,
                    order=100,
                )
                print(
                    "[openflex_isaac_sensors] WARNING: unsafe MID360 runtime follow enabled; "
                    "this may race RTX rendering",
                    flush=True,
                )
            except Exception as exc:  # pragma: no cover - depends on Isaac runtime
                print(
                    "[openflex_isaac_sensors] unsafe MID360 follow unavailable; "
                    f"keeping startup pose only: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    def _on_post_physics_step(self, _step_dt: float, _context: object | None = None) -> None:
        self.sync()

    def sync(self) -> None:
        from pxr import Gf, Usd, UsdGeom

        mount_prim = self._stage.GetPrimAtPath(self._mount_path)
        root_prim = self._stage.GetPrimAtPath(self._root_path)
        if not mount_prim or not mount_prim.IsValid() or not root_prim or not root_prim.IsValid():
            return
        world = UsdGeom.Xformable(mount_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # Do not propagate a referenced USD's unit scale into the RTX sensor.
        rigid = Gf.Matrix4d(1.0)
        rigid.SetRotateOnly(world.ExtractRotation())
        rigid.SetTranslateOnly(world.ExtractTranslation())
        if self._last_transform is not None and all(
            abs(float(rigid[row][column]) - float(self._last_transform[row][column])) <= 1.0e-9
            for row in range(4)
            for column in range(4)
        ):
            return
        self._transform_op.Set(rigid)
        self._last_transform = Gf.Matrix4d(rigid)


def _copy_rigid_world_transform(stage: object, source_path: str) -> object:
    """Return a source prim's world pose without propagating USD scale."""
    from pxr import Gf, Usd, UsdGeom

    source_prim = stage.GetPrimAtPath(source_path)
    if not source_prim or not source_prim.IsValid():
        raise RuntimeError(f"MID360 source prim does not exist: {source_path}")
    world = UsdGeom.Xformable(source_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rigid = Gf.Matrix4d(1.0)
    rigid.SetRotateOnly(world.ExtractRotation())
    rigid.SetTranslateOnly(world.ExtractTranslation())
    return rigid


def _create_fixed_kinematic_sensor_root(
    stage: object,
    mount_path: str,
    root_path: str,
) -> object:
    """Create a stage-level kinematic root at the mount's current world pose.

    This is intentionally the historical standalone MID360 physical setup,
    but the pose is initialized from the loaded robot mount.  No update
    callback is installed, so the sensor remains fixed for the A/B test.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    root_prim = root.GetPrim()
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid_body = UsdPhysics.RigidBodyAPI(root_prim)
    rigid_body.CreateKinematicEnabledAttr().Set(True)
    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr().Set(0.001)
    mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0e-6, 1.0e-6, 1.0e-6))

    xform = UsdGeom.Xformable(root_prim)
    transform_op = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform
        ),
        None,
    ) or xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(_copy_rigid_world_transform(stage, mount_path))
    return root


def _create_parented_sensor_root(stage: object, mount_path: str, root_path: str) -> object:
    """Create an identity sensor root below the robot mount hierarchy."""
    from pxr import Gf, Sdf, UsdGeom

    expected_prefix = mount_path.rstrip("/") + "/"
    if not root_path.startswith(expected_prefix):
        raise RuntimeError(
            f"parented MID360 root must be below {mount_path}: {root_path}"
        )
    root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
    xform = UsdGeom.Xformable(root)
    transform_op = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTransform
        ),
        None,
    ) or xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    transform_op.Set(Gf.Matrix4d(1.0))
    return root


def _format_world_transform(stage: object, prim_path: str) -> str:
    """Return a compact transform diagnostic for a runtime sensor mount."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return f"{prim_path}=INVALID"
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotation().GetQuat()
    return (
        f"{prim_path}: t=({float(translation[0]):.4f}, {float(translation[1]):.4f}, "
        f"{float(translation[2]):.4f}) "
        f"q=({float(rotation.GetReal()):.6f}, {float(rotation.GetImaginary()[0]):.6f}, "
        f"{float(rotation.GetImaginary()[1]):.6f}, {float(rotation.GetImaginary()[2]):.6f})"
    )


def _print_robot_mid360_diagnostics(stage: object, mount_path: str, root_path: str, lidar_path: str) -> None:
    """Print stage/pose facts needed to distinguish an empty scan from a bad mount."""
    try:
        from pxr import UsdGeom

        ground_path = "/World/MID360TestEnvironment/Ground"
        ground = stage.GetPrimAtPath(ground_path)
        lidar = stage.GetPrimAtPath(lidar_path)
        root = stage.GetPrimAtPath(root_path)
        print(
            "[openflex_isaac_sensors] MID360 diagnostics: "
            + "; ".join(
                (
                    _format_world_transform(stage, mount_path),
                    _format_world_transform(stage, root_path),
                    _format_world_transform(stage, lidar_path),
                    f"ground_valid={bool(ground and ground.IsValid())}",
                    f"root_valid={bool(root and root.IsValid())}",
                    f"lidar_valid={bool(lidar and lidar.IsValid())}",
                )
            ),
            flush=True,
        )
        if lidar and lidar.IsValid():
            for attribute_name in (
                "omni:sensor:Core:scanType",
                "omni:sensor:Core:scanRateBaseHz",
                "omni:sensor:Core:patternFiringRateHz",
                "omni:sensor:Core:validStartAzimuthDeg",
                "omni:sensor:Core:validEndAzimuthDeg",
                "omni:sensor:Core:numberOfEmitters",
                "omni:sensor:Core:maxReturns",
                "omni:sensor:Core:nearRangeM",
                "omni:sensor:Core:farRangeM",
                "omni:sensor:Core:horizontalFov",
                "omni:sensor:Core:verticalFov",
            ):
                attribute = lidar.GetAttribute(attribute_name)
                if attribute.IsValid():
                    print(
                        f"[openflex_isaac_sensors] MID360 attribute {attribute_name}="
                        f"{attribute.Get()}",
                        flush=True,
                    )
            elevation = lidar.GetAttribute(
                "omni:sensor:Core:emitterState:s001:elevationDeg"
            )
            if elevation.IsValid():
                values = elevation.Get()
                if values is not None and len(values) > 0:
                    print(
                        "[openflex_isaac_sensors] MID360 vertical emitter field: "
                        f"count={len(values)}, min={min(values):.3f}deg, "
                        f"max={max(values):.3f}deg",
                        flush=True,
                    )
        if root and root.IsValid():
            print(
                "[openflex_isaac_sensors] MID360 root xform ops: "
                + ", ".join(
                    f"{op.GetOpName()}={op.Get()!s}" for op in UsdGeom.Xformable(root).GetOrderedXformOps()
                ),
                flush=True,
            )
    except Exception as exc:  # pragma: no cover - Isaac runtime diagnostics
        print(f"[openflex_isaac_sensors] MID360 diagnostics failed: {type(exc).__name__}: {exc}", flush=True)


def create_robot_mid360(
    stage: object,
    parent_path: str,
    frame_id: str,
    topic: str,
    profile: str,
    *,
    graph_path: str,
    sensor_root_path: str | None = None,
    transport: str = "helper",
    mount_mode: str = "parented",
    object_id_map: bool = True,
    tick_rate_hz: float = 10.0,
    sensor_config: str | None = None,
    resource_sink: dict[str, object] | None = None,
) -> str:
    """Create a robot MID360 using the historical ROS helper chain.

    ``parented`` creates the sensor below the robot mount, so USD hierarchy
    propagation keeps it attached without runtime transform writes.
    ``fixed_kinematic`` and ``fixed`` retain the historical stage-level
    diagnostic modes. ``follow`` is experimental compatibility behavior.
    """
    requested_profile = profile.strip().upper()
    if requested_profile not in SUPPORTED_PROFILES:
        raise ValueError(f"unsupported MID360 profile: {profile}")
    if transport not in {"helper", "native"}:
        raise ValueError("MID360 transport must be 'helper' or 'native'")
    if mount_mode not in {"parented", "follow", "fixed_kinematic", "fixed"}:
        raise ValueError(
            "MID360 mount_mode must be 'parented', 'follow', 'fixed_kinematic', or 'fixed'"
        )
    if tick_rate_hz <= 0.0:
        raise ValueError("MID360 tick_rate_hz must be positive")

    # Keep the validated historical chain local to the consolidated package.
    # The old standalone repository exposed these helpers through
    # ``launch_sensor``.  Importing that module here made an otherwise
    # self-contained replica depend on the old repository's Python path and
    # caused the combined robot+camera+MID360 process to fail before startup.
    # These local helpers intentionally retain the same LidarSensor,
    # GenericModelOutput, and ROS2 helper scheduling semantics.
    _enable_stable_ids()
    root_path = sensor_root_path or f"{parent_path.rstrip('/')}/MID360"
    if mount_mode == "parented":
        _KINEMATIC_SENSOR_ROOTS.append(
            _create_parented_sensor_root(stage, parent_path, root_path)
        )
    elif mount_mode == "follow":
        follower = _MountedSensorFollower(stage, parent_path, root_path)
        _MOUNT_FOLLOWERS.append(follower)
    elif mount_mode == "fixed_kinematic":
        _KINEMATIC_SENSOR_ROOTS.append(
            _create_fixed_kinematic_sensor_root(stage, parent_path, root_path)
        )
    else:
        # B-mode is deliberately kept available for the next isolated test:
        # fixed pose, but no physics APIs on the stage-level sensor root.
        from pxr import Sdf, UsdGeom

        root = UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
        transform_op = root.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
        transform_op.Set(_copy_rigid_world_transform(stage, parent_path))
        _KINEMATIC_SENSOR_ROOTS.append(root)
    if resource_sink is not None:
        resource_sink.update({
            "stage": stage,
            "root_path": root_path,
            "graph_path": graph_path,
        })
    lidar_path = f"{root_path}/Lidar"
    lidar = _create_lidar(
        stage,
        lidar_path,
        tick_rate_hz=tick_rate_hz,
        sensor_config=sensor_config,
    )
    lidar_path = _lidar_prim_path(lidar)
    if resource_sink is not None:
        resource_sink.update({"lidar": lidar, "lidar_path": lidar_path})
    _configure_profile(stage, lidar_path, requested_profile, tick_rate_hz=tick_rate_hz)

    if transport == "native":
        render_product_path, sensor = _create_lidar_render_product(lidar, frame_id)
        if resource_sink is not None:
            resource_sink.update({
                "lidar_sensor": sensor,
                "render_product": _lidar_render_product(sensor),
                "render_product_path": render_product_path,
            })
        if sensor is None:
            raise RuntimeError("native MID360 transport requires the Isaac Sim LidarSensor API")
        _attach_native_pointcloud_writer(
            sensor, topic, frame_id, object_id_map=object_id_map
        )
        print(
            f"[openflex_isaac_sensors] Created MID360 native PointCloud2 writer at {render_product_path}, "
            f"topic={topic}",
            flush=True,
        )
        return lidar_path

    render_product_path, sensor = _create_lidar_render_product(lidar, frame_id)
    if resource_sink is not None:
        resource_sink.update({
            "lidar_sensor": sensor,
            "render_product": _lidar_render_product(sensor),
            "render_product_path": render_product_path,
        })
    if sensor is None:
        raise RuntimeError("historical MID360 helper transport requires the Isaac Sim LidarSensor API")
    _create_direct_helper_lidar_graph(
        graph_path=graph_path,
        render_product_path=render_product_path,
        topic=topic,
        frame_id=frame_id,
        object_id_map=object_id_map,
        frame_skip_count=_frame_skip_count(),
    )
    _print_robot_mid360_diagnostics(stage, parent_path, root_path, lidar_path)
    print(
        f"[openflex_isaac_sensors] Created MID360 at {lidar_path}, graph={graph_path}, "
        f"topic={topic}, transport=historical-helper, mount_mode={mount_mode}",
        flush=True,
    )
    return lidar_path
