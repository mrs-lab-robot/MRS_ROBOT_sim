"""Create the released RealSense and MID360 suite on a loaded robot USD."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .rig import RealSenseRig
from .sinks import IsaacSimRos2Bridge
from .mount import LocalPose, apply_local_pose


_LIVE_SENSOR_OBJECTS: list[object] = []


def _prepare_robot_sensor_rendering(*, srtx_enabled: bool = False) -> None:
    """Select the requested ROS2 camera transport before any graph exists."""
    try:
        import carb

        carb.settings.get_settings().set("/exts/omni.replicator.srtx/enabled", bool(srtx_enabled))
        print(
            "[openflex_isaac_sensors] "
            + ("Enabled SRTX native ROS2 camera transport" if srtx_enabled else
               "Disabled SRTX; using the stable Replicator ROS2 path")
            + " before robot sensor graph creation"
        )
    except Exception as exc:
        print(f"[openflex_isaac_sensors] Could not configure SRTX: {exc}")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"sensor asset config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"sensor configuration root must be a mapping: {path}")
    return value


def _first_existing(root: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "sensor asset config does not exist; checked: "
        + ", ".join(str(root / value) for value in relative_paths)
    )


def attach_realsense_calibrations(
    config: dict[str, Any], *calibration_dirs: Path
) -> dict[str, Any]:
    """Attach referenced calibration YAMLs to enabled cameras in-place."""
    for camera_name, raw_camera in (config.get("cameras", {}) or {}).items():
        if not isinstance(raw_camera, dict) or not bool(raw_camera.get("enabled", True)):
            continue
        calibration_id = str(raw_camera.get("calibration_id", "")).strip()
        if not calibration_id:
            continue
        candidates = [directory / f"{calibration_id}.yaml" for directory in calibration_dirs]
        calibration_path = next((path for path in candidates if path.is_file()), None)
        if calibration_path is None:
            raise FileNotFoundError(
                f"camera {camera_name} calibration {calibration_id} is missing; checked: "
                + ", ".join(map(str, candidates))
            )
        calibration = _read_yaml(calibration_path)
        if str(calibration.get("calibration_id", "")) != calibration_id:
            raise ValueError(
                f"camera {camera_name} calibration ID does not match {calibration_path}"
            )
        matrix = calibration.get("K")
        if not isinstance(matrix, list) or len(matrix) != 9:
            raise ValueError(f"camera {camera_name} calibration K must contain 9 values")
        if float(matrix[0]) <= 0.0 or float(matrix[4]) <= 0.0:
            raise ValueError(f"camera {camera_name} calibration focal lengths must be positive")
        if int(calibration.get("width", 0)) <= 0 or int(calibration.get("height", 0)) <= 0:
            raise ValueError(f"camera {camera_name} calibration dimensions must be positive")
        raw_camera["calibration"] = calibration
    return config


def load_realsense_config(sensor_asset_dir: Path) -> dict[str, Any]:
    """Load only the RealSense contract.

    The branch-specific snapshot may place config under ``realsense/config``;
    the transition tree keeps the ROS package layout for compatibility.
    """
    path = _first_existing(
        sensor_asset_dir,
        "realsense/config/realsense_robot_mounts.yaml",
        "config/realsense_robot_mounts.yaml",
        "isaac_sim_core/config/sensor_params/realsense/realsense_robot_mounts.yaml",
        "ros2_pkgs/openflex_isaac_sim/openflex_isaac_sensors/config/realsense_robot_mounts.yaml",
    )
    root = Path(sensor_asset_dir).expanduser()
    return attach_realsense_calibrations(
        _read_yaml(path),
        path.parent / "calibration",
        root / "realsense" / "calibration",
        root / "realsense" / "config" / "calibration",
        root / "config" / "calibration",
        root / "isaac_sim_core" / "config" / "sensor_params" / "realsense" / "calibration",
        root / "ros2_pkgs" / "openflex_isaac_sim" / "openflex_isaac_sensors" / "config" / "calibration",
    )


def load_mid360_config(sensor_asset_dir: Path) -> dict[str, Any]:
    """Load only the MID360 contract without importing RealSense code."""
    path = _first_existing(
        sensor_asset_dir,
        "mid360/config/mid360_robot_mount.yaml",
        "isaac_sim_core/config/sensor_params/mid360/mid360_robot_mount.yaml",
    )
    return _read_yaml(path)


def load_robot_sensor_config(sensor_asset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility loader for callers that intentionally request both assets."""
    camera_config = load_realsense_config(sensor_asset_dir)
    lidar_config = load_mid360_config(sensor_asset_dir)
    return camera_config, lidar_config


def resolve_robot_mount_path(configured_path: str, robot_prim_path: str) -> str:
    prefix = "/World/OpenFlex"
    if configured_path == prefix:
        return robot_prim_path
    if configured_path.startswith(prefix + "/"):
        return robot_prim_path.rstrip("/") + configured_path[len(prefix):]
    return configured_path


def _profile_flags(sensor_profile: str) -> tuple[bool, bool]:
    profile = sensor_profile.strip().lower()
    if profile not in {"none", "rgb", "rgb_depth", "lidar", "data", "teleop"}:
        raise ValueError(f"unsupported sensor profile: {sensor_profile}")
    return profile in {"rgb", "rgb_depth", "data"}, profile in {"lidar", "data"}


def _set_omnigraph_gate(attribute_path: str, enabled: bool) -> None:
    import omni.graph.core as og

    attribute = og.Controller.attribute(attribute_path)
    if attribute is None:
        raise RuntimeError(f"sensor execution gate is missing: {attribute_path}")
    attribute.set(1 if enabled else 0)


def bootstrap_camera_gates(camera_records: list[dict]) -> None:
    """Play the timeline briefly to create SyntheticData gates, then configure them.

    This must be called AFTER the ActionGraph is created and the timeline is
    playing, so the ArticulationState tensor view has been initialized first.
    """
    if not camera_records:
        return

    import omni.kit.app
    import omni.timeline
    import omni.usd

    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    if not was_playing:
        timeline.play()

    app = omni.kit.app.get_app()
    for _ in range(6):
        app.update()

    stage = omni.usd.get_context().get_stage()
    for path in (
        "/Render/PostProcess/SDGPipeline",
        "/Render/Simulation/SDGPipeline",
        "/Render/PostRender/SDGPipeline",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            prim.CreateAttribute("isaac:namespace", omni.usd.Sdf.ValueTypeNames.String).Set("")

    # We need the bridge object to configure gates. Re-import to get it.
    # The bridge was already created in create_robot_sensor_suite and stored
    # in _LIVE_SENSOR_OBJECTS.
    bridge = None
    for obj in _LIVE_SENSOR_OBJECTS:
        if isinstance(obj, IsaacSimRos2Bridge):
            bridge = obj
            break
    if bridge is not None:
        bridge.configure_camera_gates(camera_records)
    else:
        print("[openflex_isaac_sensors] Warning: no IsaacSimRos2Bridge found for gate configuration", flush=True)


def create_robot_sensor_suite(
    stage: object,
    robot_prim_path: str,
    sensor_asset_dir: Path | None = None,
    *,
    realsense_asset_dir: Path | None = None,
    mid360_asset_dir: Path | None = None,
    sensor_profile: str = "data",
    lidar_profile: str | None = None,
    lidar_transport: str = "helper",
    lidar_mount_mode: str = "parented",
    lidar_object_id_map: bool = True,
    lidar_tick_rate_hz: float | None = None,
    srtx_enabled: bool = False,
    camera_resolution: tuple[int, int] | None = None,
    publish_camera_info: bool = True,
    camera_tick_rate_hz: float | None = None,
    bootstrap_cameras: bool = True,
) -> dict[str, object]:
    """Attach configured sensors while keeping the robot asset sensor-free."""
    # ``sensor_asset_dir`` is the transition-era combined repository.  The
    # component-5 entrypoint may override each asset root independently once
    # components 1 and 2 are published as separate repositories.
    combined_root = Path(sensor_asset_dir).expanduser() if sensor_asset_dir else None
    realsense_root = Path(realsense_asset_dir).expanduser() if realsense_asset_dir else combined_root
    mid360_root = Path(mid360_asset_dir).expanduser() if mid360_asset_dir else combined_root
    enable_cameras, enable_lidar = _profile_flags(sensor_profile)
    if enable_cameras and realsense_root is None:
        raise ValueError("a RealSense asset root is required for camera startup")
    if enable_lidar and mid360_root is None:
        raise ValueError("a MID360 asset root is required for LiDAR startup")
    camera_config = load_realsense_config(realsense_root) if enable_cameras else {}
    lidar_config = load_mid360_config(mid360_root) if enable_lidar else {}
    lidar_sensor_config = lidar_config.get("sensor", {}) or {}
    lidar_model_config = str(lidar_sensor_config.get("default_profile", "Example_Rotary"))
    lidar_profile = str(
        lidar_profile
        or lidar_sensor_config.get("performance_profile")
        or lidar_sensor_config.get("default_profile")
        or "MID360_PERFORMANCE"
    )
    lidar_tick_rate_hz = float(
        lidar_tick_rate_hz
        if lidar_tick_rate_hz is not None
        else lidar_sensor_config.get("tick_rate_hz", 10.0)
    )
    created: list[str] = []
    camera_records: list[dict[str, object]] = []
    runtime_controls: dict[str, object] = {}
    bridge: IsaacSimRos2Bridge | None = None

    if enable_cameras or enable_lidar:
        _prepare_robot_sensor_rendering(srtx_enabled=srtx_enabled)

    if enable_cameras:
        robot_name = robot_prim_path.rstrip("/").rsplit("/", 1)[-1] or "openflex"
        graph_name = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in robot_name
        )
        bridge = IsaacSimRos2Bridge(f"/{graph_name}_SensorROS2")
        for name, raw_config in (camera_config.get("cameras", {}) or {}).items():
            if not isinstance(raw_config, Mapping) or not bool(raw_config.get("enabled", True)):
                continue
            config = dict(raw_config)
            if camera_resolution is not None:
                width, height = camera_resolution
                if width < 1 or height < 1:
                    raise ValueError("camera_resolution dimensions must be positive")
                config["width"] = int(width)
                config["height"] = int(height)
            if camera_tick_rate_hz is not None:
                if camera_tick_rate_hz <= 0.0:
                    raise ValueError("camera_tick_rate_hz must be positive")
                config["tick_rate_hz"] = float(camera_tick_rate_hz)
            config["mount_prim_path"] = resolve_robot_mount_path(
                str(config["mount_prim_path"]), robot_prim_path
            )
            config["parent_prim"] = resolve_robot_mount_path(
                str(config.get("parent_prim", robot_prim_path)), robot_prim_path
            )
            rig = RealSenseRig(name, config)
            rig.create(
                stage,
                str(config["parent_prim"]),
                config.get("local_pose"),
                config,
                create_render_product=True,
            )
            namespace = str(config["node_namespace"])
            # Publish the released camera contract directly.  Keeping a
            # single canonical topic avoids rendering or relaying each frame
            # twice while preserving the names used by RViz and downstream
            # OpenFlex applications.
            camera_records.append({
                "camera_key": name,
                "render_product_path": rig.render_product_path or "",
                "camera_prim_path": rig.camera_prim_path or "",
                "frame_id": str(config["optical_frame"]),
                "rgb_frame_id": str(config["optical_frame"]),
                "depth_frame_id": str(config.get("depth_optical_frame", config["optical_frame"])),
                "node_namespace": "",
                "rgb_topic": f"/{namespace}/color/image",
                "depth_topic": f"/{namespace}/depth/image",
                "camera_info_topic": f"/{namespace}/color/camera_info",
                "depth_camera_info_topic": f"/{namespace}/depth/camera_info",
                "tick_rate_hz": float(config.get("tick_rate_hz", 30.0)),
                "width": int(config.get("width", 640)),
                "height": int(config.get("height", 480)),
                "publish_camera_info": bool(publish_camera_info),
                "queue_size": int((camera_config.get("queues", {}) or {}).get("ros2_queue_size", 5)),
                "qos_profile": "Sensor Data",
            })
            created.append(rig.camera_prim_path or name)
            _LIVE_SENSOR_OBJECTS.append(rig)
        if camera_records:
            bridge.attach_cameras(camera_records)
            _LIVE_SENSOR_OBJECTS.append(bridge)
            for record in camera_records:
                namespace = str(record.get("rgb_topic", "")).strip("/").split("/", 1)[0]
                suffix = namespace.removeprefix("cam_")
                sensor_id = f"camera_{suffix}"
                camera_key = str(record["camera_key"])
                runtime_controls[sensor_id] = (
                    lambda enabled, key=camera_key: bridge.set_camera_enabled(key, enabled)
                )

    lidar_path = ""
    if enable_lidar:
        from .mid360 import (
            create_physics_imu_graph,
            create_robot_mid360,
            robot_lidar_graph_path,
            robot_sensor_root_path,
        )

        lidar = lidar_config.get("sensor", {}) or {}
        mount_path = resolve_robot_mount_path(str(lidar["mount_prim_path"]), robot_prim_path)
        mount = stage.GetPrimAtPath(mount_path)
        if not mount or not mount.IsValid():
            # Some referenced USD compositions expose camera mounts but omit
            # the authored MID360 mount. Create an equivalent session-local
            # Xform only after the replica has collected its joint chain.
            # Keeping this fallback in the sensor contract avoids patching or
            # re-composing the robot asset just to attach a runtime sensor.
            parent_path = resolve_robot_mount_path(
                str(lidar.get("parent_prim", robot_prim_path)), robot_prim_path
            )
            parent = stage.GetPrimAtPath(parent_path)
            if not parent or not parent.IsValid():
                raise RuntimeError(
                    f"MID360 parent prim does not exist: {parent_path}; cannot create mount {mount_path}"
                )
            fallback_pose = LocalPose.from_mapping(
                lidar.get("fallback_mount_pose") or lidar.get("local_pose")
            )
            apply_local_pose(stage, mount_path, fallback_pose)
            mount = stage.GetPrimAtPath(mount_path)
            if not mount or not mount.IsValid():
                raise RuntimeError(f"failed to create MID360 runtime mount: {mount_path}")
            print(f"[openflex_isaac_sensors] Created missing MID360 runtime mount: {mount_path}")
        sensor_root_path = (
            mount_path.rstrip("/") + "/MID360_Runtime"
            if lidar_mount_mode == "parented"
            else robot_sensor_root_path(robot_prim_path)
        )
        lidar_path = create_robot_mid360(
            stage,
            mount_path,
            str(lidar.get("frame_id", "livox_frame")),
            str(lidar.get("pointcloud_topic", "/openflex/livox_frame/lidar")),
            lidar_profile,
            graph_path=robot_lidar_graph_path(robot_prim_path),
            sensor_root_path=sensor_root_path,
            transport=lidar_transport,
            mount_mode=lidar_mount_mode,
            object_id_map=lidar_object_id_map,
            tick_rate_hz=lidar_tick_rate_hz,
            sensor_config=lidar_model_config,
        )
        create_physics_imu_graph(
            stage,
            sensor_root_path,
            str(lidar.get("frame_id", "livox_frame")),
            str(lidar.get("imu_topic", "/livox/imu")),
        )
        lidar_graph_path = robot_lidar_graph_path(robot_prim_path)
        runtime_controls["lidar"] = lambda enabled: _set_omnigraph_gate(
            f"{lidar_graph_path}/LidarEnableGate.inputs:step", enabled
        )
        imu_graph_path = f"{sensor_root_path}/IMU_ROS2_Graph"
        runtime_controls["imu"] = lambda enabled: _set_omnigraph_gate(
            f"{imu_graph_path}/IMUEnableGate.inputs:step", enabled
        )
        created.extend([sensor_root_path, lidar_path, sensor_root_path + "/IMU"])

    if bridge is not None and camera_records:
        if bootstrap_cameras:
            # Inline bootstrap: play timeline, create SyntheticData gates, configure.
            # Callers that need the ActionGraph created BEFORE the first play()
            # (to avoid ArticulationState tensor view invalidation) should pass
            # bootstrap_cameras=False and call bootstrap_camera_gates() later.
            import omni.kit.app
            import omni.timeline

            omni.timeline.get_timeline_interface().play()
            app = omni.kit.app.get_app()
            for _ in range(6):
                app.update()
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            for path in (
                "/Render/PostProcess/SDGPipeline",
                "/Render/Simulation/SDGPipeline",
                "/Render/PostRender/SDGPipeline",
            ):
                prim = stage.GetPrimAtPath(path)
                if prim and prim.IsValid():
                    prim.CreateAttribute("isaac:namespace", omni.usd.Sdf.ValueTypeNames.String).Set("")
            bridge.configure_camera_gates(camera_records)
        # Always include camera_records in the result so the caller can
        # bootstrap later if needed.
        result_camera_records = camera_records
    else:
        result_camera_records = []

    return {
        "success": True,
        "created": created,
        "camera_records": camera_records,
        "runtime_controls": runtime_controls,
        "camera_count": len(camera_records),
        "lidar_prim_path": lidar_path,
        "sensor_profile": sensor_profile,
        "lidar_profile": lidar_profile,
        "lidar_transport": lidar_transport,
        "lidar_mount_mode": lidar_mount_mode,
        "lidar_object_id_map": bool(lidar_object_id_map),
        "lidar_tick_rate_hz": float(lidar_tick_rate_hz),
        "srtx_enabled": bool(srtx_enabled),
        "camera_resolution": list(camera_resolution) if camera_resolution is not None else None,
        "publish_camera_info": bool(publish_camera_info),
        "camera_tick_rate_hz": float(camera_tick_rate_hz) if camera_tick_rate_hz is not None else None,
        "realsense_asset_dir": str(realsense_root) if realsense_root else "",
        "mid360_asset_dir": str(mid360_root) if mid360_root else "",
    }
