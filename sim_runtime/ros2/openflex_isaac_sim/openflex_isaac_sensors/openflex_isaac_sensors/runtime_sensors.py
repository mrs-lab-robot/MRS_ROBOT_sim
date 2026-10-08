"""On-demand camera, MID360 and IMU lifecycle for a running robot stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from .integration import (
    _prepare_robot_sensor_rendering,
    load_mid360_config,
    load_realsense_config,
    resolve_robot_mount_path,
)
from .mount import LocalPose, apply_local_pose
from .rig import RealSenseRig
from .runtime_control import RuntimeSensorManager, RuntimeSensorResource
from .sinks import IsaacSimRos2Bridge


_CAMERA_LABELS = {
    "camera_head": "头部相机",
    "camera_left": "左手相机",
    "camera_right": "右手相机",
    "camera_base": "底盘相机",
}


def _valid_prim(stage: object, path: str) -> bool:
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def _remove_prim(stage: object, path: str) -> None:
    if not path or not _valid_prim(stage, path):
        return
    from pxr import Sdf

    stage.RemovePrim(Sdf.Path(path))


def _destroy_handle(value: object | None) -> None:
    if value is None:
        return
    destroy = getattr(value, "destroy", None)
    if callable(destroy):
        destroy()
        return
    # Isaac Sim 5.1's LidarRtx exposes resource cleanup through its destructor
    # rather than a public destroy() method. Detach its Replicator resources
    # explicitly so destroying an on-demand sensor is deterministic.
    for method_name in ("detach_all_writers", "detach_all_annotators"):
        detach = getattr(value, method_name, None)
        if callable(detach):
            detach()
    render_product = getattr(value, "render_product", None) or getattr(
        value, "_render_product", None
    )
    destroy_product = getattr(render_product, "destroy", None)
    if callable(destroy_product):
        destroy_product()
        if hasattr(value, "_render_product"):
            value._render_product = None


def _make_camera_runtime_resource(
    *,
    bridge: IsaacSimRos2Bridge,
    camera_key: str,
    rig: RealSenseRig,
    stage: object,
    camera_prim_path: str,
    render_product_path: str,
    ready: Callable[[], bool] | None = None,
    settle_updates: int = 6,
) -> RuntimeSensorResource:
    """Stop publishing first, then release a RenderProduct after Kit updates.

    Hydra may still be consuming the current RenderProduct when an API destroy
    request arrives. Removing that USD prim in the same update causes Kit 6.0
    to dereference a missing product on the next render pass.
    """
    updates = 0
    publishers_disabled = False
    camera_paused = False
    bridge_destroyed = False
    rig_destroyed = False

    def destroy() -> None:
        nonlocal updates, publishers_disabled, camera_paused, bridge_destroyed
        updates = 0
        if not publishers_disabled:
            bridge.set_camera_enabled(camera_key, False)
            publishers_disabled = True
        if not camera_paused:
            prim = stage.GetPrimAtPath(camera_prim_path)
            attribute = prim.GetAttribute("omni:sensor:tickRate") if prim else None
            if attribute and attribute.IsValid():
                attribute.Set(0.0)
            camera_paused = True
        if not bridge_destroyed:
            bridge.destroy(stage)
            bridge_destroyed = True

    def destroy_ready() -> bool:
        nonlocal updates, rig_destroyed
        updates += 1
        if updates < max(1, int(settle_updates)):
            return False
        if not rig_destroyed:
            rig.destroy()
            rig_destroyed = True
        return not any(
            _valid_prim(stage, path)
            for path in (render_product_path, camera_prim_path)
            if path
        )

    return RuntimeSensorResource(
        destroy=destroy,
        ready=ready or (lambda: True),
        destroy_ready=destroy_ready,
    )


class _CameraPublisherBootstrap:
    """Start a camera graph before looking up its generated SyntheticData gates."""

    def __init__(
        self,
        bridge: IsaacSimRos2Bridge,
        camera_name: str,
        camera_records: list[dict[str, Any]],
        *,
        warmup_updates: int = 6,
        set_target_tick_rate: Callable[[], None] | None = None,
    ) -> None:
        self._bridge = bridge
        self._camera_name = camera_name
        self._camera_records = camera_records
        self._warmup_updates = max(1, int(warmup_updates))
        self._set_target_tick_rate = set_target_tick_rate
        self._target_tick_rate_set = set_target_tick_rate is None
        self._updates = 0
        self._enabled = False
        self._ready = False

    def ready(self) -> bool:
        """Advance bootstrap one Kit update at a time; safe to poll repeatedly."""
        if self._ready:
            return True

        self._updates += 1
        if not self._target_tick_rate_set and self._updates >= 2:
            self._set_target_tick_rate()
            self._target_tick_rate_set = True
        if not self._enabled and self._updates >= self._warmup_updates:
            # The camera helper creates /Render/PostProcess/SDGPipeline only
            # after its first execution. Do not query its per-render-product
            # gates while this sensor's own execution gate is still closed.
            self._bridge.set_camera_enabled(self._camera_name, True)
            self._enabled = True
            return False

        if self._enabled and self._updates >= 2 * self._warmup_updates:
            self._bridge.configure_camera_gates(self._camera_records)
            self._ready = True

        return self._ready


class RobotSensorRuntime:
    """Register available sensor factories without authoring any stage prims."""

    def __init__(
        self,
        *,
        stage_getter: Callable[[], object],
        robot_prim_path: str,
        realsense_asset_dir: str | Path,
        mid360_asset_dir: str | Path,
        lidar_profile: str | None = None,
        lidar_transport: str = "helper",
        lidar_object_id_map: bool = False,
        lidar_tick_rate_hz: float | None = None,
    ) -> None:
        self._stage_getter = stage_getter
        self.robot_prim_path = robot_prim_path.rstrip("/")
        self._camera_config = load_realsense_config(Path(realsense_asset_dir).expanduser())
        self._lidar_config = load_mid360_config(Path(mid360_asset_dir).expanduser())
        sensor_config = self._lidar_config.get("sensor", {}) or {}
        self._lidar_model_config = str(sensor_config.get("default_profile", "Example_Rotary"))
        self._lidar_profile = str(
            lidar_profile
            or sensor_config.get("performance_profile")
            or sensor_config.get("default_profile")
            or "MID360_PERFORMANCE"
        )
        self._lidar_transport = str(lidar_transport)
        self._lidar_object_id_map = bool(lidar_object_id_map)
        self._lidar_tick_rate_hz = float(
            lidar_tick_rate_hz
            if lidar_tick_rate_hz is not None
            else sensor_config.get("tick_rate_hz", 10.0)
        )
        self._active_mid360: set[str] = set()
        self._mount_path = ""
        self._mount_created = False
        self._sensor_root_path = ""
        self._sensor_root_created = False

    def register(self, manager: RuntimeSensorManager) -> None:
        """Expose the configured catalog; factories run only on explicit create."""
        cameras = self._camera_config.get("cameras", {}) or {}
        for camera_name, raw_config in cameras.items():
            if not isinstance(raw_config, Mapping) or not bool(raw_config.get("enabled", True)):
                continue
            config = dict(raw_config)
            namespace = str(config.get("node_namespace", camera_name)).strip("/")
            short_name = namespace.removeprefix("cam_")
            sensor_id = f"camera_{short_name}"
            topic = f"/{namespace}/color/image"
            manager.register(
                sensor_id,
                label=_CAMERA_LABELS.get(sensor_id, str(camera_name)),
                topic=topic,
                create=lambda name=str(camera_name), value=config: self._create_camera(
                    name, value
                ),
            )

        sensor = self._lidar_config.get("sensor", {}) or {}
        manager.register(
            "lidar",
            label="Livox MID360S 雷达",
            topic=str(sensor.get("pointcloud_topic", "/openflex/livox_frame/lidar")),
            create=self._create_lidar,
        )
        manager.register(
            "imu",
            label="IMU",
            topic=str(sensor.get("imu_topic", "/livox/imu")),
            create=self._create_imu,
        )

    def _create_camera(self, name: str, raw_config: Mapping[str, Any]) -> RuntimeSensorResource:
        stage = self._stage_getter()
        config = dict(raw_config)
        config["mount_prim_path"] = resolve_robot_mount_path(
            str(config["mount_prim_path"]), self.robot_prim_path
        )
        config["parent_prim"] = resolve_robot_mount_path(
            str(config.get("parent_prim", self.robot_prim_path)), self.robot_prim_path
        )
        _prepare_robot_sensor_rendering(srtx_enabled=False)

        rig = RealSenseRig(name, config)
        bridge: IsaacSimRos2Bridge | None = None
        try:
            rig.create(
                stage,
                str(config["parent_prim"]),
                config.get("local_pose"),
                config,
                create_render_product=False,
            )
            camera_prim = stage.GetPrimAtPath(rig.camera_prim_path)
            tick_rate_attribute = camera_prim.GetAttribute("omni:sensor:tickRate")
            if not tick_rate_attribute or not tick_rate_attribute.IsValid():
                raise RuntimeError(
                    f"camera sensor tick-rate attribute is missing: {rig.camera_prim_path}"
                )
            tick_rate_attribute.Set(0.0)
            rig.create_python_render_product(
                publish_rgb=bool(config.get("publish_rgb", True)),
                publish_depth=bool(config.get("publish_depth", True)),
            )
            namespace = str(config["node_namespace"]).strip("/")
            target_tick_rate_hz = float(config.get("tick_rate_hz", 30.0))
            camera_key = name
            record = {
                "camera_key": camera_key,
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
                "tick_rate_hz": target_tick_rate_hz,
                "width": int(config.get("width", 640)),
                "height": int(config.get("height", 480)),
                "publish_camera_info": True,
                "queue_size": int((self._camera_config.get("queues", {}) or {}).get(
                    "ros2_queue_size", 5
                )),
                "qos_profile": "Sensor Data",
            }
            graph_path = f"{rig.camera_prim_path}/ROS2_Graph"
            bridge = IsaacSimRos2Bridge(graph_path)
            bridge.attach_cameras([record])
        except BaseException as create_error:
            cleanup_errors = []
            if bridge is not None:
                try:
                    bridge.destroy(stage)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            try:
                rig.destroy()
            except BaseException as exc:
                cleanup_errors.append(exc)
            if cleanup_errors:
                raise RuntimeError(
                    f"camera creation failed: {create_error}; cleanup also failed: "
                    + "; ".join(map(str, cleanup_errors))
                ) from create_error
            raise

        def set_target_tick_rate() -> None:
            attribute = stage.GetPrimAtPath(camera_prim_path).GetAttribute(
                "omni:sensor:tickRate"
            )
            if not attribute or not attribute.IsValid():
                raise RuntimeError(
                    f"camera sensor tick-rate attribute is missing: {camera_prim_path}"
                )
            attribute.Set(target_tick_rate_hz)

        bootstrap = _CameraPublisherBootstrap(
            bridge,
            name,
            [record],
            set_target_tick_rate=set_target_tick_rate if target_tick_rate_hz > 0.0 else None,
        )
        camera_prim_path = rig.camera_prim_path
        render_product_path = rig.render_product_path
        return _make_camera_runtime_resource(
            bridge=bridge,
            camera_key=name,
            rig=rig,
            stage=stage,
            camera_prim_path=camera_prim_path or "",
            render_product_path=render_product_path or "",
            ready=bootstrap.ready,
        )

    def _ensure_mid360_mount_and_root(self, stage: object, config: Mapping[str, Any]) -> str:
        sensor = config.get("sensor", {}) or {}
        mount_path = resolve_robot_mount_path(
            str(sensor["mount_prim_path"]), self.robot_prim_path
        )
        mount = stage.GetPrimAtPath(mount_path)
        if not mount or not mount.IsValid():
            parent_path = resolve_robot_mount_path(
                str(sensor.get("parent_prim", self.robot_prim_path)), self.robot_prim_path
            )
            if not _valid_prim(stage, parent_path):
                raise RuntimeError(f"MID360 parent prim does not exist: {parent_path}")
            apply_local_pose(
                stage,
                mount_path,
                LocalPose.from_mapping(sensor.get("fallback_mount_pose") or sensor.get("local_pose")),
            )
            self._mount_created = True
        self._mount_path = mount_path
        root_path = mount_path + "/MID360_Runtime"
        if not _valid_prim(stage, root_path):
            stage.DefinePrim(root_path, "Xform")
            self._sensor_root_created = True
        self._sensor_root_path = root_path
        return root_path

    def _set_gate(self, path: str, enabled: bool) -> None:
        import omni.graph.core as og

        attribute = og.Controller.attribute(path)
        if attribute is None:
            raise RuntimeError(f"sensor execution gate is missing: {path}")
        attribute.set(1 if enabled else 0)

    def _create_lidar(self) -> RuntimeSensorResource:
        from . import mid360

        stage = self._stage_getter()
        sensor_config = self._lidar_config.get("sensor", {}) or {}
        root_path = ""
        mount_path = ""
        graph_path = mid360.robot_lidar_graph_path(self.robot_prim_path)
        resource: dict[str, object] = {"stage": stage, "graph_path": graph_path}
        sensor_handle_count = len(mid360._LIDAR_SENSOR_HANDLES)
        product_handle_count = len(mid360._LIDAR_RENDER_PRODUCT_HANDLES)
        try:
            root_path = self._ensure_mid360_mount_and_root(stage, self._lidar_config)
            mount_path = self._mount_path
            resource.update(
                lidar_path=root_path + "/Lidar",
                root_path=root_path,
            )
            lidar_path = mid360.create_robot_mid360(
                stage,
                mount_path,
                str(sensor_config.get("frame_id", "livox_frame")),
                str(sensor_config.get("pointcloud_topic", "/openflex/livox_frame/lidar")),
                self._lidar_profile,
                graph_path=graph_path,
                sensor_root_path=root_path,
                transport=self._lidar_transport,
                mount_mode="parented",
                object_id_map=self._lidar_object_id_map,
                tick_rate_hz=self._lidar_tick_rate_hz,
                sensor_config=self._lidar_model_config,
                resource_sink=resource,
            )
        except BaseException as create_error:
            if "lidar_sensor" not in resource and len(mid360._LIDAR_SENSOR_HANDLES) > sensor_handle_count:
                resource["lidar_sensor"] = mid360._LIDAR_SENSOR_HANDLES[-1]
            if "render_product" not in resource and len(mid360._LIDAR_RENDER_PRODUCT_HANDLES) > product_handle_count:
                resource["render_product"] = mid360._LIDAR_RENDER_PRODUCT_HANDLES[-1]
            cleanup_error = None
            try:
                self._cleanup_lidar(stage, resource, graph_path, lidar_path="")
            except BaseException as exc:
                cleanup_error = exc
            finally:
                self._trim_mid360_handles(mid360, sensor_handle_count, product_handle_count)
                try:
                    self._release_mid360_root(stage)
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise RuntimeError(
                    f"MID360 creation failed: {create_error}; cleanup also failed: {cleanup_error}"
                ) from create_error
            raise

        self._active_mid360.add("lidar")
        updates = 0

        def ready() -> bool:
            nonlocal updates
            updates += 1
            if updates < 3:
                return False
            self._set_gate(f"{graph_path}/LidarEnableGate.inputs:step", True)
            return True

        def destroy() -> None:
            self._cleanup_lidar(stage, resource, graph_path, lidar_path)
            self._active_mid360.discard("lidar")
            self._release_mid360_root(stage)

        return RuntimeSensorResource(destroy=destroy, ready=ready)

    def _create_imu(self) -> RuntimeSensorResource:
        from .mid360 import create_physics_imu_graph

        stage = self._stage_getter()
        sensor_config = self._lidar_config.get("sensor", {}) or {}
        root_path = ""
        graph_path = ""
        imu_path = ""
        try:
            root_path = self._ensure_mid360_mount_and_root(stage, self._lidar_config)
            frame_id = str(sensor_config.get("frame_id", "livox_frame"))
            topic = str(sensor_config.get("imu_topic", "/livox/imu"))
            graph_path = root_path + "/IMU_ROS2_Graph"
            imu_path = create_physics_imu_graph(stage, root_path, frame_id, topic)
        except BaseException:
            try:
                self._cleanup_imu(stage, graph_path, imu_path)
            finally:
                self._release_mid360_root(stage)
            raise

        self._active_mid360.add("imu")
        updates = 0

        def ready() -> bool:
            nonlocal updates
            updates += 1
            if updates < 2:
                return False
            self._set_gate(f"{graph_path}/IMUEnableGate.inputs:step", True)
            return True

        def destroy() -> None:
            self._cleanup_imu(stage, graph_path, imu_path)
            self._active_mid360.discard("imu")
            self._release_mid360_root(stage)

        return RuntimeSensorResource(destroy=destroy, ready=ready)

    @staticmethod
    def _cleanup_imu(stage: object, graph_path: str, imu_path: str) -> None:
        from .mid360 import release_physics_imu_sensor

        release_physics_imu_sensor(imu_path)
        _remove_prim(stage, graph_path)
        _remove_prim(stage, imu_path)

    def _cleanup_lidar(
        self,
        stage: object,
        resource: Mapping[str, object],
        graph_path: str,
        lidar_path: str,
    ) -> None:
        from . import mid360

        errors = []
        try:
            _remove_prim(stage, graph_path)
        except BaseException as exc:
            errors.append(exc)
        handles = (
            resource.get("lidar_sensor"),
            resource.get("lidar"),
        )
        destroyed_handles: set[int] = set()
        for handle in handles:
            if handle is None or id(handle) in destroyed_handles:
                continue
            destroyed_handles.add(id(handle))
            try:
                _destroy_handle(handle)
            except BaseException as exc:
                errors.append(exc)
        render_product = resource.get("render_product")
        product_path = str(resource.get("render_product_path", ""))
        if render_product is not None:
            try:
                product_path = product_path or str(render_product.GetPath())
            except Exception:
                product_path = product_path or str(getattr(render_product, "path", ""))
            if resource.get("lidar_sensor") is None:
                try:
                    _destroy_handle(render_product)
                except BaseException as exc:
                    errors.append(exc)
        for path in (product_path, lidar_path, str(resource.get("lidar_path", ""))):
            try:
                _remove_prim(stage, path)
            except BaseException as exc:
                errors.append(exc)
        for key, value in (
            ("_LIDAR_SENSOR_HANDLES", resource.get("lidar_sensor")),
            ("_LIDAR_RENDER_PRODUCT_HANDLES", render_product),
        ):
            values = getattr(mid360, key, [])
            while value is not None and value in values:
                values.remove(value)
        if errors:
            raise RuntimeError("MID360 cleanup failed: " + "; ".join(map(str, errors)))

    @staticmethod
    def _trim_mid360_handles(module: object, sensors_at_start: int, products_at_start: int) -> None:
        del module._LIDAR_SENSOR_HANDLES[sensors_at_start:]
        del module._LIDAR_RENDER_PRODUCT_HANDLES[products_at_start:]

    def _release_mid360_root(self, stage: object) -> None:
        if self._active_mid360:
            return
        if self._sensor_root_created and self._sensor_root_path:
            _remove_prim(stage, self._sensor_root_path)
        if self._sensor_root_path:
            from . import mid360

            mid360._KINEMATIC_SENSOR_ROOTS[:] = [
                root for root in mid360._KINEMATIC_SENSOR_ROOTS
                if str(root.GetPath()) != self._sensor_root_path
            ]
        if self._mount_created and self._mount_path:
            _remove_prim(stage, self._mount_path)
        self._sensor_root_path = ""
        self._sensor_root_created = False
        self._mount_path = ""
        self._mount_created = False
