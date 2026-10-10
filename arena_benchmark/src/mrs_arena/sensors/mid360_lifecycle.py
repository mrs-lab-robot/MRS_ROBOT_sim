"""Arena-specific, on-demand MID360 and physics-IMU lifecycle."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

_ROS2_BRIDGE_EXTENSION = "isaacsim.ros2.bridge"
_SENSOR_KIT_ARGS = (
    "--/rtx/hydra/supportMultiTickRate=true",
    "--/renderer/raytracingMotion/enabled=true",
    "--/renderer/multiGpu/enabled=false",
    "--/exts/omni.replicator.srtx/enabled=false",
)


def configure_sensor_kit_args(kit_args: str | None) -> str:
    """Align Arena's RTX sensor renderer settings with the validated ROS app."""
    arguments = shlex.split(str(kit_args or ""))
    required_paths = {argument.split("=", 1)[0] for argument in _SENSOR_KIT_ARGS}
    arguments = [
        argument
        for argument in arguments
        if argument.split("=", 1)[0] not in required_paths
    ]
    arguments.extend(_SENSOR_KIT_ARGS)
    return shlex.join(arguments)


def resolve_robot_mount_path(robot_prim_path: str, configured_mount_path: str) -> str:
    """Map the canonical robot-relative mount suffix into an Arena env prim."""
    robot_path = str(robot_prim_path).rstrip("/")
    if not robot_path.startswith("/"):
        raise ValueError("robot prim path must be absolute")
    base_link_suffix = "/Geometry/base_link"
    if robot_path.endswith(base_link_suffix):
        robot_path = robot_path[: -len(base_link_suffix)]
    parts = [part for part in str(configured_mount_path).split("/") if part]
    try:
        geometry_index = parts.index("Geometry")
    except ValueError as error:
        raise ValueError("mount path must contain Geometry/base_link/Mid360Mount") from error
    suffix = parts[geometry_index:]
    if suffix[:2] != ["Geometry", "base_link"] or suffix[-1:] != ["Mid360Mount"]:
        raise ValueError("mount path must contain Geometry/base_link/Mid360Mount")
    return robot_path + "/" + "/".join(suffix)


def lidar_pointcloud_topic(sensor_config: dict[str, Any]) -> str:
    """Use the classic MID360 PointCloud2 topic as the relay input."""
    topic = str(sensor_config.get("pointcloud_topic", "/openflex/livox_frame/lidar")).strip()
    if not topic.startswith("/"):
        raise ValueError("MID360 PointCloud2 topic must be absolute")
    return topic


def _resolve_extension_manager(app: Any, *, kit_app_getter: Any | None = None) -> Any:
    """Get the Kit extension manager from either app wrapper or Kit singleton."""
    getter = getattr(app, "get_extension_manager", None)
    if callable(getter):
        return getter()
    if kit_app_getter is None:
        from omni.kit.app import get_app

        kit_app_getter = get_app
    kit_app = kit_app_getter()
    getter = getattr(kit_app, "get_extension_manager", None)
    if not callable(getter):
        raise RuntimeError("Kit application does not expose an extension manager")
    return getter()


def _valid_prim(stage: Any, path: str) -> bool:
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def _remove_prim(stage: Any, path: str) -> None:
    if not path or not _valid_prim(stage, path):
        return
    from pxr import Sdf

    stage.RemovePrim(Sdf.Path(path))


def _destroy_handle(value: Any) -> None:
    if value is None:
        return
    destroy = getattr(value, "destroy", None)
    if callable(destroy):
        destroy()
        return
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


class ArenaMid360Runtime:
    """Create MID360/IMU resources only when requested through the sensor API."""

    def __init__(
        self,
        *,
        app: Any,
        robot_prim_path: str,
        repository_root: str | Path,
    ) -> None:
        self._app = app
        self._robot_prim_path = str(robot_prim_path).rstrip("/")
        self._repository_root = Path(repository_root).expanduser().resolve()
        from mrs_robot_lab.assets.asset_resolver import AssetResolver

        mount_config_path = AssetResolver(self._repository_root).resolve("openflex_lidar_mounts")
        import yaml

        with mount_config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        self._sensor_config = dict(config.get("sensor", {}) or {})
        if not self._sensor_config:
            raise ValueError(f"MID360 mount configuration is empty: {mount_config_path}")
        self._extension_manager: Any | None = None
        self._bridge_ready_updates = 0
        self._mid360_module: Any | None = None
        self._root_path = ""
        self._mount_path = ""
        self._root_created = False
        self._prepared_stage: Any | None = None
        self._active_sensors: set[str] = set()
        self._prepared_lidar: dict[str, Any] | None = None
        self._prepared_imu: dict[str, Any] | None = None
        self._lidar_relay: Any | None = None
        self._preparation_error = ""

    def prepare_sensors(self) -> None:
        """Author gated MID360 and IMU resources before Arena's first env.step."""
        if self._prepared_lidar is not None and self._prepared_imu is not None:
            return
        if self._preparation_error:
            raise RuntimeError(f"sensor startup preparation failed: {self._preparation_error}")

        try:
            self._request_ros2_bridge()
            deadline = time.monotonic() + 10.0
            while self._bridge_ready_updates < 3:
                self._app.update()
                if self._extension_is_enabled():
                    self._bridge_ready_updates += 1
                else:
                    self._bridge_ready_updates = 0
                if time.monotonic() >= deadline:
                    raise TimeoutError("isaacsim.ros2.bridge did not become ready")

            stage = self._stage()
            self._prepared_stage = stage
            root_path = self._ensure_sensor_root(stage)
            mid360 = self._mid360()
            profile = str(
                self._sensor_config.get("performance_profile")
                or self._sensor_config.get("default_profile")
                or "MID360_PERFORMANCE"
            )
            frame_id = str(self._sensor_config.get("frame_id", "livox_frame"))
            source_topic = lidar_pointcloud_topic(self._sensor_config)
            lidar_graph_path = root_path + "/Lidar_ROS2_Graph"
            lidar_resource: dict[str, Any] = {
                "stage": stage,
                "root_path": root_path,
                "graph_path": lidar_graph_path,
                "lidar_path": root_path + "/Lidar",
                "sensor_count": len(mid360._LIDAR_SENSOR_HANDLES),
                "product_count": len(mid360._LIDAR_RENDER_PRODUCT_HANDLES),
            }
            self._prepared_lidar = lidar_resource
            try:
                mid360.create_robot_mid360(
                    stage,
                    self._mount_path,
                    frame_id,
                    source_topic,
                    profile,
                    graph_path=lidar_graph_path,
                    sensor_root_path=root_path,
                    transport="helper",
                    mount_mode="parented",
                    object_id_map=False,
                    tick_rate_hz=float(self._sensor_config.get("tick_rate_hz", 10.0)),
                    sensor_config=str(
                        self._sensor_config.get("default_profile", "MID360_APPROX")
                    ),
                    resource_sink=lidar_resource,
                )
                self._set_gate(lidar_graph_path + "/LidarEnableGate.inputs:step", False)
            except BaseException:
                self._recover_partial_lidar_handles(
                    lidar_resource,
                    mid360,
                    lidar_resource["sensor_count"],
                    lidar_resource["product_count"],
                )
                self._prepared_lidar = None
                raise

            imu_graph_path = root_path + "/IMU_ROS2_Graph"
            imu_resource: dict[str, Any] = {
                "stage": stage,
                "root_path": root_path,
                "graph_path": imu_graph_path,
                "imu_path": root_path + "/IMU",
            }
            self._prepared_imu = imu_resource
            try:
                imu_resource["imu_path"] = mid360.create_physics_imu_graph(
                    stage,
                    root_path,
                    frame_id,
                    str(self._sensor_config.get("imu_topic", "/livox/imu")),
                )
                self._set_gate(imu_graph_path + "/IMUEnableGate.inputs:step", False)
            except BaseException:
                self._cleanup_imu(imu_resource)
                self._prepared_imu = None
                raise

            print(
                f"ARENA_SENSOR_RESOURCES_PREPARED lidar={lidar_resource.get('lidar_path', '')} "
                f"imu={imu_resource['imu_path']} gates=closed",
                flush=True,
            )
        except BaseException as error:
            self._preparation_error = str(error)
            try:
                self._cleanup_prepared_resources()
            except BaseException as cleanup_error:
                self._preparation_error += f"; cleanup failed: {cleanup_error}"
            raise

    def create_lidar(self) -> SensorResource:
        from mrs_robot_lab.recorders.arena_sensor_api import SensorResource

        if self._preparation_error:
            raise RuntimeError(f"MID360 unavailable: {self._preparation_error}")
        if self._prepared_lidar is None:
            raise RuntimeError("MID360 resources were not prepared before simulation startup")
        resource = self._prepared_lidar
        gate_open = False

        def ready() -> bool:
            nonlocal gate_open
            if self._lidar_relay is None:
                source_topic = lidar_pointcloud_topic(self._sensor_config)
                frame_id = str(self._sensor_config.get("frame_id", "livox_frame"))
                self._lidar_relay = self._start_livox_relay(source_topic, frame_id)
            if not self._lidar_relay.is_ready():
                return False
            if not gate_open:
                self._set_gate(resource["graph_path"] + "/LidarEnableGate.inputs:step", True)
                gate_open = True
            return self._lidar_relay.has_data()

        def destroy() -> None:
            nonlocal gate_open
            if gate_open:
                self._set_gate(resource["graph_path"] + "/LidarEnableGate.inputs:step", False)
                gate_open = False
            if self._lidar_relay is not None:
                self._lidar_relay.stop()
                self._lidar_relay = None
            self._active_sensors.discard("lidar")

        return SensorResource(
            destroy=destroy,
            ready=ready,
            health=lambda: self._lidar_relay is not None and self._lidar_relay.is_healthy(),
        )

    def create_imu(self) -> SensorResource:
        from mrs_robot_lab.recorders.arena_sensor_api import SensorResource

        if self._preparation_error:
            raise RuntimeError(f"MID360 IMU unavailable: {self._preparation_error}")
        if self._prepared_imu is None:
            raise RuntimeError("IMU resources were not prepared before simulation startup")
        resource = self._prepared_imu
        gate_open = False

        def ready() -> bool:
            nonlocal gate_open
            if not self._extension_is_enabled():
                raise RuntimeError("isaacsim.ros2.bridge was disabled after startup")
            if not gate_open:
                self._set_gate(resource["graph_path"] + "/IMUEnableGate.inputs:step", True)
                gate_open = True
                self._active_sensors.add("imu")
            return gate_open and self._extension_is_enabled()

        def destroy() -> None:
            nonlocal gate_open
            if gate_open:
                self._set_gate(resource["graph_path"] + "/IMUEnableGate.inputs:step", False)
                gate_open = False
            self._active_sensors.discard("imu")

        return SensorResource(
            destroy=destroy,
            ready=ready,
            health=self._extension_is_enabled,
        )

    def shutdown(self) -> None:
        """Stop relay and remove all startup-owned graphs and render products."""
        self._cleanup_prepared_resources()

    def _cleanup_prepared_resources(self) -> None:
        errors: list[BaseException] = []
        lidar = self._prepared_lidar
        imu = self._prepared_imu
        if lidar is not None:
            try:
                self._set_gate(lidar["graph_path"] + "/LidarEnableGate.inputs:step", False)
            except BaseException as error:
                errors.append(error)
        if imu is not None:
            try:
                self._set_gate(imu["graph_path"] + "/IMUEnableGate.inputs:step", False)
            except BaseException as error:
                errors.append(error)
        if self._lidar_relay is not None:
            try:
                self._lidar_relay.stop()
            except BaseException as error:
                errors.append(error)
            self._lidar_relay = None
        self._active_sensors.clear()
        if imu is not None:
            try:
                self._cleanup_imu(imu)
            except BaseException as error:
                errors.append(error)
        if lidar is not None:
            try:
                self._cleanup_lidar(lidar)
            except BaseException as error:
                errors.append(error)
        stage = (lidar or imu or {}).get("stage") or self._prepared_stage
        if stage is not None:
            try:
                self._release_sensor_root(stage)
            except BaseException as error:
                errors.append(error)
        self._prepared_lidar = None
        self._prepared_imu = None
        self._prepared_stage = None
        if errors:
            raise RuntimeError("sensor shutdown cleanup failed: " + "; ".join(map(str, errors)))

    def _stage(self) -> Any:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Arena USD stage is not available")
        return stage

    def _ensure_sensor_root(self, stage: Any) -> str:
        if not self._mount_path:
            configured_mount = str(
                self._sensor_config.get(
                    "mount_prim_path", "/World/OpenFlex/Geometry/base_link/Mid360Mount"
                )
            )
            self._mount_path = resolve_robot_mount_path(
                self._robot_prim_path, configured_mount
            )
        if not _valid_prim(stage, self._mount_path):
            raise RuntimeError(f"MID360 mount prim does not exist: {self._mount_path}")
        if not self._root_path:
            from pxr import Sdf, UsdGeom

            self._root_path = self._mount_path + "/MID360_Runtime"
            self._root_created = not _valid_prim(stage, self._root_path)
            if self._root_created:
                UsdGeom.Xform.Define(stage, Sdf.Path(self._root_path))
        return self._root_path

    def _request_ros2_bridge(self) -> None:
        if self._extension_manager is None:
            self._extension_manager = _resolve_extension_manager(self._app)
        manager = self._extension_manager
        enabled = getattr(manager, "is_extension_enabled", None)
        if callable(enabled) and enabled(_ROS2_BRIDGE_EXTENSION):
            return
        enable_immediate = getattr(manager, "set_extension_enabled_immediate", None)
        if callable(enable_immediate):
            enable_immediate(_ROS2_BRIDGE_EXTENSION, True)
            return
        enable = getattr(manager, "enable_extension", None)
        if callable(enable):
            enable(_ROS2_BRIDGE_EXTENSION)
            return
        raise RuntimeError("Isaac Sim extension manager cannot enable isaacsim.ros2.bridge")

    def _extension_is_enabled(self) -> bool:
        if self._extension_manager is None:
            return False
        enabled = getattr(self._extension_manager, "is_extension_enabled", None)
        return bool(enabled(_ROS2_BRIDGE_EXTENSION)) if callable(enabled) else True

    def _ros2_bridge_enabled(self) -> bool:
        if not self._extension_is_enabled():
            self._bridge_ready_updates = 0
            return False
        self._bridge_ready_updates += 1
        return self._bridge_ready_updates >= 3

    def _mid360(self) -> Any:
        if self._mid360_module is not None:
            return self._mid360_module
        module_path = (
            self._repository_root
            / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_sensors/"
            / "openflex_isaac_sensors/mid360.py"
        )
        spec = importlib.util.spec_from_file_location(
            "mrs_arena_mid360_helpers", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load MID360 runtime helpers: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._mid360_module = module
        return module

    def _start_livox_relay(self, source_topic: str, frame_id: str) -> Any:
        relay_path = self._repository_root / "arena_benchmark/scripts/livox_ros2_relay.py"
        module_name = "mrs_arena_livox_relay_process"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, relay_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load Livox relay process manager: {relay_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        ros_setup = Path(os.environ.get("ARENA_ROS_SETUP", "/opt/ros/humble/setup.bash"))
        default_workspace = self._repository_root.parent.parent / "openflex_all/openflex_ws"
        workspace = Path(
            os.environ.get("OPENFLEX_ROS2_WORKSPACE", str(default_workspace))
        ).expanduser()
        workspace_setup = workspace / "install/setup.bash"
        python = Path(os.environ.get("ARENA_ROS_SYSTEM_PYTHON", "/usr/bin/python3")).expanduser()
        if not ros_setup.is_file():
            raise FileNotFoundError(f"ROS 2 setup script not found: {ros_setup}")
        if not workspace_setup.is_file():
            raise FileNotFoundError(
                f"OpenFlex ROS workspace setup not found: {workspace_setup}; "
                "set OPENFLEX_ROS2_WORKSPACE to its workspace root"
            )
        if not python.is_file():
            raise FileNotFoundError(f"system ROS 2 Python not found: {python}")

        command = [
            str(python),
            str(relay_path),
            "--source-topic",
            source_topic,
            "--output-topic",
            str(self._sensor_config.get("livox_topic", "/livox/lidar")),
            "--frame-id",
            frame_id,
        ]
        shell = (
            f"set -e; source {shlex.quote(str(ros_setup))}; "
            f"source {shlex.quote(str(workspace_setup))}; exec {shlex.join(command)}"
        )
        env = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONEXECUTABLE",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
        ):
            env.pop(name, None)
        env.setdefault("ROS_DOMAIN_ID", "0")
        return module.ManagedLivoxRelay.start(
            ["/bin/bash", "--noprofile", "--norc", "-c", shell],
            cwd=str(self._repository_root),
            env=env,
        )

    def _set_gate(self, path: str, enabled: bool) -> None:
        import omni.graph.core as og

        attribute = og.Controller.attribute(path)
        if attribute is None:
            raise RuntimeError(f"sensor execution gate is missing: {path}")
        attribute.set(1 if enabled else 0)

    def _cleanup_lidar(self, resource: dict[str, Any]) -> None:
        mid360 = self._mid360()
        stage = resource.get("stage")
        if stage is None:
            return
        sensor_count = resource.pop("sensor_count", None)
        product_count = resource.pop("product_count", None)
        errors = []
        try:
            _remove_prim(stage, str(resource.get("graph_path", "")))
        except BaseException as error:
            errors.append(error)

        handles = (resource.get("lidar_sensor"), resource.get("lidar"))
        destroyed: set[int] = set()
        for handle in handles:
            if handle is None or id(handle) in destroyed:
                continue
            destroyed.add(id(handle))
            try:
                _destroy_handle(handle)
            except BaseException as error:
                errors.append(error)

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
                except BaseException as error:
                    errors.append(error)
        paths = {
            product_path,
            str(resource.get("lidar_path", "")),
            str(resource.get("root_path", "")) + "/Lidar",
        }
        for path in paths:
            try:
                _remove_prim(stage, path)
            except BaseException as error:
                errors.append(error)

        for key, value in (
            ("_LIDAR_SENSOR_HANDLES", resource.get("lidar_sensor")),
            ("_LIDAR_RENDER_PRODUCT_HANDLES", render_product),
        ):
            values = getattr(mid360, key, [])
            while value is not None and value in values:
                values.remove(value)
        if sensor_count is not None and product_count is not None:
            del mid360._LIDAR_SENSOR_HANDLES[sensor_count:]
            del mid360._LIDAR_RENDER_PRODUCT_HANDLES[product_count:]
        if errors:
            raise RuntimeError("MID360 cleanup failed: " + "; ".join(map(str, errors)))

    def _recover_partial_lidar_handles(
        self,
        resource: dict[str, Any],
        mid360: Any,
        sensor_count: int,
        product_count: int,
    ) -> None:
        resource["sensor_count"] = sensor_count
        resource["product_count"] = product_count
        if "lidar_sensor" not in resource and len(mid360._LIDAR_SENSOR_HANDLES) > sensor_count:
            resource["lidar_sensor"] = mid360._LIDAR_SENSOR_HANDLES[-1]
        if "render_product" not in resource and len(mid360._LIDAR_RENDER_PRODUCT_HANDLES) > product_count:
            resource["render_product"] = mid360._LIDAR_RENDER_PRODUCT_HANDLES[-1]
        try:
            self._cleanup_lidar(resource)
        finally:
            self._active_sensors.discard("lidar")

    def _cleanup_imu(self, resource: dict[str, Any]) -> None:
        stage = resource.get("stage")
        if stage is None:
            return
        mid360 = self._mid360()
        errors = []
        try:
            mid360.release_physics_imu_sensor(str(resource.get("imu_path", "")))
        except BaseException as error:
            errors.append(error)
        for path in (
            str(resource.get("graph_path", "")),
            str(resource.get("imu_path", "")),
        ):
            try:
                _remove_prim(stage, path)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise RuntimeError("IMU cleanup failed: " + "; ".join(map(str, errors)))

    def _release_sensor_root(self, stage: Any) -> None:
        if self._active_sensors or not self._root_path:
            return
        root_path = self._root_path
        if self._root_created:
            _remove_prim(stage, root_path)
        if self._mid360_module is not None:
            roots = getattr(self._mid360_module, "_KINEMATIC_SENSOR_ROOTS", [])
            self._mid360_module._KINEMATIC_SENSOR_ROOTS[:] = [
                root
                for root in roots
                if not callable(getattr(root, "GetPath", None))
                or str(root.GetPath()) != root_path
            ]
        self._root_path = ""
        self._root_created = False
