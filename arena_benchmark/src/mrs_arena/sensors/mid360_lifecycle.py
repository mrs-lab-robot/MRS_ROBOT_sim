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
_RTX_SENSOR_EXTENSION = "isaacsim.sensors.experimental.rtx"
_SENSOR_KIT_ARGS = (
    # Stable-version value: the 2026-09-16 measured chain, the 2026-10-14
    # direct probe, and the standalone performance harness all run with
    # multi-tick-rate support enabled (RTX sensors tick at their own rate).
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
    for extension in (_ROS2_BRIDGE_EXTENSION, _RTX_SENSOR_EXTENSION):
        if not any(
            arguments[index] == "--enable" and arguments[index + 1] == extension
            for index in range(len(arguments) - 1)
        ):
            arguments.extend(("--enable", extension))
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
        self._gmo_report_count = 0
        self._extract_probe_annotators: dict[str, Any] = {}
        self._shadow_runtime: Any | None = None
        self._last_gmo_signature: tuple[int, int] | None = None
        self._gmo_stale_polls = 0
        self._gmo_resubscribe_count = 0
        self._preparation_error = ""
        # Direct-poll transport state (the 2026-10-14 direct probe port):
        # rclpy publisher inside the Kit process + per-frame GMO polling.
        self._direct_node: Any | None = None
        self._direct_publisher: Any | None = None
        self._direct_publish_count = 0
        self._last_direct_signature: tuple[int, int] | None = None
        self._direct_poll_error_count = 0

    def _requested_transport(self) -> str:
        """Transport requested for this session; direct_poll is the default.

        ``ARENA_MID360_TRANSPORT=helper`` restores the historical ROS2 helper
        graph behavior for A/B reproduction of the Warp CUDA 700 failure.
        """
        requested = os.environ.get("ARENA_MID360_TRANSPORT", "").strip().lower()
        return requested if requested in {"direct_poll", "helper", "native"} else "direct_poll"

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
                "frame_id": frame_id,
                "sensor_count": len(mid360._LIDAR_SENSOR_HANDLES),
                "product_count": len(mid360._LIDAR_RENDER_PRODUCT_HANDLES),
            }
            self._prepared_lidar = lidar_resource
            requested_transport = self._requested_transport()
            try:
                mid360.create_robot_mid360(
                    stage,
                    self._mount_path,
                    frame_id,
                    source_topic,
                    profile,
                    graph_path=lidar_graph_path,
                    sensor_root_path=root_path,
                    transport=requested_transport,
                    mount_mode="parented",
                    object_id_map=False,
                    tick_rate_hz=float(self._sensor_config.get("tick_rate_hz", 10.0)),
                    sensor_config=str(
                        self._sensor_config.get("default_profile", "MID360_APPROX")
                    ),
                    resource_sink=lidar_resource,
                )
                transport_used = str(lidar_resource.get("transport_used", requested_transport))
                if transport_used != "direct_poll":
                    # Only the ROS2 helper graph path owns a LidarEnableGate.
                    # The native registry writer and direct-poll paths have no
                    # graph, so the gate attribute does not exist for them.
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

            # Keep the RTX sensor render mode cleared across the launch window;
            # the tiled cameras re-set it at their creation below.
            self._hold_rtx_sensor_render_mode()

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
        transport_used = str(resource.get("transport_used", "helper"))
        direct_poll = transport_used == "direct_poll"

        def ready() -> bool:
            nonlocal gate_open
            if self._lidar_relay is None:
                source_topic = lidar_pointcloud_topic(self._sensor_config)
                frame_id = str(self._sensor_config.get("frame_id", "livox_frame"))
                self._lidar_relay = self._start_livox_relay(source_topic, frame_id)
            if not self._lidar_relay.is_ready():
                return False
            self._hold_rtx_sensor_render_mode(resource)
            if direct_poll:
                # Probe-style path: no graph, no gate. Poll the GMO buffer and
                # publish PointCloud2 from this process every frame until the
                # relay confirms data flowing on /openflex/livox_frame/lidar.
                self._ensure_direct_publisher()
                self._poll_and_publish(resource)
                return self._lidar_relay.has_data()
            if transport_used == "helper" and not gate_open:
                # Helper path publishes only while the gate is open, so the
                # PointCloud2 stream exists only for an actively requested lidar.
                self._set_gate(resource["graph_path"] + "/LidarEnableGate.inputs:step", True)
                gate_open = True
                self._active_sensors.add("lidar")
            if self._lidar_relay.has_data():
                return True
            self._report_gmo_state(resource)
            self._resubscribe_gmo(resource)
            return False

        def health() -> bool:
            if direct_poll:
                try:
                    self._poll_and_publish(resource)
                except BaseException as error:
                    self._direct_poll_error_count += 1
                    if self._direct_poll_error_count % 30 == 1:
                        print(
                            f"ARENA_MID360_DIRECT_POLL_ERROR n={self._direct_poll_error_count} "
                            f"{type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
                    return False
            return self._lidar_relay is not None and self._lidar_relay.is_healthy()

        def destroy() -> None:
            nonlocal gate_open
            if transport_used == "helper" and gate_open:
                try:
                    self._set_gate(resource["graph_path"] + "/LidarEnableGate.inputs:step", False)
                except BaseException as error:
                    print(
                        f"ARENA_MID360_GATE_CLOSE_ERROR {type(error).__name__}: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                gate_open = False
            if transport_used == "native":
                lidar_sensor = resource.get("lidar_sensor")
                detach = getattr(lidar_sensor, "detach_writer", None)
                writer_name = str(resource.get("writer_name", "RtxLidarROS2PublishPointCloud"))
                if callable(detach):
                    try:
                        detach(writer_name)
                    except BaseException as error:
                        print(
                            f"ARENA_MID360_WRITER_DETACH_ERROR {type(error).__name__}: {error}",
                            file=sys.stderr,
                            flush=True,
                        )
            if self._lidar_relay is not None:
                self._lidar_relay.stop()
                self._lidar_relay = None
            self._active_sensors.discard("lidar")

        return SensorResource(
            destroy=destroy,
            ready=ready,
            health=health,
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

    def _report_gmo_state(self, resource: dict[str, Any]) -> None:
        """Report once whether the GMO annotator produces data while the relay does not.

        Distinguishes \"no GMO data\" (annotator/gate never ticks) from \"GMO data
        exists but the native writer did not forward it\".
        """
        self._gmo_report_count += 1
        if self._gmo_report_count not in (10, 30, 60):
            return
        stage = getattr(self._prepared_lidar or {}, "get", lambda key: None)("stage")
        try:
            lidar_sensor = resource.get("lidar_sensor")
            get_data = getattr(lidar_sensor, "get_data", None)
            has_gmo = False
            data = None
            if callable(get_data):
                data, _ = get_data("generic-model-output")
                has_gmo = data is not None
            gmo_brief = ""
            if isinstance(data, dict):
                gmo_brief = " keys=" + str(sorted(data.keys()))
                points = data.get("data")
                if points is not None:
                    shape = getattr(points, "shape", None)
                    dtype = getattr(points, "dtype", None)
                    size = getattr(points, "size", 0)
                    numpy_points = None
                    try:
                        numpy_points = getattr(points, "numpy", None)
                        numpy_points = numpy_points() if callable(numpy_points) else None
                    except BaseException:
                        numpy_points = None
                    nonzero = None
                    try:
                        import numpy as np

                        nonzero = int(np.count_nonzero(numpy_points)) if numpy_points is not None else None
                    except BaseException:
                        nonzero = None
                    gmo_brief += f" shape={shape} dtype={dtype} size={size} nonzero={nonzero} type={type(points).__name__}"
            else:
                shape = getattr(data, "shape", None)
                dtype = getattr(data, "dtype", None)
                size = getattr(data, "size", 0)
                gmo_brief = f" type={type(data).__name__} shape={shape} dtype={dtype} size={size}"
            print(
                f"ARENA_MID360_GMO_STATE poll={self._gmo_report_count} has_data={has_gmo}{gmo_brief}",
                flush=True,
            )
            modality_brief = ""
            import numpy as np

            try:
                from isaacsim.sensors.experimental.rtx.generic_model_output import (
                    getModelOutputFromBuffer,
                )

                numpy_buffer = getattr(data, "numpy", None)
                numpy_buffer = numpy_buffer() if callable(numpy_buffer) else None
                if numpy_buffer is not None:
                    parsed = getModelOutputFromBuffer(numpy_buffer)
                    points_x = getattr(parsed, "x", None)
                    print(
                        f"ARENA_MID360_GMO_DECODE numElements={parsed.numElements} "
                        f"outputType={parsed.outputType} scanComplete={parsed.scanComplete} "
                        f"rows={parsed.numRows} cols={parsed.numCols} "
                        f"tsNs={parsed.timestampNs} sizeB={parsed.sizeInBytes} "
                        f"x_nonzero={0 if points_x is None else int(np.count_nonzero(points_x))}",
                        flush=True,
                    )
            except BaseException as error:
                print(
                    f"ARENA_MID360_GMO_DECODE error={type(error).__name__}: {error}",
                    flush=True,
                )
        except BaseException as error:
            print(
                f"ARENA_MID360_GMO_STATE poll={self._gmo_report_count} error={type(error).__name__}: {error}",
                flush=True,
            )
        if os.environ.get("ARENA_MID360_PROBE", "") != "1":
            return
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            current_time = timeline.get_current_time()
            print(
                "ARENA_MID360_TIMELINE playing="
                f"{timeline.is_playing()} timeCode={current_time}",
                flush=True,
            )
        except BaseException as error:
            print(f"ARENA_MID360_TIMELINE error={type(error).__name__}: {error}", flush=True)
        if stage is not None:
            self._probe_replicator_graph(stage)
        self._probe_extract_annotator(resource)
        self._probe_shadow_sensor(resource, stage)

    def _probe_shadow_sensor(self, resource: dict[str, Any], stage: Any) -> None:
        """Compare a /World-level twin sensor against the mount-level sensor.

        Distinguishes "this process's RTX sensor pipeline is broken" from
        "the sensor placement under the robot mount breaks ray returns".
        """
        try:
            import numpy as np
            from isaacsim.sensors.experimental.rtx.generic_model_output import (
                getModelOutputFromBuffer,
            )

            if self._shadow_runtime is None:
                if stage is None:
                    return
                mid360 = self._mid360()
                lidar = mid360._create_lidar(
                    stage,
                    "/World/MID360_Probe/Lidar",
                    tick_rate_hz=10.0,
                    sensor_config="MID360_APPROX",
                )
                _render_product_path, runtime = mid360._create_lidar_render_product(
                    lidar, "probe_frame"
                )
                self._shadow_runtime = runtime
                print("ARENA_MID360_SHADOW created", flush=True)
                return
            data, _info = self._shadow_runtime.get_data("generic-model-output")
            numpy_buffer = getattr(data, "numpy", None)
            numpy_buffer = numpy_buffer() if callable(numpy_buffer) else None
            if numpy_buffer is None:
                print("ARENA_MID360_SHADOW no buffer", flush=True)
                return
            gmo = getModelOutputFromBuffer(numpy_buffer)
            points_x = getattr(gmo, "x", None)
            print(
                f"ARENA_MID360_SHADOW numElements={gmo.numElements} "
                f"x_nonzero={0 if points_x is None else int(np.count_nonzero(points_x))} "
                f"scanComplete={gmo.scanComplete}",
                flush=True,
            )
        except BaseException as error:
            print(
                f"ARENA_MID360_SHADOW error={type(error).__name__}: {error}",
                flush=True,
            )

    def _probe_replicator_graph(self, stage: Any) -> None:
        """List Replicator graph prims and their step attributes for one poll."""
        try:
            import carb

            # Probe: does clearing the RTX-sensor render mode revive the sensor?
            print(
                "ARENA_MID360_RTX_SENSORS_FLAG "
                f"{carb.settings.get_settings().get('/isaaclab/render/rtx_sensors')}",
                flush=True,
            )
        except BaseException:
            pass
        try:
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith("/Render"):
                    continue
                if not any(
                    token in path
                    for token in ("Replicator", "SimulationGate", "PointCloud", "rtx")
                ):
                    continue
                hits = []
                for name in prim.GetPropertyNames():
                    if "step" in name or "Input" in name:
                        attribute = prim.GetAttribute(name)
                        hits.append(f"{name}={attribute.Get()!r}")
                print(
                    f"ARENA_MID360_GRAPH_PROBEP prim={path} type={prim.GetTypeName()} "
                    + " ".join(hits[:12]),
                    flush=True,
                )
        except BaseException as error:
            print(f"ARENA_MID360_GRAPH_PROBEP error={type(error).__name__}: {error}", flush=True)

    def _probe_extract_annotator(self, resource: dict[str, Any]) -> None:
        """Read the IsaacExtractRTXSensorPointCloud buffer through a fresh annotator."""
        render_product_path = str(resource.get("render_product_path", ""))
        if not render_product_path:
            return
        try:
            import omni.replicator.core as rep

            attribute_name = "extract_probe_" + str(resource.get("lidar_path", "lidar"))[-8:]
            annotator = self._extract_probe_annotators.get(attribute_name)
            if annotator is None:
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "IsaacExtractRTXSensorPointCloud", device="cpu"
                )
                annotator.attach([render_product_path])
                self._extract_probe_annotators[attribute_name] = annotator
                print("ARENA_MID360_EXTRACT_PROBE attached", flush=True)
                return
            data = annotator.get_data()
            brief = ""
            has_points = False
            if isinstance(data, dict):
                brief = " keys=" + str(sorted(data.keys()))
                points = data.get("data")
                if isinstance(points, dict):
                    brief += " data_keys=" + str(sorted(points.keys()))
                    has_points = True
                elif points is not None:
                    shape = getattr(points, "shape", None)
                    dtype = getattr(points, "dtype", None)
                    has_points = getattr(points, "size", 0) > 0
                    brief += f" shape={shape} dtype={dtype}"
            else:
                has_points = getattr(data, "size", 0) > 0
            print(
                f"ARENA_MID360_EXTRACT_PROBE has_data={has_points}{brief}",
                flush=True,
            )
        except BaseException as error:
            print(
                f"ARENA_MID360_EXTRACT_PROBE error={type(error).__name__}: {error}",
                flush=True,
            )

    def _ensure_direct_publisher(self) -> None:
        """Create the in-process rclpy PointCloud2 publisher once.

        The Isaac Sim venv ships rclpy and the ROS 2 message packages (the
        2026-10-14 direct probe imported both successfully after enabling
        ``isaacsim.ros2.bridge``). Publishing from this process uses the same
        RMW the historical ROS2RtxLidarHelper bridge node used, so the relay
        subprocess discovers it identically.
        """
        if self._direct_publisher is not None:
            return
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import PointCloud2

        if not rclpy.ok():
            rclpy.init()
        self._direct_node = Node("arena_mid360_direct_publisher")
        topic = lidar_pointcloud_topic(self._sensor_config)
        self._direct_publisher = self._direct_node.create_publisher(
            PointCloud2, topic, qos_profile_sensor_data
        )
        print(f"ARENA_MID360_DIRECT_PUBLISHER topic={topic}", flush=True)

    def _poll_and_publish(self, resource: dict[str, Any]) -> bool:
        """Poll the GMO buffer once and publish a PointCloud2 when it advanced.

        This is the exact read path validated by the 2026-10-14 direct probe
        (first non-empty GMO at update 6, numElements 481-497): get_data ->
        numpy buffer -> getModelOutputFromBuffer. The (timestampNs,
        numElements) signature deduplicates unchanged frames so a 10-20 Hz
        sensor is not republished at the render rate.
        """
        if self._direct_publisher is None:
            return False
        lidar_sensor = resource.get("lidar_sensor")
        get_data = getattr(lidar_sensor, "get_data", None)
        if not callable(get_data):
            return False
        data, _info = get_data("generic-model-output")
        buffer = getattr(data, "numpy", None)
        buffer = buffer() if callable(buffer) else None
        if buffer is None:
            return False
        from isaacsim.sensors.experimental.rtx.generic_model_output import (
            getModelOutputFromBuffer,
        )

        parsed = getModelOutputFromBuffer(buffer)
        num_elements = int(getattr(parsed, "numElements", 0) or 0)
        stamp_ns = int(getattr(parsed, "timestampNs", 0) or 0)
        if num_elements <= 0:
            return False
        signature = (stamp_ns, num_elements)
        if signature == self._last_direct_signature:
            return True
        self._last_direct_signature = signature
        mid360 = self._mid360()
        packed, pack_stamp, count = mid360.gmo_pointcloud2_buffer(parsed)
        if count <= 0:
            return False
        from sensor_msgs.msg import PointField, PointCloud2

        message = PointCloud2()
        stamp_sec, stamp_nsec = divmod(max(0, int(pack_stamp)), 1_000_000_000)
        message.header.stamp.sec = int(stamp_sec)
        message.header.stamp.nanosec = int(stamp_nsec)
        message.header.frame_id = str(self._sensor_config.get("frame_id", "livox_frame"))
        message.height = 1
        message.width = count
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="offset_time", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="line", offset=20, datatype=PointField.UINT8, count=1),
        ]
        message.point_step = int(mid360._POINTCLOUD2_POINT_STEP)
        message.row_step = message.point_step * count
        message.is_bigendian = False
        message.is_dense = True
        message.data = packed
        self._direct_publisher.publish(message)
        self._direct_publish_count += 1
        if self._direct_publish_count % 50 == 1:
            print(
                f"ARENA_MID360_DIRECT_PUBLISH count={self._direct_publish_count} "
                f"points={count} tsNs={pack_stamp}",
                flush=True,
            )
        return True

    def _shutdown_direct_publisher(self) -> None:
        """Tear down the rclpy publisher/node created for direct polling."""
        if self._direct_node is None and self._direct_publisher is None:
            return
        try:
            if self._direct_node is not None:
                self._direct_node.destroy_node()
        except BaseException as error:
            print(
                f"ARENA_MID360_DIRECT_PUBLISHER_DESTROY_ERROR {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        self._direct_node = None
        self._direct_publisher = None
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except BaseException as error:
            print(
                f"ARENA_MID360_RCLPY_SHUTDOWN_ERROR {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

    def _hold_rtx_sensor_render_mode(self, resource: dict[str, Any] | None = None) -> None:
        """Clear ``/isaaclab/render/rtx_sensors`` when the lidar is activated.

        The tiled cameras set this flag at creation; while it is True the RTX
        sensor pipeline kills the standalone lidar render product's GMO stream
        (verified 2026-10-13, run kit_20261013_235x: with the flag True at
        creation the sensor delivers one frame then numElements=0 forever).
        The flag is only re-read when cameras are created, so clearing it at
        lidar activation does not affect running cameras.
        """
        if os.environ.get("ARENA_MID360_HOLD_RTX_SENSORS", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        try:
            import carb

            settings = carb.settings.get_settings()
            settings.set_bool("/isaaclab/render/rtx_sensors", False)
            print(
                "ARENA_MID360_RTX_SENSORS_HELD flag_false "
                f"render_product={(resource or {}).get('render_product_path', '')}",
                flush=True,
            )
        except BaseException as error:
            print(
                f"ARENA_MID360_RTX_SENSORS_HELD failed: {type(error).__name__}: {error}",
                flush=True,
            )

    def _resubscribe_gmo(self, resource: dict[str, Any]) -> None:
        """Keep the GMO annotator subscribed while the camera filter is open.

        Tiled-camera ``Camera.__init__`` flips ``/isaaclab/render/rtx_sensors``
        true before its renderer exists, so every env.step runs ShadowRtRender-
        Product and drops non-tiled render products (our lidar's hydra texture)
        from the RTX pipeline — the GMO buffer freezes (verified A/B: a single
        ``/app/player/playSimulations`` false-update-true toggle in a bare
        process kills the stream identically). While the buffer is stale we
        re-attach the annotator so the sensor regains a subscription and keeps
        producing frames.
        """
        if os.environ.get("ARENA_MID360_GMO_RESUBSCRIBE", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return
        signature = self._gmo_signature(resource)
        if signature is None:
            return
        if signature != self._last_gmo_signature:
            self._last_gmo_signature = signature
            self._gmo_stale_polls = 0
            return
        self._gmo_stale_polls += 1
        if self._gmo_stale_polls < 3:
            return
        self._gmo_stale_polls = 0
        self._gmo_resubscribe_count += 1
        self._gmo_attach(resource)

    def _gmo_attach(self, resource: dict[str, Any]) -> None:
        """(Re)attach the generic-model-output annotator to the hydra texture."""
        render_product_path = str(resource.get("render_product_path", ""))
        try:
            import omni.replicator.core as rep

            reusable = rep.AnnotatorRegistry.get_annotator(
                "GenericModelOutput", device="cpu", do_array_copy=False
            )
            reusable.attach([render_product_path])
            if self._gmo_resubscribe_count % 10 == 0:
                print(
                    f"ARENA_MID360_GMO_RESUB#{self._gmo_resubscribe_count} attached "
                    f"{render_product_path}",
                    flush=True,
                )
        except BaseException as error:
            if self._gmo_resubscribe_count % 10 == 0:
                print(
                    f"ARENA_MID360_GMO_RESUB#{self._gmo_resubscribe_count} failed: "
                    f"{type(error).__name__}: {error}",
                    flush=True,
                )

    def _gmo_signature(self, resource: dict[str, Any]) -> tuple[int, int] | None:
        """Return (timestampNs, numElements) of the latest GMO frame, or None."""
        lidar_sensor = resource.get("lidar_sensor")
        get_data = getattr(lidar_sensor, "get_data", None)
        if not callable(get_data):
            return None
        try:
            data, _info = get_data("generic-model-output")
            numpy_buffer = getattr(data, "numpy", None)
            numpy_buffer = numpy_buffer() if callable(numpy_buffer) else None
            if numpy_buffer is None:
                return None
            from isaacsim.sensors.experimental.rtx.generic_model_output import (
                getModelOutputFromBuffer,
            )

            gmo = getModelOutputFromBuffer(numpy_buffer)
            return int(gmo.timestampNs), int(gmo.numElements)
        except BaseException:
            return None

    def shutdown(self) -> None:
        """Stop relay and remove all startup-owned graphs and render products."""
        self._shutdown_direct_publisher()
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
            else:
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
        if errors:
            raise RuntimeError("sensor shutdown cleanup failed: " + "; ".join(map(str, errors)))
        self._prepared_lidar = None
        self._prepared_imu = None
        self._prepared_stage = None

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
