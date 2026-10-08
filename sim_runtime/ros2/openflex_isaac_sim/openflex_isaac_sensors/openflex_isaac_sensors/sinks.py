"""Output adapters that consume FramePacket without owning rendering."""

from __future__ import annotations

from pathlib import Path
import json
import re
import threading
import time
from typing import Any

from .frame_packet import FramePacket
from .transport import BoundedFrameQueue


def _enable_isaac_extension(extension_name: str) -> bool:
    """Enable a Kit extension through the stable Isaac Core utilities API."""
    from isaacsim.core.utils.extensions import enable_extension

    return bool(enable_extension(extension_name))


class JsonlSink:
    """Small, dependency-free metadata sink for smoke tests and audit trails."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")

    def write(self, packet: FramePacket) -> None:
        packet.validate()
        payload: dict[str, Any] = {
            "episode_id": packet.episode_id,
            "snapshot_id": packet.snapshot_id,
            "camera_name": packet.camera_name,
            "frame_id": packet.frame_id,
            "sample_sim_time_ns": packet.sample_sim_time_ns,
            "capture_wall_time_ns": packet.capture_wall_time_ns,
            "calibration_id": packet.calibration_id,
            "depth_semantics": packet.depth_semantics,
            "source_state_seq": packet.source_state_seq,
            "rgb_encoding": packet.rgb_encoding,
            "depth_encoding": packet.depth_encoding,
            "rgb_shape": list(getattr(packet.rgb, "shape", ())) if packet.rgb is not None else None,
            "depth_shape": list(getattr(packet.depth_m, "shape", ())) if packet.depth_m is not None else None,
            "has_imu": packet.imu is not None,
        }
        self._stream.write(json.dumps(payload, sort_keys=True) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


class AsyncJsonlSink:
    """Write packet metadata from a worker without touching Kit APIs."""

    def __init__(self, path: str | Path, maxsize: int = 8) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")
        self._queue: BoundedFrameQueue[FramePacket] = BoundedFrameQueue(maxsize)
        self._stop = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name=f"jsonl-{self.path.stem}",
            daemon=True,
        )
        self._worker.start()

    def write(self, packet: FramePacket) -> None:
        self._queue.put(packet)

    def _run(self) -> None:
        while not self._stop.is_set() or len(self._queue):
            packet = self._queue.get()
            if packet is None:
                time.sleep(0.001)
                continue
            JsonlSink.write(self, packet)

    def stats(self) -> dict[str, int]:
        return self._queue.stats()

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            raise RuntimeError(f"async sink did not stop: {self.path}")
        self._stream.close()


class IsaacSimRos2Bridge:
    """Attach ROS2 helpers to an existing RenderProduct.

    This adapter deliberately does not create cameras or RenderProducts. The
    caller owns the USD camera and passes its single Replicator RenderProduct.
    """

    def __init__(self, graph_path: str = "/World/StandaloneSensors/ROS2") -> None:
        self.graph_path = graph_path
        self._graph = None
        self._tick_created = False
        self._graph_render_nodes: dict[str, str] = {}
        self._graph_camera_specs: dict[str, tuple[str, tuple[int, int]]] = {}
        self._gate_config: dict[str, dict[str, Any]] = {}
        self._qos_config: dict[str, Any] = {}
        self._runtime_camera_gates: dict[str, tuple[str, ...]] = {}

    @staticmethod
    def _node_prefix(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or "camera"

    def attach_camera(
        self,
        render_product_path: str,
        *,
        frame_id: str,
        node_namespace: str = "",
        rgb_topic: str = "color/image_raw",
        depth_topic: str = "depth/image_rect_raw",
        camera_info_topic: str = "color/camera_info",
        frame_skip_count: int = 0,
    ) -> object:
        return self.attach_cameras([{
            "render_product_path": render_product_path,
            "frame_id": frame_id,
            "node_namespace": node_namespace,
            "rgb_topic": rgb_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "frame_skip_count": frame_skip_count,
        }])

    def attach_cameras(self, cameras: list[dict[str, Any]]) -> object:
        """Attach all camera helpers in one official multi-camera graph edit."""
        if not cameras:
            raise ValueError("at least one camera is required")
        import omni.graph.core as og
        import usdrt.Sdf

        _enable_isaac_extension("isaacsim.ros2.bridge")
        # Isaac Sim 6.0's camera helper can use the native SRTX callback path
        # for image transport. Enable it before the helper nodes initialize so
        # they do not fall back to the Replicator writer path.
        import carb

        if carb.settings.get_settings().get_as_bool("/exts/omni.replicator.srtx/enabled"):
            _enable_isaac_extension("omni.replicator.srtx")
        keys = og.Controller.Keys
        tick_node = "OnPlaybackTick"
        qos_node = "SensorDataQoS"
        context_node = "ROS2Context"
        create_nodes = [
            (tick_node, "omni.graph.action.OnPlaybackTick"),
            (context_node, "isaacsim.ros2.bridge.ROS2Context"),
            (qos_node, "isaacsim.ros2.bridge.ROS2QoSProfile"),
        ]
        set_values = []
        connections = []
        qos_profile = str(cameras[0].get("qos_profile", "Sensor Data"))
        if qos_profile not in {"Sensor Data", "Custom"}:
            raise ValueError(f"unsupported ROS2 QoS profile: {qos_profile}")
        set_values.append((f"{qos_node}.inputs:createProfile", qos_profile))
        if qos_profile == "Custom":
            set_values.extend([
                (f"{qos_node}.inputs:reliability", "reliable"),
                (f"{qos_node}.inputs:history", "keepLast"),
                (f"{qos_node}.inputs:durability", "volatile"),
            ])
        queue_sizes: dict[str, int] = {}
        for index, camera in enumerate(cameras):
            render_product_path = str(camera.get("render_product_path", ""))
            camera_prim_path = str(camera.get("camera_prim_path", ""))
            graph_owned = bool(camera.get("graph_owned_render_product", False))
            if graph_owned and not camera_prim_path:
                raise ValueError("camera_prim_path is required for graph-owned RenderProducts")
            if not graph_owned and not render_product_path:
                raise ValueError("render_product_path is required")
            frame_id = str(camera.get("frame_id", "camera_optical_frame"))
            rgb_frame_id = str(camera.get("rgb_frame_id", frame_id))
            depth_frame_id = str(camera.get("depth_frame_id", frame_id))
            node_namespace = str(camera.get("node_namespace", ""))
            queue_size = int(camera.get("queue_size", 5))
            if queue_size < 1:
                raise ValueError(f"{node_namespace or frame_id}: queue_size must be positive")
            queue_sizes[str(camera.get("camera_key", index))] = queue_size
            prefix = self._node_prefix(node_namespace or frame_id) + f"_{index}"
            render_node = f"{prefix}_CreateRenderProduct"
            rgb_node = f"{prefix}_RGB"
            depth_node = f"{prefix}_Depth"
            info_node = f"{prefix}_CameraInfo"
            depth_info_node = f"{prefix}_DepthCameraInfo"
            exec_gate_step = int(camera.get("exec_gate_step", 0))
            exec_gate_nodes: dict[str, str] = {}
            publish_rgb = bool(camera.get("publish_rgb", True))
            publish_depth = bool(camera.get("publish_depth", True))
            if not publish_rgb and not publish_depth:
                raise ValueError("at least one of publish_rgb or publish_depth must be enabled")
            sensor_nodes = []
            if graph_owned:
                create_nodes.append((render_node, "isaacsim.core.nodes.IsaacCreateRenderProduct"))
            if publish_rgb:
                create_nodes.append((rgb_node, "isaacsim.ros2.bridge.ROS2CameraHelper"))
                sensor_nodes.append(rgb_node)
            if publish_depth:
                create_nodes.append((depth_node, "isaacsim.ros2.bridge.ROS2CameraHelper"))
                sensor_nodes.append(depth_node)
            publish_camera_info = bool(camera.get("publish_camera_info", True))
            if publish_camera_info:
                create_nodes.append((info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"))
            depth_camera_info_topic = str(camera.get("depth_camera_info_topic", ""))
            publish_depth_camera_info = bool(depth_camera_info_topic) and publish_depth
            if publish_depth_camera_info:
                create_nodes.append((depth_info_node, "isaacsim.ros2.bridge.ROS2CameraInfoHelper"))
            info_nodes = tuple(
                node
                for node, enabled in (
                    (info_node, publish_camera_info),
                    (depth_info_node, publish_depth_camera_info),
                )
                if enabled
            )
            runtime_gate_nodes: dict[str, str] = {}
            for node in tuple(sensor_nodes) + info_nodes:
                gate_node = f"{node}_RuntimeEnableGate"
                runtime_gate_nodes[node] = gate_node
                create_nodes.append((gate_node, "isaacsim.core.nodes.IsaacSimulationGate"))
                set_values.append((f"{gate_node}.inputs:step", 0))
            if exec_gate_step > 0 and not graph_owned:
                for sensor_node in tuple(sensor_nodes) + info_nodes:
                    gate_node = f"{sensor_node}_ExecGate"
                    exec_gate_nodes[sensor_node] = gate_node
                    create_nodes.append((gate_node, "isaacsim.core.nodes.IsaacSimulationGate"))
                    set_values.append((f"{gate_node}.inputs:step", exec_gate_step))
                    connections.append((f"{tick_node}.outputs:tick", f"{gate_node}.inputs:execIn"))
            if graph_owned:
                set_values.extend([
                    (f"{render_node}.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
                    (f"{render_node}.inputs:width", int(camera.get("width", 640))),
                    (f"{render_node}.inputs:height", int(camera.get("height", 480))),
                ])
                render_nodes = tuple(sensor_nodes) + info_nodes
                for node in render_nodes:
                    gate_node = runtime_gate_nodes[node]
                    connections.extend([
                        (f"{render_node}.outputs:execOut", f"{gate_node}.inputs:execIn"),
                        (f"{gate_node}.outputs:execOut", f"{node}.inputs:execIn"),
                        (f"{render_node}.outputs:renderProductPath", f"{node}.inputs:renderProductPath"),
                    ])
            else:
                render_nodes = tuple(sensor_nodes) + info_nodes
                for node in render_nodes:
                    set_values.append((f"{node}.inputs:renderProductPath", render_product_path))
            sensor_specs = []
            if publish_rgb:
                sensor_specs.append((
                    rgb_node,
                    camera.get("rgb_topic", "color/image_raw"),
                    camera.get("rgb_type", "rgb"),
                    rgb_frame_id,
                ))
            if publish_depth:
                sensor_specs.append((
                    depth_node,
                    camera.get("depth_topic", "depth/image_rect_raw"),
                    "depth",
                    depth_frame_id,
                ))
            for node, topic, sensor_type, sensor_frame_id in sensor_specs:
                set_values.extend([
                    (f"{node}.inputs:frameId", sensor_frame_id),
                    (f"{node}.inputs:nodeNamespace", node_namespace),
                    (f"{node}.inputs:topicName", str(topic)),
                    (f"{node}.inputs:type", sensor_type),
                    (f"{node}.inputs:frameSkipCount", int(camera.get("frame_skip_count", 0))),
                    (f"{node}.inputs:queueSize", queue_size),
                ])
                connections.append((f"{qos_node}.outputs:qosProfile", f"{node}.inputs:qosProfile"))
                connections.append((f"{context_node}.outputs:context", f"{node}.inputs:context"))
                if not graph_owned:
                    source = exec_gate_nodes.get(node, tick_node)
                    output = f"{source}.outputs:execOut" if source != tick_node else f"{source}.outputs:tick"
                    gate_node = runtime_gate_nodes[node]
                    connections.extend([
                        (output, f"{gate_node}.inputs:execIn"),
                        (f"{gate_node}.outputs:execOut", f"{node}.inputs:execIn"),
                    ])
            if publish_camera_info:
                set_values.extend([
                    (f"{info_node}.inputs:frameId", rgb_frame_id),
                    (f"{info_node}.inputs:nodeNamespace", node_namespace),
                    (f"{info_node}.inputs:topicName", str(camera.get("camera_info_topic", "color/camera_info"))),
                    (f"{info_node}.inputs:frameSkipCount", int(camera.get("frame_skip_count", 0))),
                    (f"{info_node}.inputs:queueSize", queue_size),
                ])
                connections.append((f"{qos_node}.outputs:qosProfile", f"{info_node}.inputs:qosProfile"))
                connections.append((f"{context_node}.outputs:context", f"{info_node}.inputs:context"))
                if not graph_owned:
                    source = exec_gate_nodes.get(info_node, tick_node)
                    output = f"{source}.outputs:execOut" if source != tick_node else f"{source}.outputs:tick"
                    gate_node = runtime_gate_nodes[info_node]
                    connections.extend([
                        (output, f"{gate_node}.inputs:execIn"),
                        (f"{gate_node}.outputs:execOut", f"{info_node}.inputs:execIn"),
                    ])
            if publish_depth_camera_info:
                set_values.extend([
                    (f"{depth_info_node}.inputs:frameId", depth_frame_id),
                    (f"{depth_info_node}.inputs:nodeNamespace", node_namespace),
                    (f"{depth_info_node}.inputs:topicName", depth_camera_info_topic),
                    (f"{depth_info_node}.inputs:frameSkipCount", int(camera.get("frame_skip_count", 0))),
                    (f"{depth_info_node}.inputs:queueSize", queue_size),
                ])
                connections.append((f"{qos_node}.outputs:qosProfile", f"{depth_info_node}.inputs:qosProfile"))
                connections.append((f"{context_node}.outputs:context", f"{depth_info_node}.inputs:context"))
                if not graph_owned:
                    source = exec_gate_nodes.get(depth_info_node, tick_node)
                    output = f"{source}.outputs:execOut" if source != tick_node else f"{source}.outputs:tick"
                    gate_node = runtime_gate_nodes[depth_info_node]
                    connections.extend([
                        (output, f"{gate_node}.inputs:execIn"),
                        (f"{gate_node}.outputs:execOut", f"{depth_info_node}.inputs:execIn"),
                    ])
            camera_key = str(camera.get("camera_key", index))
            self._runtime_camera_gates[camera_key] = tuple(
                f"{self.graph_path}/{gate_node}.inputs:step"
                for gate_node in runtime_gate_nodes.values()
            )
            if graph_owned:
                self._graph_render_nodes[camera_key] = render_node
                self._graph_camera_specs[str(camera.get("camera_key", index))] = (
                    camera_prim_path,
                    (int(camera.get("width", 640)), int(camera.get("height", 480))),
                )
        if qos_profile == "Custom":
            set_values.append((f"{qos_node}.inputs:depth", max(queue_sizes.values())))
        graph, _, _, _ = og.Controller.edit(
            {"graph_path": self.graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: create_nodes,
                keys.SET_VALUES: set_values,
                keys.CONNECT: connections,
            },
        )
        self._graph = graph
        self._tick_created = True
        self._qos_config = {
            "profile": qos_profile,
            "reliability": "reliable" if qos_profile == "Custom" else "bestEffort",
            "history": "keepLast",
            "durability": "volatile",
            "queue_sizes": queue_sizes,
        }
        return graph

    def set_camera_enabled(self, camera_key: str, enabled: bool) -> None:
        """Gate every ROS 2 stream belonging to a camera on/off."""
        gate_paths = self._runtime_camera_gates.get(str(camera_key))
        if not gate_paths:
            raise KeyError(f"no runtime camera gates for {camera_key!r}")
        import omni.graph.core as og

        step = 1 if enabled else 0
        for path in gate_paths:
            og.Controller.attribute(path).set(step)

    def destroy(self, stage: object | None = None) -> None:
        """Remove this camera's isolated publisher graph from the active stage."""
        if stage is None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        import pxr.Sdf

        prim = stage.GetPrimAtPath(self.graph_path)
        if prim and prim.IsValid():
            stage.RemovePrim(pxr.Sdf.Path(self.graph_path))
        self._graph = None
        self._tick_created = False
        self._graph_render_nodes.clear()
        self._graph_camera_specs.clear()
        self._gate_config.clear()
        self._qos_config.clear()
        self._runtime_camera_gates.clear()

    def configure_camera_gates(self, cameras: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Configure Isaac Sim 6.0 SyntheticData gates after helper startup.

        ``frameSkipCount`` is deprecated in Isaac Sim 6.0. The camera prim's
        ``omni:sensor:tickRate`` controls when a new sample is rendered; these
        gates publish once for each available RGB, depth, and CameraInfo sample.
        """
        if self._graph is None:
            raise RuntimeError("attach_cameras() must be called before configure_camera_gates()")

        import omni.graph.core as og
        import omni.syntheticdata
        import carb

        use_srtx = carb.settings.get_settings().get_as_bool("/exts/omni.replicator.srtx/enabled")

        records: dict[str, dict[str, Any]] = {}
        for index, camera in enumerate(cameras):
            camera_key = str(camera.get("camera_key", index))
            render_product_path = str(camera.get("render_product_path", ""))
            if not render_product_path:
                raise ValueError(f"{camera_key}: render_product_path is required")

            camera_prim_path = str(camera.get("camera_prim_path", ""))
            tick_rate_hz = float(camera.get("tick_rate_hz", 0.0))
            if camera_prim_path and tick_rate_hz > 0.0:
                import omni.usd

                prim = omni.usd.get_context().get_stage().GetPrimAtPath(camera_prim_path)
                tick_attr = prim.GetAttribute("omni:sensor:tickRate") if prim else None
                if tick_attr and tick_attr.IsValid():
                    tick_attr.Set(tick_rate_hz)

            gate_paths: dict[str, str] = {}
            if not use_srtx:
                rv_rgb = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar("Rgb")
                rv_depth = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                    "DistanceToImagePlane"
                )
                if bool(camera.get("publish_rgb", True)) and camera.get("rgb_type", "rgb") == "rgb":
                    gate_paths["rgb"] = omni.syntheticdata.SyntheticData._get_node_path(
                        rv_rgb + "IsaacSimulationGate", render_product_path
                    )
                if bool(camera.get("publish_depth", True)):
                    gate_paths["depth"] = omni.syntheticdata.SyntheticData._get_node_path(
                        rv_depth + "IsaacSimulationGate", render_product_path
                    )
                if bool(camera.get("publish_camera_info", True)) or bool(
                    camera.get("depth_camera_info_topic", "")
                ):
                    gate_paths["camera_info"] = omni.syntheticdata.SyntheticData._get_node_path(
                        "PostProcessDispatchIsaacSimulationGate", render_product_path
                    )
            gate_steps: dict[str, int] = {}
            for name, gate_path in gate_paths.items():
                if not gate_path:
                    raise RuntimeError(
                        f"{camera_key}: SyntheticData gate was not found for {name} "
                        f"on {render_product_path}"
                    )
                attribute = og.Controller.attribute(gate_path + ".inputs:step")
                attribute.set(1)
                gate_steps[name] = int(attribute.get())

            records[camera_key] = {
                "render_product": render_product_path,
                "camera_prim": camera_prim_path or None,
                "tick_rate_hz": tick_rate_hz or None,
                "qos": dict(self._qos_config),
                "gate_paths": gate_paths,
                "gate_steps": gate_steps,
            }

        self._gate_config = records
        return {key: dict(value) for key, value in records.items()}

    @property
    def gate_config(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._gate_config.items()}

    @property
    def qos_config(self) -> dict[str, Any]:
        return dict(self._qos_config)

    def get_graph_render_product_path(self, camera_key: str) -> str | None:
        """Read a graph-owned RenderProduct output after the graph has ticked."""
        if self._graph is None:
            raise RuntimeError("attach_cameras() has not been called")
        import omni.graph.core as og

        node_name = self._graph_render_nodes.get(str(camera_key))
        if not node_name:
            raise KeyError(f"no graph-owned RenderProduct node for camera {camera_key!r}")
        value = og.Controller.attribute(
            f"{self.graph_path}/{node_name}.outputs:renderProductPath"
        ).get()
        if value:
            return str(value)

        # The graph node may have created the product while its output token
        # is still empty on the Python side during the first update. Resolve
        # the same stable product from Replicator's registry instead.
        import omni.replicator.core as rep
        camera_spec = self._graph_camera_specs.get(str(camera_key))
        if camera_spec is None:
            raise KeyError(f"no graph-owned camera spec for {camera_key!r}")
        camera_path, resolution = camera_spec
        matches = []
        for product in rep.functional.get.renderproduct():
            targets = product.GetRelationship("camera").GetTargets()
            product_resolution = tuple(product.GetAttribute("resolution").Get() or ())
            if targets and str(targets[0]) == camera_path and product_resolution == resolution:
                matches.append(str(product.GetPath()))
        if len(matches) == 1:
            return matches[0]
        return None

    def attach_tf_tree(
        self,
        target_prim_path: str,
        *,
        parent_prim_path: str | None = None,
        topic_name: str = "tf_static",
        node_namespace: str = "",
        static_publisher: bool = True,
        mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        mount_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        mount_frame: str = "camera_mount",
        camera_frame: str = "camera",
        optical_frame: str = "camera_optical_frame",
        imu_frame: str | None = None,
    ) -> object:
        """Publish a static standalone TF chain without creating another sensor.

        The deprecated target-prim pose-tree path requires physics object
        handles and is unsuitable for a camera-only Xform. Raw TF publishers
        keep this independent stage valid; robot integration can replace this
        thin adapter with a dynamic robot TF source.
        """
        if not target_prim_path.startswith("/"):
            raise ValueError("target_prim_path must be an absolute USD path")
        import omni.graph.core as og
        import usdrt.Sdf

        keys = og.Controller.Keys
        tf_graph_path = self.graph_path + "_TF_" + self._node_prefix(mount_frame)
        links = [
            ("standalone_sensors", mount_frame, list(mount_translation), list(mount_quaternion_wxyz)),
            (mount_frame, camera_frame, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            (camera_frame, optical_frame, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
        ]
        if imu_frame:
            links.append((camera_frame, imu_frame, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]))
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ]
        set_values = []
        connections = []
        for index, (parent_frame, child_frame, translation, quaternion_wxyz) in enumerate(links):
            node = f"TF{index}"
            create_nodes.append((node, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
            set_values.extend([
                (f"{node}.inputs:parentFrameId", parent_frame),
                (f"{node}.inputs:childFrameId", child_frame),
                (f"{node}.inputs:topicName", topic_name),
                (f"{node}.inputs:nodeNamespace", node_namespace),
                (f"{node}.inputs:staticPublisher", static_publisher),
                (f"{node}.inputs:translation", translation),
                (f"{node}.inputs:rotation", [quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]]),
            ])
            connections.extend([
                # Keep the execution source alive for the whole startup
                # window. StaticPublisher uses static QoS, while repeated
                # execution makes DDS discovery robust when the subscriber
                # starts around bridge initialization.
                ("OnPlaybackTick.outputs:tick", f"{node}.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", f"{node}.inputs:timeStamp"),
            ])
        graph, _, _, _ = og.Controller.edit(
            {"graph_path": tf_graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: create_nodes,
                keys.SET_VALUES: set_values,
                keys.CONNECT: connections,
            },
        )
        return graph

    @property
    def graph(self) -> object:
        if self._graph is None:
            raise RuntimeError("attach_camera() has not been called")
        return self._graph
