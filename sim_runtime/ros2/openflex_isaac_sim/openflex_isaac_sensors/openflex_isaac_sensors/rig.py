"""Isaac Sim 6.0 RealSense rig with a robot-independent mounting contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping
from uuid import uuid4

from .diagnostics import SensorDiagnostics
from .frame_packet import FramePacket
from .mount import LocalPose, apply_local_pose, resolve_mount_prim_path


class _DirectCameraProduct:
    """Read standard camera annotators from a graph-owned RenderProduct."""

    def __init__(self, resolution: tuple[int, int]) -> None:
        self._resolution = resolution
        self._annotators: dict[str, object] = {}

    def attach(self, render_product_path: str, quality_level: str) -> None:
        import omni.replicator.core as rep

        if quality_level != "L0_aligned_fast":
            raise ValueError("graph-owned RenderProducts currently support L0_aligned_fast only")
        self._annotators = {
            "rgb": rep.AnnotatorRegistry.get_annotator("rgb", device="cuda", do_array_copy=False),
            "distance_to_image_plane": rep.AnnotatorRegistry.get_annotator(
                "distance_to_image_plane", device="cuda", do_array_copy=False
            ),
        }
        for annotator in self._annotators.values():
            annotator.attach([render_product_path])

    def get_data(self, annotator_name: str) -> object | None:
        annotator = self._annotators[annotator_name]
        data = annotator.get_data(device="cuda")
        if isinstance(data, dict):
            data = data.get("data")
        if data is None or not getattr(data, "shape", None) or data.shape[0] == 0:
            return None
        import warp as wp

        if not isinstance(data, wp.array):
            data = wp.array(data, device="cuda")
        channels = 4 if annotator_name == "rgb" else 1
        data = data.reshape((*self._resolution, channels))
        return data[:, :, :3] if annotator_name == "rgb" else data


class _IsaacPhysicsImuAdapter:
    """Expose Isaac Sim 5.1 IMUSensor samples through the rig packet contract."""

    def __init__(self, sensor: object) -> None:
        self._sensor = sensor

    def initialize(self) -> None:
        initialize = getattr(self._sensor, "initialize", None)
        if callable(initialize):
            initialize()

    def get_data(self) -> dict[str, object]:
        frame = self._sensor.get_current_frame()
        return {
            "orientation": frame.get("orientation"),
            "linear_acceleration": frame.get("lin_acc"),
            "angular_velocity": frame.get("ang_vel"),
            "time": frame.get("time"),
            "physics_step": frame.get("physics_step"),
        }

    def destroy(self) -> None:
        destroy = getattr(self._sensor, "destroy", None)
        if callable(destroy):
            destroy()
        self._sensor = None


def _author_camera_prim(stage: object, camera_path: str, config: "RigConfig") -> object:
    """Author a standard USD camera and the rate attribute consumed by ROS helpers.

    Isaac Sim 5.1 ships the stable ``isaacsim.sensors.camera`` extension, but
    the runtime ROS path only needs a USD Camera prim plus a Replicator render
    product.  Keeping the prim authoring independent of a sensor wrapper also
    avoids the Isaac Sim 6-only ``sensors.experimental`` package.
    """
    from pxr import Gf, Sdf, UsdGeom

    camera = UsdGeom.Camera.Define(stage, Sdf.Path(camera_path))
    camera.GetFocalLengthAttr().Set(float(config.focal_length_mm))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(config.near_m), float(config.far_m)))
    if config.calibration is not None:
        matrix = config.calibration.get("K")
        if not isinstance(matrix, (list, tuple)) or len(matrix) != 9:
            raise ValueError(f"camera {config.name} calibration K must contain 9 values")
        calibration_width = int(config.calibration.get("width", 0))
        calibration_height = int(config.calibration.get("height", 0))
        fx, fy = float(matrix[0]), float(matrix[4])
        if calibration_width <= 0 or calibration_height <= 0 or fx <= 0.0 or fy <= 0.0:
            raise ValueError(f"camera {config.name} calibration dimensions/focal lengths are invalid")
        camera.GetHorizontalApertureAttr().Set(
            float(config.focal_length_mm) * calibration_width / fx
        )
        camera.GetVerticalApertureAttr().Set(
            float(config.focal_length_mm) * calibration_height / fy
        )
    prim = camera.GetPrim()
    tick_rate = prim.GetAttribute("omni:sensor:tickRate")
    if not tick_rate or not tick_rate.IsValid():
        tick_rate = prim.CreateAttribute(
            "omni:sensor:tickRate", Sdf.ValueTypeNames.Float, custom=True
        )
    tick_rate.Set(float(config.tick_rate_hz))
    return camera


@dataclass(slots=True)
class RigConfig:
    name: str
    model: str
    tick_rate_hz: float = 30.0
    width: int = 640
    height: int = 480
    quality_level: str = "L0_aligned_fast"
    focal_length_mm: float = 3.2
    near_m: float = 0.05
    far_m: float = 10.0
    calibration_id: str = "nominal_unvalidated"
    calibration: Mapping[str, Any] | None = None
    baseline_mm: float = 50.0
    depth_focal_length_px: float = 615.0
    depth_min_m: float = 0.10
    depth_max_m: float = 10.0
    imu_enabled: bool = False
    imu_rate_hz: float = 200.0
    optical_frame: str = "camera_color_optical_frame"
    imu_frame: str = "camera_imu_frame"

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "RigConfig":
        model = str(value.get("model", "D435"))
        imu = value.get("imu", {}) or {}
        return cls(
            name=name,
            model=model,
            tick_rate_hz=float(value.get("tick_rate_hz", 30.0)),
            width=int(value.get("width", 640)),
            height=int(value.get("height", 480)),
            quality_level=str(value.get("quality_level", "L0_aligned_fast")),
            focal_length_mm=float(value.get("focal_length_mm", 3.2)),
            near_m=float(value.get("near_m", 0.05)),
            far_m=float(value.get("far_m", 10.0)),
            calibration_id=str(value.get("calibration_id", "nominal_unvalidated")),
            calibration=(
                value.get("calibration")
                if isinstance(value.get("calibration"), Mapping)
                else None
            ),
            baseline_mm=float(value.get("baseline_mm", 50.0)),
            depth_focal_length_px=float(value.get("depth_focal_length_px", 615.0)),
            depth_min_m=float(value.get("depth_min_m", 0.10)),
            depth_max_m=float(value.get("depth_max_m", 10.0)),
            imu_enabled=bool(imu.get("enabled", False)),
            imu_rate_hz=float(imu.get("publish_rate_hz", 200.0)),
            optical_frame=str(value.get("optical_frame", f"{name}_color_optical_frame")),
            imu_frame=str(imu.get("frame", f"{name}_imu_frame")),
        )


def _make_prim_editable(stage: object, prim_path: str) -> object:
    """Make a referenced robot link editable before adding runtime sensors."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return prim
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
        prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsInstanceProxy():
        return prim

    ancestors: list[object] = []
    current = prim
    while current and current.IsValid():
        if current.IsInstanceable():
            ancestors.append(current)
        current = current.GetParent()

    for ancestor in ancestors:
        ancestor.SetInstanceable(False)

    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid() and not prim.IsInstanceProxy():
        return prim
    instance_root = str(ancestors[0].GetPath()) if ancestors else prim_path
    raise RuntimeError(
        "robot link remains an instance proxy after disabling instanceability: "
        f"{prim_path} (instance root: {instance_root})"
    )


class RealSenseRig:
    """Create one physical RGB-D view and optionally one D435i IMU."""

    def __init__(self, name: str = "realsense", config: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self.config = RigConfig.from_mapping(name, config or {})
        self._stage: object | None = None
        self._sensor: object | None = None
        self._imu_sensor: object | None = None
        self._camera_object: object | None = None
        self._camera_path: str | None = None
        self._camera_prim_path: str | None = None
        self._mount_path: str | None = None
        self._mount_created = False
        self._render_product_handle: object | None = None
        self._local_pose = LocalPose()
        self._render_product_path: str | None = None
        self._period_s = 1.0 / self.config.tick_rate_hz
        self._next_due_s = 0.0
        self._frame_id = 0
        self._episode_id = 0
        self._active = False
        self._diagnostics = SensorDiagnostics(name)

    def create(
        self,
        stage: object,
        parent_prim_path: str,
        local_pose: LocalPose | Mapping[str, Any] | None,
        config: Mapping[str, Any] | None = None,
        create_render_product: bool = True,
    ) -> "RealSenseRig":
        """Create the rig relative to a generic parent, with no robot imports."""
        self._stage = stage
        if config is not None:
            self.config = RigConfig.from_mapping(self.name, config)
            self._period_s = 1.0 / self.config.tick_rate_hz
        pose = local_pose if isinstance(local_pose, LocalPose) else LocalPose.from_mapping(local_pose)
        self._local_pose = pose
        mount_path = resolve_mount_prim_path(parent_prim_path, self.name, (config or {}).get("mount_prim_path"))
        parent = _make_prim_editable(stage, parent_prim_path)
        if not parent or not parent.IsValid():
            raise RuntimeError(
                f"RealSense parent prim does not exist: {parent_prim_path}; "
                f"cannot create mount {mount_path}"
            )
        mount_parent_path = mount_path.rsplit("/", 1)[0]
        if mount_parent_path != parent_prim_path.rstrip("/"):
            raise RuntimeError(
                f"RealSense mount {mount_path} is not below its configured parent "
                f"{parent_prim_path}"
            )
        existing_mount = stage.GetPrimAtPath(mount_path)
        self._mount_created = not bool(existing_mount and existing_mount.IsValid())
        if existing_mount and existing_mount.IsValid():
            actual_parent = str(existing_mount.GetParent().GetPath())
            if actual_parent != parent_prim_path.rstrip("/"):
                raise RuntimeError(
                    f"RealSense mount {mount_path} has parent {actual_parent}, "
                    f"expected {parent_prim_path}"
                )
        self._mount_path = mount_path
        apply_local_pose(stage, mount_path, pose)

        self._camera_path = f"{mount_path}/{self.name}_Camera"
        self._camera_prim_path = self._camera_path
        camera = _author_camera_prim(stage, self._camera_path, self.config)
        self._camera_object = camera

        # Keep the ROS optical frame explicit in the USD tree. The camera
        # itself remains the only render source; this is a zero-cost TF frame.
        stage.DefinePrim(f"{self._camera_path}/{self.config.optical_frame}", "Xform")

        resolution = (self.config.height, self.config.width)
        if self.config.quality_level not in {"L0_aligned_fast", "L1_stereo_noise"}:
            raise ValueError(f"unsupported quality level: {self.config.quality_level}")
        self._sensor = _DirectCameraProduct(resolution)
        if create_render_product:
            self.create_python_render_product()

        if self.config.imu_enabled:
            if self.config.model != "D435i":
                raise ValueError("IMU is only valid when model is D435i")
            import numpy as np
            from isaacsim.sensors.physics import IMUSensor

            imu_path = f"{mount_path}/{self.config.imu_frame}"
            self._imu_sensor = _IsaacPhysicsImuAdapter(
                IMUSensor(
                    prim_path=imu_path,
                    name=f"{self.name}_imu",
                    frequency=max(1, int(round(self.config.imu_rate_hz))),
                    translation=np.zeros(3, dtype=np.float64),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                )
            )

        if self._render_product_path is None:
            render_product = getattr(self._sensor, "render_product", None)
            get_path = getattr(render_product, "GetPath", None)
            if callable(get_path):
                render_product_path = get_path()
                if render_product_path:
                    self._render_product_path = str(render_product_path)
        return self

    def start(self) -> None:
        if self._sensor is None:
            raise RuntimeError("create() must be called before start()")
        if self._imu_sensor is not None:
            self._imu_sensor.initialize()
        if isinstance(self._sensor, _DirectCameraProduct):
            if not self._render_product_path:
                raise RuntimeError("configure_render_product() must be called before start()")
            self._sensor.attach(self._render_product_path, self.config.quality_level)
        elif self.config.quality_level == "L1_stereo_noise":
            self._sensor.attach_annotators(["rgb", "depth_sensor_distance"])
        else:
            self._sensor.attach_annotators(["rgb", "distance_to_image_plane"])
        self._active = True

    def configure_render_product(self, render_product_path: str) -> None:
        if not render_product_path:
            raise ValueError("render_product_path is required")
        self._render_product_path = str(render_product_path)

    def create_python_render_product(self, *, publish_rgb: bool = True, publish_depth: bool = True) -> str:
        """Create the single product used by the output path."""
        if self._camera_path is None:
            raise RuntimeError("create() must be called before create_python_render_product()")
        if not publish_rgb and not publish_depth:
            raise ValueError("at least one of publish_rgb or publish_depth must be enabled")
        import omni.replicator.core as rep

        render_product_name = f"{self.name}_{uuid4().hex}"
        product = rep.create.render_product(
            self._camera_path,
            (self.config.width, self.config.height),
            name=render_product_name,
        )
        self._render_product_handle = product
        self._render_product_path = str(product.path)
        return self._render_product_path

    def record_physics(self, sample_sim_time: float) -> None:
        self._diagnostics.record_physics(sample_sim_time)

    def reset_measurement(self) -> None:
        """Reset timing diagnostics after RTX warm-up, without resetting the rig."""

        self._diagnostics.reset_measurement()

    def poll_due_frames(self, sample_sim_time: float) -> list[FramePacket]:
        if not self._active or self._sensor is None or sample_sim_time + 1e-12 < self._next_due_s:
            return []
        annotator = "depth_sensor_distance" if self.config.quality_level == "L1_stereo_noise" else "distance_to_image_plane"
        rgb_result = self._sensor.get_data("rgb")
        depth_result = self._sensor.get_data(annotator)
        # CameraSensor returns (data, metadata), while a graph-owned direct
        # reader returns the data object itself. Keep both ownership modes
        # behind the same FramePacket contract.
        rgb = rgb_result[0] if isinstance(rgb_result, tuple) else rgb_result
        depth = depth_result[0] if isinstance(depth_result, tuple) else depth_result
        if rgb is None or depth is None:
            return []
        capture_wall_time_ns = time.time_ns()
        imu_data = self._imu_sensor.get_data() if self._imu_sensor is not None else None
        packet = FramePacket.owned(
            episode_id=self._episode_id,
            snapshot_id=None,
            camera_name=self.name,
            frame_id=self._frame_id,
            sample_sim_time_ns=max(0, int(round(sample_sim_time * 1e9))),
            capture_wall_time_ns=capture_wall_time_ns,
            calibration_id=self.config.calibration_id,
            rgb=rgb,
            depth_m=depth,
            depth_semantics=("z_depth_stereo_noise" if self.config.quality_level == "L1_stereo_noise" else "z_depth_ideal_aligned"),
            source_state_seq=self._frame_id,
            rgb_encoding="rgb8",
            depth_encoding="32FC1_m",
            imu=imu_data,
        )
        packet.validate()
        self._diagnostics.record_frame(self._frame_id, sample_sim_time, capture_wall_time_ns)
        self._frame_id += 1
        self._next_due_s = max(self._next_due_s + self._period_s, sample_sim_time + self._period_s)
        return [packet]

    def diagnostics(self) -> dict[str, Any]:
        result = self._diagnostics.summary()
        render_product_details: dict[str, Any] = {}
        if self._stage is not None and self._render_product_path:
            try:
                prim = self._stage.GetPrimAtPath(self._render_product_path)
                render_product_details = {
                    "valid": bool(prim and prim.IsValid()),
                    "children": [
                        {
                            "path": str(child.GetPath()),
                            "type": child.GetTypeName(),
                        }
                        for child in prim.GetChildren()
                    ]
                    if prim and prim.IsValid()
                    else [],
                    "ordered_vars": [
                        str(path)
                        for path in prim.GetRelationship("orderedVars").GetTargets()
                    ]
                    if prim and prim.IsValid()
                    else [],
                }
            except Exception as exc:
                render_product_details = {"error": repr(exc)}
        result.update({
            "model": self.config.model,
            "quality_level": self.config.quality_level,
            "camera_prim": self._camera_path,
            "render_product": self._render_product_path,
            "render_product_details": render_product_details,
            "imu_enabled": self._imu_sensor is not None,
        })
        return result

    @property
    def render_product_path(self) -> str | None:
        return self._render_product_path

    @property
    def camera_prim_path(self) -> str | None:
        return self._camera_path

    @property
    def mount_prim_path(self) -> str | None:
        return self._mount_path

    @property
    def local_pose(self) -> LocalPose:
        return self._local_pose

    def reset(self, episode_id: int) -> None:
        self._episode_id = int(episode_id)
        self._frame_id = 0
        self._next_due_s = 0.0

    def close(self) -> None:
        self._active = False
        self._sensor = None
        self._imu_sensor = None

    def destroy(self) -> None:
        """Release runtime handles and remove only prims owned by this rig."""
        errors: list[BaseException] = []
        # CameraSensor owns its RenderProduct handle. The direct ROS camera
        # path creates a separate Replicator RenderProduct, whose destroy()
        # method also releases its Hydra texture and SyntheticData references.
        for resource in (
            self._sensor,
            self._imu_sensor,
            self._render_product_handle,
            self._camera_object,
        ):
            destroy = getattr(resource, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except BaseException as exc:
                    errors.append(exc)

        stage = self._stage
        if stage is not None:
            from pxr import Sdf

            # RenderProduct prims must be released through their owning API;
            # deleting the USD prim directly can leave Hydra with a dangling
            # render target and crash the next render pass.
            paths = [self._camera_prim_path]
            if self._mount_created:
                paths.append(self._mount_path)
            for path in paths:
                if not path:
                    continue
                try:
                    prim = stage.GetPrimAtPath(path)
                    if prim and prim.IsValid():
                        stage.RemovePrim(Sdf.Path(path))
                except BaseException as exc:
                    errors.append(exc)

        self.close()
        self._camera_object = None
        self._stage = None
        self._camera_path = None
        self._camera_prim_path = None
        self._render_product_handle = None
        self._render_product_path = None
        self._mount_path = None
        self._mount_created = False
        if errors:
            raise RuntimeError("failed to release RealSense resources: " + "; ".join(map(str, errors)))
