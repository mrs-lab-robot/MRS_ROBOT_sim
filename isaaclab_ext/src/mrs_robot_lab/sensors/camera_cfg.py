"""Isaac Lab camera wrappers backed by the runtime's canonical mount config."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from mrs_robot_lab.assets.asset_resolver import AssetResolver


_ROBOT_PRIM = "/World/OpenFlex"
_ENV_ROBOT_PRIM = "{ENV_REGEX_NS}/Robot"
_CAPTURE_CAMERA_MOUNTS = {
    "base_camera": "base_d435",
    "head_camera": "head_d435",
    "left_wrist_camera": "left_wrist_d405",
    "right_wrist_camera": "right_wrist_d405",
}


@dataclass(frozen=True)
class CameraMount:
    """Validated runtime camera mounting and image-interface parameters."""

    name: str
    prim_path: str
    parent_prim_path: str
    optical_frame: str
    width: int
    height: int
    focal_length_mm: float
    near_m: float
    far_m: float
    update_period_s: float
    translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    calibration_id: str | None = None


def load_camera_mounts(
    resolver: AssetResolver | None = None,
    *,
    robot_prim_path: str = _ENV_ROBOT_PRIM,
    sensor_frequencies_hz: Mapping[str, float] | None = None,
) -> tuple[CameraMount, ...]:
    """Read runtime camera mounts, optionally selecting capture IDs and rates.

    ``sensor_frequencies_hz=None`` preserves the full canonical mount listing.
    Passing a mapping makes the selection explicit and maps stable SessionConfig
    sensor IDs to the runtime's calibrated mount names.
    """

    requested_rates: dict[str, float] | None = None
    if sensor_frequencies_hz is not None:
        unknown = set(sensor_frequencies_hz) - set(_CAPTURE_CAMERA_MOUNTS)
        if unknown:
            raise ValueError(
                "unknown camera sensor ID(s): " + ", ".join(sorted(unknown))
            )
        requested_rates = {}
        for sensor_id, raw_frequency in sensor_frequencies_hz.items():
            if isinstance(raw_frequency, bool):
                raise ValueError(f"camera frequency for {sensor_id} must be positive")
            try:
                frequency = float(raw_frequency)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"camera frequency for {sensor_id} must be positive"
                ) from error
            if not math.isfinite(frequency) or frequency <= 0:
                raise ValueError(f"camera frequency for {sensor_id} must be positive")
            requested_rates[_CAPTURE_CAMERA_MOUNTS[sensor_id]] = frequency

    resource_resolver = resolver or AssetResolver()
    path = resource_resolver.resolve("openflex_camera_mounts")
    document = _read_yaml(path)
    cameras = document.get("cameras")
    if not isinstance(cameras, dict):
        raise ValueError(f"camera mount config has no cameras mapping: {path}")

    mounts: list[CameraMount] = []
    for name, value in cameras.items():
        if not isinstance(value, dict) or not value.get("enabled", False):
            continue
        if requested_rates is not None and name not in requested_rates:
            continue
        mount_path = value.get("mount_prim_path")
        parent_path = value.get("parent_prim")
        local_pose = value.get("local_pose")
        if not isinstance(mount_path, str) or not mount_path.startswith(_ROBOT_PRIM + "/"):
            raise ValueError(f"camera {name} mount must be inside {_ROBOT_PRIM}")
        if not isinstance(parent_path, str) or not parent_path.startswith(_ROBOT_PRIM + "/"):
            raise ValueError(f"camera {name} parent must be inside {_ROBOT_PRIM}")
        if not mount_path.startswith(parent_path + "/"):
            raise ValueError(f"camera {name} mount must be a descendant of its configured parent")
        if not isinstance(local_pose, dict):
            raise ValueError(f"camera {name} must define a local_pose mapping")
        translation = _finite_tuple(local_pose.get("translation_m"), 3, f"{name}.translation_m")
        quaternion = _finite_tuple(local_pose.get("quaternion_wxyz"), 4, f"{name}.quaternion_wxyz")
        norm = math.sqrt(sum(component * component for component in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"camera {name} quaternion_wxyz must be normalized")
        # width/height 是「硬件契约尺寸」（ROS 话题必须与真机一致，见
        # tests/unit/legacy_ros/test_robot_camera_resolution_matches_hardware.py）；
        # render_width/render_height 是渲染分辨率（性能旋钮），只影响 IsaacLab 的
        # TiledCamera render product，不影响对外发布的图像尺寸。
        width = _positive_integer(
            value.get("render_width", value.get("width")), f"{name}.render_width"
        )
        height = _positive_integer(
            value.get("render_height", value.get("height")), f"{name}.render_height"
        )
        rate = _positive_number(value.get("tick_rate_hz"), f"{name}.tick_rate_hz")
        focal = _positive_number(value.get("focal_length_mm"), f"{name}.focal_length_mm")
        near = _positive_number(value.get("near_m"), f"{name}.near_m")
        far = _positive_number(value.get("far_m"), f"{name}.far_m")
        if near >= far:
            raise ValueError(f"camera {name} near_m must be less than far_m")
        calibration_id = value.get("calibration_id")
        if calibration_id is not None and (
            not isinstance(calibration_id, str) or not calibration_id.strip()
        ):
            raise ValueError(f"camera {name}.calibration_id must be a non-empty string when set")
        relative_parent = parent_path[len(_ROBOT_PRIM):]
        mounts.append(
            CameraMount(
                name=name,
                prim_path=f"{robot_prim_path}{relative_parent}/{name}_LabCamera",
                parent_prim_path=f"{robot_prim_path}{relative_parent}",
                optical_frame=str(value.get("optical_frame", f"{name}_optical_frame")),
                width=width,
                height=height,
                focal_length_mm=focal,
                near_m=near,
                far_m=far,
                update_period_s=1.0 / (requested_rates.get(name, rate) if requested_rates is not None else rate),
                translation_m=translation,
                quaternion_wxyz=quaternion,
                calibration_id=calibration_id,
            )
        )
    if not mounts and requested_rates is None:
        raise ValueError(f"camera mount config contains no enabled cameras: {path}")
    if requested_rates is not None:
        found = {mount.name for mount in mounts}
        missing = set(requested_rates) - found
        if missing:
            raise ValueError(
                "selected camera mount(s) are not enabled in the runtime config: "
                + ", ".join(sorted(missing))
            )
    return tuple(mounts)


def build_camera_cfgs(
    resolver: AssetResolver | None = None,
    *,
    robot_prim_path: str = _ENV_ROBOT_PRIM,
    sensor_frequencies_hz: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build Isaac Lab wrappers on existing robot camera mounts."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    configs: dict[str, Any] = {}
    resource_resolver = resolver or AssetResolver()
    # 标定文件与挂点文件同目录（sim_runtime/config/sensors/realsense/calibration/）。
    # AssetResolver 只解析文件，故由挂点文件路径派生标定目录。
    mounts_path = Path(resource_resolver.resolve("openflex_camera_mounts"))
    calibration_dir = mounts_path.parent / "calibration"
    for mount in load_camera_mounts(
        resource_resolver,
        robot_prim_path=robot_prim_path,
        sensor_frequencies_hz=sensor_frequencies_hz,
    ):
        spawn_kwargs: dict[str, Any] = {
            "focal_length": mount.focal_length_mm,
            "clipping_range": (mount.near_m, mount.far_m),
        }
        if mount.calibration_id is not None:
            # 与经典链路 (rig.py) 相同：用标定 K 反算横向光圈
            # horizontal_aperture = focal_mm × calib_width / fx，还原相机真实 HFOV。
            # （此前未设置会回退到 20.955mm 默认光圈，HFOV≈146° 呈超广角/畸变。）
            # 纵向光圈交给 Isaac Lab 按图像宽高比推导，保证像素为正方形。
            calibration = _read_yaml(calibration_dir / f"{mount.calibration_id}.yaml")
            matrix = calibration.get("K")
            if not isinstance(matrix, (list, tuple)) or len(matrix) != 9:
                raise ValueError(f"camera {mount.name} calibration K must contain 9 values")
            fx = _positive_number(matrix[0], f"{mount.name}.calibration.fx")
            calibration_width = _positive_integer(
                calibration.get("width"), f"{mount.name}.calibration.width"
            )
            spawn_kwargs["horizontal_aperture"] = (
                mount.focal_length_mm * calibration_width / fx
            )
        configs[mount.name] = CameraCfg(
            prim_path=mount.prim_path,
            update_period=mount.update_period_s,
            height=mount.height,
            width=mount.width,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(**spawn_kwargs),
            offset=CameraCfg.OffsetCfg(
                pos=mount.translation_m,
                # 挂点姿态来自经典 USD 挂点，顺序为 wxyz；Isaac Lab OffsetCfg.rot 要求
                # (x, y, z, w)，且该值就是 USD 相机本体朝向，须用 "opengl" 约定直传，
                # 才能复现经典相机的朝向（此前直接传 wxyz 导致四个相机旋转全错）。
                rot=(
                    mount.quaternion_wxyz[1],
                    mount.quaternion_wxyz[2],
                    mount.quaternion_wxyz[3],
                    mount.quaternion_wxyz[0],
                ),
                convention="opengl",
            ),
        )
    return configs


def add_capture_cameras(
    scene: Any,
    sensor_frequencies_hz: Mapping[str, float],
    resolver: AssetResolver | None = None,
    *,
    sensor_factory=None,
) -> tuple[str, ...]:
    """Instantiate selected camera sensors after the task has spawned its robot.

    The current task environments author the robot and scene assets in
    ``_setup_scene`` instead of declaring them all in ``InteractiveSceneCfg``.
    Creating cameras from the scene config would therefore try to attach them
    before their parent robot links exist. This helper preserves the environment's
    single ``InteractiveScene``/``SimulationContext`` ownership and registers the
    cameras only after the robot and its clones are on the stage.
    """

    if not isinstance(sensor_frequencies_hz, Mapping):
        raise TypeError("sensor_frequencies_hz must be a mapping")
    if not sensor_frequencies_hz:
        return ()
    configs = build_camera_cfgs(
        resolver,
        robot_prim_path="{ENV_REGEX_NS}/Robot",
        sensor_frequencies_hz=sensor_frequencies_hz,
    )
    factory = sensor_factory
    registered: list[str] = []
    for name, camera_cfg in configs.items():
        camera_cfg.prim_path = camera_cfg.prim_path.format(
            ENV_REGEX_NS=scene.env_regex_ns
        )
        sensor = factory(camera_cfg) if factory is not None else camera_cfg.class_type(camera_cfg)
        scene.sensors[name] = sensor
        registered.append(name)
    return tuple(registered)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"could not read camera mount config {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"camera mount config root must be a mapping: {path}")
    return document


def _finite_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != size
        or any(isinstance(component, bool) or not isinstance(component, (int, float)) for component in value)
    ):
        raise ValueError(f"{name} must contain {size} numeric values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)
