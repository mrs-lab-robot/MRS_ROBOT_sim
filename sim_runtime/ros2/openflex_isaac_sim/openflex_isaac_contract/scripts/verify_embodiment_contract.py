#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import yaml

from openflex_isaac_contract.manifest import ManifestError, load_manifest, validate_manifest


def source_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "config" / "dependencies.repos").is_file():
            return parent
    return Path(__file__).resolve().parents[4]


def default_bringup_config_dir() -> Path:
    try:
        config_dir = Path(get_package_share_directory("openflex_isaac_bringup")) / "config"
        if config_dir.is_dir():
            return config_dir
    except PackageNotFoundError:
        pass

    for repo_dir in (Path.cwd(), source_repo_root()):
        config_dir = (
            repo_dir
            / "ros2_pkgs"
            / "openflex_isaac_sim"
            / "openflex_isaac_bringup"
            / "config"
        )
        if config_dir.is_dir():
            return config_dir
    return (
        source_repo_root()
        / "ros2_pkgs"
        / "openflex_isaac_sim"
        / "openflex_isaac_bringup"
        / "config"
    )


def default_sensor_config_path() -> Path:
    try:
        config_path = (
            Path(get_package_share_directory("openflex_isaac_bringup"))
            / "config"
            / "sensors.isaac.yaml"
        )
        if config_path.is_file():
            return config_path
    except PackageNotFoundError:
        pass
    return (
        source_repo_root()
        / "isaac_sim_core"
        / "config"
        / "sensor_params"
        / "sensors.isaac.yaml"
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def _controller_params(controllers: Mapping[str, Any], controller_name: str) -> Mapping[str, Any]:
    controller = controllers.get(controller_name, {})
    if not isinstance(controller, Mapping):
        return {}
    params = controller.get("ros__parameters", {})
    return params if isinstance(params, Mapping) else {}


def _actions_for_controller(manifest: Mapping[str, Any], controller_name: str) -> list[Mapping[str, Any]]:
    actions = manifest.get("actions", [])
    if not isinstance(actions, list):
        return []
    return [
        action
        for action in actions
        if isinstance(action, Mapping) and action.get("controller") == controller_name
    ]


def _controller_joints(controllers: Mapping[str, Any], controller_name: str) -> list[str]:
    params = _controller_params(controllers, controller_name)
    return [str(joint) for joint in params.get("joints", [])]


def verify_controller_contract(manifest: Mapping[str, Any], controllers: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    runtime = manifest.get("runtime", {})
    expected_update_rate = runtime.get("controller_manager_update_rate_hz") if isinstance(runtime, Mapping) else None
    manager_params = _controller_params(controllers, "controller_manager")
    if manager_params.get("update_rate") != expected_update_rate:
        errors.append("controller_manager update_rate does not match contract runtime")

    base_actions = _actions_for_controller(manifest, "swerve_drive_controller")
    if len(base_actions) != 1 or base_actions[0].get("name") != "base_twist":
        errors.append("contract must declare exactly one base_twist action for swerve_drive_controller")
    else:
        expected_topic = base_actions[0].get("topic")
        if _controller_params(controllers, "swerve_drive_controller").get("cmd_vel_topic") != expected_topic:
            errors.append("swerve_drive_controller cmd_vel_topic does not match contract base_twist topic")

    controller_names: list[str] = []
    for action in manifest.get("actions", []):
        if not isinstance(action, Mapping):
            continue
        controller_name = action.get("controller")
        if action.get("joints") and isinstance(controller_name, str) and controller_name not in controller_names:
            controller_names.append(controller_name)
    for controller_name in controller_names:
        controller_actions = _actions_for_controller(manifest, controller_name)
        if controller_name not in controllers and all(
            action.get("external_adapter") is True for action in controller_actions
        ):
            continue
        joints = [
            str(joint)
            for action in controller_actions
            for joint in action.get("joints", [])
        ]
        if _controller_joints(controllers, controller_name) != joints:
            errors.append(f"controller {controller_name} joints do not match contract")
    return errors


def _sensor_by_role(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    for sensor in manifest.get("sensors", []):
        if isinstance(sensor, Mapping) and sensor.get("role") == role:
            return sensor
    raise KeyError(role)


def verify_sensor_contract(manifest: Mapping[str, Any], sensors: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    target_topics = sensors.get("target_topics", {})
    role_to_config = {
        "base_camera": ("base_camera", "base_rgb_link"),
        "head_camera": ("head_camera", "head_link"),
        "left_wrist_camera": ("left_wrist_camera", "left_wrist_link"),
        "right_wrist_camera": ("right_wrist_camera", "right_wrist_link"),
    }
    frame_mappings = sensors.get("isaac_urdf_sensors", {})
    for role, (config_key, frame_key) in role_to_config.items():
        contract_sensor = _sensor_by_role(manifest, role)
        configured = target_topics.get(config_key, {})
        if not isinstance(configured, Mapping):
            errors.append(f"sensor role {role} is missing from sensors.isaac.yaml")
            continue
        if configured.get("color_image") != contract_sensor.get("color_topic"):
            errors.append(
                f"sensor role {role} color_topic {contract_sensor.get('color_topic')} "
                "is missing from sensors.isaac.yaml"
            )
        if configured.get("depth_image") != contract_sensor.get("depth_topic"):
            errors.append(
                f"sensor role {role} depth_topic {contract_sensor.get('depth_topic')} "
                "is missing from sensors.isaac.yaml"
            )
        if configured.get("color_camera_info") != contract_sensor.get("camera_info_topic"):
            errors.append(
                f"sensor role {role} camera_info_topic {contract_sensor.get('camera_info_topic')} "
                "is missing from sensors.isaac.yaml"
            )
        if configured.get("depth_camera_info") != contract_sensor.get("depth_camera_info_topic"):
            errors.append(
                f"sensor role {role} depth_camera_info_topic "
                f"{contract_sensor.get('depth_camera_info_topic')} is missing from sensors.isaac.yaml"
            )
        if not isinstance(frame_mappings, Mapping) or frame_mappings.get(frame_key) != contract_sensor.get("frame_id"):
            errors.append(f"sensor role {role} frame_id does not match isaac_urdf_sensors")

    lidar = _sensor_by_role(manifest, "lidar")
    lidar_config = target_topics.get("lidar", {})
    if not isinstance(lidar_config, Mapping) or lidar_config.get("pointcloud") != lidar.get("pointcloud_topic"):
        errors.append("lidar pointcloud topic does not match contract")
    if not isinstance(frame_mappings, Mapping) or frame_mappings.get("lidar_link") != lidar.get("frame_id"):
        errors.append("lidar frame_id does not match isaac_urdf_sensors")

    imu = _sensor_by_role(manifest, "imu")
    imu_config = target_topics.get("imu", {})
    if not isinstance(imu_config, Mapping) or imu_config.get("livox") != imu.get("topic"):
        errors.append("imu topic does not match contract")
    return errors


def parse_args() -> argparse.Namespace:
    bringup_config_dir = default_bringup_config_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--controllers",
        type=Path,
        default=bringup_config_dir / "controllers.isaac.mobile_base.yaml",
    )
    parser.add_argument(
        "--sensors",
        type=Path,
        default=default_sensor_config_path(),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.contract)
        errors = validate_manifest(manifest)
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            return 1
        controllers = load_yaml(args.controllers)
        sensors = load_yaml(args.sensors)
    except FileNotFoundError as error:
        missing_path = error.filename or "the requested verifier config"
        print(
            f"[ERROR] Missing verifier config: {missing_path}. Build and source "
            "openflex_isaac_bringup, or pass --controllers/--sensors explicitly."
        )
        return 1
    except (ManifestError, yaml.YAMLError, ValueError, OSError) as error:
        print(f"[ERROR] {error}")
        return 1
    errors.extend(verify_controller_contract(manifest, controllers))
    errors.extend(verify_sensor_contract(manifest, sensors))
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1
    print("[OK] OpenFleX embodiment contract matches Isaac controller and sensor configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
