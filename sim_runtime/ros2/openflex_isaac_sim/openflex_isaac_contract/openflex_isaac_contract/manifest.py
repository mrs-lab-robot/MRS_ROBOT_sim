from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ManifestError(RuntimeError):
    """Raised when the OpenFleX embodiment manifest cannot be loaded."""


REQUIRED_ACTION_NAMES = {
    "base_twist",
    "lift_position",
    "lift_velocity",
    "head_position",
    "left_arm_position",
    "left_gripper_position",
    "right_arm_position",
    "right_gripper_position",
}
REQUIRED_SENSOR_ROLES = {
    "base_camera",
    "head_camera",
    "left_wrist_camera",
    "right_wrist_camera",
    "lidar",
    "imu",
    "odom",
}
REQUIRED_LAYOUT_OBSERVATIONS = {"joint_state", "base_odom", "base_twist", "livox_imu"}
ALLOWED_UNITS = {"m", "rad", "m/s", "rad/s", "m/s^2", "unitless"}
SUPPORTED_ISAAC_RUNTIMES = {"5.1", "6.0"}


def default_manifest_path() -> Path:
    module_path = Path(__file__).resolve()
    source_path = module_path.parents[1] / "config" / "embodiment.yaml"
    if source_path.is_file():
        return source_path
    install_paths = (
        module_path.parents[4]
        / "share"
        / "openflex_isaac_contract"
        / "config"
        / "embodiment.yaml",
        module_path.parents[5]
        / "share"
        / "openflex_isaac_contract"
        / "config"
        / "embodiment.yaml",
    )
    for install_path in install_paths:
        if install_path.is_file():
            return install_path
    return install_paths[0]


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = default_manifest_path() if path is None else Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"OpenFleX embodiment manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ManifestError(f"OpenFleX embodiment manifest must be a mapping: {manifest_path}")
    return data


def component_names(manifest: Mapping[str, Any], section: str) -> list[str]:
    values = manifest.get(section, [])
    if not isinstance(values, list):
        return []
    return [str(value.get("name", "")) for value in values if isinstance(value, Mapping)]


def action_dimension(manifest: Mapping[str, Any]) -> int:
    total = 0
    for action in manifest.get("actions", []):
        if isinstance(action, Mapping):
            dimension = action.get("dimension")
            if (
                action.get("counts_toward_action", True)
                and isinstance(dimension, int)
                and not isinstance(dimension, bool)
                and dimension > 0
            ):
                total += dimension
    return total


def _duplicate_name_errors(section: str, values: object) -> list[str]:
    if not isinstance(values, list):
        return [f"{section} must be a list"]

    errors: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            errors.append(f"{section} item must be a mapping")
            continue
        name = str(value.get("name", ""))
        if not name:
            errors.append(f"{section} item is missing name")
        elif name in seen:
            errors.append(f"{section} contains duplicate name: {name}")
        seen.add(name)
    return errors


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_action(
    action: Mapping[str, Any], topic_records: Mapping[str, Mapping[str, Any]], joints: set[str]
) -> list[str]:
    errors: list[str] = []
    name = str(action.get("name", ""))
    dimension = action.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        return [f"actions {name} dimension must be a positive integer"]

    for key in ("command_type", "topic", "controller", "limit_source", "frequency_source"):
        if not isinstance(action.get(key), str) or not action[key]:
            errors.append(f"actions {name} is missing required field: {key}")

    for key in ("fields", "units", "limits"):
        value = action.get(key)
        if not isinstance(value, list) or len(value) != dimension:
            errors.append(f"actions {name} {key} length must match dimension {dimension}")

    fields = action.get("fields", [])
    if isinstance(fields, list) and any(not isinstance(field, str) or not field for field in fields):
        errors.append(f"actions {name} fields must contain non-empty strings")

    units = action.get("units", [])
    if isinstance(units, list):
        for unit in units:
            if not isinstance(unit, str):
                errors.append(f"actions {name} contains unsupported unit type: {type(unit).__name__}")
            elif unit not in ALLOWED_UNITS:
                errors.append(f"actions {name} contains unsupported unit: {unit}")

    limits = action.get("limits", [])
    if isinstance(limits, list):
        for index, limit in enumerate(limits):
            if not isinstance(limit, Mapping):
                errors.append(f"actions {name} limit {index} must be a mapping")
                continue
            minimum = limit.get("minimum")
            maximum = limit.get("maximum")
            if not _is_number(minimum) or not _is_number(maximum) or minimum > maximum:
                errors.append(f"actions {name} limit {index} must define ordered numeric minimum and maximum")

    frequency = action.get("default_command_frequency_hz")
    if not _is_number(frequency) or frequency <= 0:
        errors.append(f"actions {name} default_command_frequency_hz must be positive")
    safety_clipping = action.get("safety_clipping")
    if not isinstance(safety_clipping, str) or not safety_clipping:
        errors.append(f"actions {name} safety_clipping must be a non-empty string")

    topic = action.get("topic")
    if isinstance(topic, str) and topic:
        topic_record = topic_records.get(topic)
        if topic_record is None:
            errors.append(f"actions {name} topic is not declared in ros2_topics: {topic}")
        elif topic_record.get("direction") != "command" or topic_record.get("type") != action.get("command_type"):
            errors.append(f"actions {name} topic declaration does not match command_type: {topic}")

    action_joints = action.get("joints")
    if action_joints is not None:
        if not isinstance(action_joints, list) or len(action_joints) != dimension:
            errors.append(f"actions {name} joints length must match dimension {dimension}")
        elif any(not isinstance(joint, str) or not joint for joint in action_joints):
            errors.append(f"actions {name} joints must contain non-empty strings")
        else:
            alias_name = action.get("joint_alias_of")
            for joint in action_joints:
                if joint in joints and not alias_name:
                    errors.append(f"actions contains duplicate joint: {joint}")
                joints.add(joint)
    return errors


def _validate_observation_layout(manifest: Mapping[str, Any]) -> list[str]:
    layouts = manifest.get("observation_layout")
    if not isinstance(layouts, list):
        return ["observation_layout must be a list"]

    errors: list[str] = []
    observation_names = set(component_names(manifest, "observations"))
    seen: set[str] = set()
    for layout in layouts:
        if not isinstance(layout, Mapping):
            errors.append("observation_layout item must be a mapping")
            continue
        observation = str(layout.get("observation", ""))
        if not observation:
            errors.append("observation_layout item is missing observation")
        elif observation in seen:
            errors.append(f"observation_layout contains duplicate observation: {observation}")
        elif observation not in observation_names:
            errors.append(f"observation_layout references unknown observation: {observation}")
        seen.add(observation)

        dimension = layout.get("dimension")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            errors.append(f"observation_layout {observation} dimension must be a positive integer")
            continue
        for key in ("fields", "order", "units"):
            value = layout.get(key)
            if not isinstance(value, list) or len(value) != dimension:
                errors.append(
                    f"observation_layout {observation} {key} length must match dimension {dimension}"
                )
        fields = layout.get("fields")
        order = layout.get("order")
        if isinstance(fields, list) and isinstance(order, list) and fields != order:
            errors.append(f"observation_layout {observation} order must match fields")
        units = layout.get("units")
        if isinstance(units, list):
            for unit in units:
                if not isinstance(unit, str):
                    errors.append(
                        f"observation_layout {observation} contains unsupported unit type: "
                        f"{type(unit).__name__}"
                    )
                elif unit not in ALLOWED_UNITS:
                    errors.append(f"observation_layout {observation} contains unsupported unit: {unit}")
        if not isinstance(layout.get("source"), str) or not layout.get("source"):
            errors.append(f"observation_layout {observation} is missing source")

    for observation in sorted(REQUIRED_LAYOUT_OBSERVATIONS - seen):
        errors.append(f"observation_layout is missing required observation: {observation}")
    return errors


def validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    robot = manifest.get("robot")
    if not isinstance(robot, Mapping):
        errors.append("robot must be a mapping")
    else:
        if robot.get("id") != "openflex":
            errors.append("robot.id must be openflex")
        if robot.get("active_isaac_runtime") not in SUPPORTED_ISAAC_RUNTIMES:
            errors.append(
                "robot.active_isaac_runtime must be one of: "
                + ", ".join(sorted(SUPPORTED_ISAAC_RUNTIMES))
            )
        if robot.get("real_domain_id") != 0:
            errors.append("robot.real_domain_id must be 0")
        if not isinstance(robot.get("sim_domain_id"), int):
            errors.append("robot.sim_domain_id must be an integer")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("runtime must be a mapping")
    else:
        if runtime.get("isaac_sim_version") not in SUPPORTED_ISAAC_RUNTIMES:
            errors.append(
                "runtime.isaac_sim_version must be one of: "
                + ", ".join(sorted(SUPPORTED_ISAAC_RUNTIMES))
            )
        if not _is_number(runtime.get("controller_manager_update_rate_hz")):
            errors.append("runtime.controller_manager_update_rate_hz must be numeric")
        if not isinstance(runtime.get("source"), str) or not runtime.get("source"):
            errors.append("runtime is missing source")

    domains = manifest.get("domains")
    if not isinstance(domains, Mapping):
        errors.append("domains must be a mapping")
    else:
        physical = domains.get("physical")
        simulation = domains.get("simulation")
        if not isinstance(physical, Mapping) or physical.get("ros_domain_id") != 0:
            errors.append("domains.physical.ros_domain_id must be 0")
        expected_sim_domain = robot.get("sim_domain_id") if isinstance(robot, Mapping) else None
        if not isinstance(simulation, Mapping) or simulation.get("ros_domain_id") != expected_sim_domain:
            errors.append(
                "domains.simulation.ros_domain_id must match robot.sim_domain_id"
            )
        if not isinstance(domains.get("source"), str) or not domains.get("source"):
            errors.append("domains is missing source")

    errors.extend(_duplicate_name_errors("actions", manifest.get("actions")))
    errors.extend(_duplicate_name_errors("observations", manifest.get("observations")))
    errors.extend(_duplicate_name_errors("ros2_topics", manifest.get("ros2_topics")))

    topics = manifest.get("ros2_topics")
    topic_records: dict[str, Mapping[str, Any]] = {}
    if isinstance(topics, list):
        for topic in topics:
            if not isinstance(topic, Mapping):
                continue
            name = topic.get("name")
            if not isinstance(name, str) or not name:
                continue
            if topic.get("direction") not in {"command", "observation"}:
                errors.append(f"ros2_topics {name} has unsupported direction")
            if not isinstance(topic.get("type"), str) or not topic.get("type"):
                errors.append(f"ros2_topics {name} is missing type")
            topic_records[name] = topic

    actions = manifest.get("actions")
    action_names: set[str] = set()
    joints: set[str] = set()
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            action_names.add(str(action.get("name", "")))
            errors.extend(_validate_action(action, topic_records, joints))
    for name in sorted(REQUIRED_ACTION_NAMES - action_names):
        errors.append(f"actions is missing required name: {name}")

    sensors = manifest.get("sensors")
    if not isinstance(sensors, list):
        errors.append("sensors must be a list")
    else:
        roles: set[str] = set()
        for sensor in sensors:
            if not isinstance(sensor, Mapping):
                errors.append("sensors item must be a mapping")
                continue
            role = str(sensor.get("role", ""))
            if not role:
                errors.append("sensors item is missing role")
            elif role in roles:
                errors.append(f"sensors contains duplicate role: {role}")
            roles.add(role)

            required_fields = {
                "base_camera": ("frame_id", "color_topic", "depth_topic", "camera_info_topic", "depth_camera_info_topic"),
                "head_camera": ("frame_id", "color_topic", "depth_topic", "camera_info_topic", "depth_camera_info_topic"),
                "left_wrist_camera": ("frame_id", "color_topic", "depth_topic", "camera_info_topic", "depth_camera_info_topic"),
                "right_wrist_camera": ("frame_id", "color_topic", "depth_topic", "camera_info_topic", "depth_camera_info_topic"),
                "lidar": ("frame_id", "pointcloud_topic", "raw_pointcloud_topic"),
                "imu": ("frame_id", "topic"),
                "odom": ("frame_id", "child_frame_id", "topic"),
            }.get(role, ())
            for field in required_fields:
                value = sensor.get(field)
                if not isinstance(value, str) or not value:
                    errors.append(f"sensors {role} is missing required field: {field}")
                elif field.endswith("topic") and value not in topic_records:
                    errors.append(f"sensors {role} topic is not declared in ros2_topics: {value}")
        for role in sorted(REQUIRED_SENSOR_ROLES - roles):
            errors.append(f"sensors is missing required role: {role}")

    observations = manifest.get("observations")
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, Mapping):
                continue
            name = str(observation.get("name", ""))
            topic = observation.get("topic")
            topic_record = topic_records.get(topic) if isinstance(topic, str) else None
            if topic_record is None:
                errors.append(f"observations {name} topic is not declared in ros2_topics: {topic}")
            elif topic_record.get("direction") != "observation" or topic_record.get("type") != observation.get("type"):
                errors.append(f"observations {name} topic declaration does not match type: {topic}")

    profiles = manifest.get("controller_profiles")
    if not isinstance(profiles, Mapping):
        errors.append("controller_profiles must be a mapping")
    else:
        parity_profile = profiles.get("ros2_physical_parity")
        if not isinstance(parity_profile, Mapping):
            errors.append("controller_profiles must define ros2_physical_parity")
        elif parity_profile.get("action_dimension") != action_dimension(manifest):
            errors.append(
                "ros2_physical_parity.action_dimension must match counted action dimension "
                f"{action_dimension(manifest)}"
            )
    errors.extend(_validate_observation_layout(manifest))
    return errors
