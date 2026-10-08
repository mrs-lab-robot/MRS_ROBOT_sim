#!/usr/bin/python3
"""Generate an Isaac Sim URDF from the authoritative integrated OpenFleX xacro."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
import xacro


FRAME_LINK_MASS = "0.001"
FRAME_LINK_INERTIA = "0.000001"
DEFAULT_VISUAL_LINK_MASS = "0.001"
DEFAULT_VISUAL_LINK_INERTIA = "0.000001"
ISAAC_WHEEL_COLLISION_RADIUS = "0.075"
ISAAC_WHEEL_COLLISION_LENGTH = "0.06"
ISAAC_LIFT_COLLISION_BOXES: dict[str, tuple[str, str]] = {
    # Axis-aligned bounds of the original collision meshes in link-local metres.
    "lift_base_link": ("0.00675 0.0407973 0.63", "0.1835 0.1712 1.5298"),
    "lift_carriage_link": ("-0.00375 0.0215 0", "0.1925 0.347 0.23"),
}
USD_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
USD_IDENTIFIER_CHAR_RE = re.compile(r"[^A-Za-z0-9_]")
REAL_ROBOT_BLACK = "0.020 0.035 0.043 1.0"
REAL_ROBOT_GRAPHITE = "0.106 0.122 0.125 1.0"
REAL_ROBOT_RUBBER = "0.015 0.019 0.020 1.0"
REAL_ROBOT_COLUMN_ALUMINUM = "0.486 0.494 0.478 1.0"
REAL_ROBOT_BRUSHED_ALUMINUM = "0.784 0.800 0.816 1.0"

STEERING_JOINTS = ("fl_steering_joint", "fr_steering_joint", "bl_steering_joint", "br_steering_joint")
WHEEL_JOINTS = ("fl_wheel_joint", "fr_wheel_joint", "bl_wheel_joint", "br_wheel_joint")
ARM_JOINTS = tuple(
    f"openarmx_{side}_joint{index}" for side in ("left", "right") for index in range(1, 8)
)
LIFT_VELOCITY_JOINTS = ("lift_joint",)
# The default Isaac control graph is position-driven for the arms. The bundled
# Isaac ROS controller still reads a flattened command-interface list while
# initializing each joint, so retain the arm effort entries required by that
# legacy contract even though normal arm commands remain position-driven.
EFFORT_JOINTS = ARM_JOINTS
FINGER_JOINTS = (
    "openarmx_left_finger_joint1",
    "openarmx_left_finger_joint2",
    "openarmx_right_finger_joint1",
    "openarmx_right_finger_joint2",
)
HEAD_JOINTS = ("openarmx_head_yaw_joint", "openarmx_head_pitch_joint")
POSITION_JOINTS = (
    *STEERING_JOINTS,
    "lift_joint",
    *ARM_JOINTS,
    *FINGER_JOINTS,
    *HEAD_JOINTS,
)
# Keep arm drive settings aligned with the known-good Isaac importer path.
# The Isaac-specific tag is applied after direct USD loading by the simulator
# spawn hook, which restores the same runtime dynamics as the old importer.
LINK_INERTIAL_OVERRIDES: dict[str, tuple[str, str, str, str]] = {}
LINK_INERTIAL_ORIGIN_OVERRIDES: dict[str, tuple[str, str]] = {}
DRIVE_PROFILES: dict[str, dict[str, str]] = {
    "lift": {"stiffness": "100000000.0", "damping": "500000.0", "joint_friction": "50.0"},
    "arm": {
        "stiffness": "100000000.0",
        "damping": "300000.0",
        "joint_friction": "5.0",
    },
    "finger": {"stiffness": "10000000.0", "damping": "50000.0", "joint_friction": "1.0"},
    "head": {"stiffness": "30000000.0", "damping": "100000.0", "joint_friction": "2.0"},
}
# Isaac Sim 6 creates the generic OmniLidar asset first; launch_sensor.py then
# authors the MID360 approximation attributes on that prim.
LIVOX_APPROX_LIDAR_CONFIG = os.environ.get(
    "OPENFLEX_ISAAC_LIDAR_CONFIG", "Example_Rotary"
).strip() or "Example_Rotary"
# Isaac Sim 5.1 URDF camera adjustment point. Values are relative to each
# camera's parent link and use ROS URDF units: xyz in metres, rpy in radians.
# RGB and depth sensors for the same physical camera intentionally share a pose.
CAMERA_MOUNTS: dict[str, tuple[str, str]] = {
    "base": ("0.01 0 0", "-3.1415926 0 0"),
    "head": ("-0.0023  0.0813 0.0605", "1.57079632679 0 0"),
    "left_hand": ("0.09 0 0 ", "0 2.3562 0"),
    "right_hand": ("0.09 0 0", "0 2.3562 0"),
}
# Intel RealSense nominal fields of view from the public realsense2_description
# model used by agilexrobotics/piper_isaac_sim. Values are radians.
CAMERA_OPTICS = {
    "d435i": {"rgb": (math.radians(69.4), math.radians(42.5)),
              "depth": (math.radians(87.0), math.radians(58.0))},
    "d405": {"rgb": (math.radians(87.0), math.radians(58.0)),
             "depth": (math.radians(87.0), math.radians(58.0))},
}
PHYSICS_MATERIALS = {
    "openflex_rubber_black": {
        "rgba": REAL_ROBOT_RUBBER,
        "static_friction": "1.25",
        "dynamic_friction": "1.05",
        "restitution": "0.0",
    },
    "openflex_black": {
        "rgba": REAL_ROBOT_BLACK,
        "static_friction": "0.80",
        "dynamic_friction": "0.65",
        "restitution": "0.0",
    },
    "openflex_graphite": {
        "rgba": REAL_ROBOT_GRAPHITE,
        "static_friction": "0.75",
        "dynamic_friction": "0.60",
        "restitution": "0.0",
    },
    "openflex_column_aluminum": {
        "rgba": REAL_ROBOT_COLUMN_ALUMINUM,
        "static_friction": "0.65",
        "dynamic_friction": "0.50",
        "restitution": "0.0",
    },
    "openflex_sensor_black": {
        "rgba": REAL_ROBOT_RUBBER,
        "static_friction": "0.60",
        "dynamic_friction": "0.45",
        "restitution": "0.0",
    },
    "openflex_brushed_aluminum": {
        "rgba": REAL_ROBOT_BRUSHED_ALUMINUM,
        "static_friction": "0.55",
        "dynamic_friction": "0.45",
        "restitution": "0.0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Authoritative integrated xacro.")
    parser.add_argument("--output", required=True, type=Path, help="Generated Isaac URDF path.")
    parser.add_argument("--robot-name", default="openflex", help="Robot name used by Isaac prim paths.")
    parser.add_argument("--command-topic", default="/openflex/joint_command")
    parser.add_argument("--state-topic", default="/openflex/joint_states")
    parser.add_argument(
        "--keep-package-uris",
        action="store_true",
        help="Keep package:// mesh paths instead of resolving them to file:// URIs.",
    )
    parser.add_argument(
        "--preserve-visual-materials",
        action="store_true",
        help="Keep visual materials from the source xacro for RViz robot_description output.",
    )
    parser.add_argument(
        "--hide-head-camera",
        action="store_true",
        help="Remove only the physical Femto camera mesh from the head visual model.",
    )
    parser.add_argument(
        "--hide-lift-mast",
        action="store_true",
        help="Remove only the tall lift mast visual while preserving collision and control.",
    )
    parser.add_argument(
        "--disable-sensors",
        action="store_true",
        help="Do not add Isaac RGB, depth, or LiDAR sensors.",
    )
    parser.add_argument(
        "--sensor-profile",
        choices=("none", "rgb", "rgb_depth", "lidar", "data", "teleop"),
        default="data",
        help="Isaac-only sensor subset; data enables RGB, depth and LiDAR.",
    )
    parser.add_argument("--camera-width", type=int, default=224)
    parser.add_argument("--camera-height", type=int, default=224)
    parser.add_argument(
        "--mid360-xyz",
        default=os.environ.get("OPENFLEX_MID360_XYZ", "0.30 0.0 0.12"),
        help="MID360 translation relative to base_link, in metres (x y z).",
    )
    parser.add_argument(
        "--mid360-rpy",
        default=os.environ.get("OPENFLEX_MID360_RPY", "-0.5236 0 -1.5708"),
        help="MID360 rotation relative to base_link, in radians (roll pitch yaw).",
    )
    parser.add_argument("--enable-head", default="true", choices=("true", "false"))
    args = parser.parse_args()
    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    return args


def add_element(parent: ET.Element, tag: str, **attributes: str) -> ET.Element:
    return ET.SubElement(parent, tag, {key: value for key, value in attributes.items() if value is not None})


def add_text(parent: ET.Element, tag: str, text: str, **attributes: str) -> ET.Element:
    element = add_element(parent, tag, **attributes)
    element.text = text
    return element


def remove_existing_simulation_blocks(root: ET.Element) -> tuple[int, int, int]:
    removed_ros2_control = 0
    removed_mujoco = 0
    removed_gazebo = 0
    for element in list(root.findall("ros2_control")):
        root.remove(element)
        removed_ros2_control += 1
    for element in list(root.findall("mujoco")):
        root.remove(element)
        removed_mujoco += 1
    for element in list(root.findall("gazebo")):
        root.remove(element)
        removed_gazebo += 1
    return removed_ros2_control, removed_mujoco, removed_gazebo


def add_default_inertials(root: ET.Element) -> tuple[int, int]:
    added = 0
    overridden = 0
    for link in root.findall("link"):
        link_name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is not None:
            if link_name in LINK_INERTIAL_OVERRIDES:
                set_inertial_values(inertial, LINK_INERTIAL_OVERRIDES[link_name])
                set_inertial_origin_override(inertial, link_name)
                overridden += 1
            continue
        mass, ixx, iyy, izz = default_inertial_values(link)
        inertial = add_element(link, "inertial")
        add_element(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        set_inertial_values(inertial, (mass, ixx, iyy, izz))
        set_inertial_origin_override(inertial, link_name)
        added += 1
    return added, overridden


def remove_massless_fixed_frame_inertials(root: ET.Element) -> int:
    """避免把纯坐标系导入为 articulation 内的刚体。

    集成 xacro 中有 `base_footprint`、光学坐标系等纯固定坐标系。为
    方便 URDF 完整性，前面的默认惯性补全过程会给它们添加极小质量；
    Isaac 的导入器会把这些链接也解释成刚体，进而可能在根链接处报
    closed-articulation 错误。没有 visual/collision 的固定坐标系不参与
    碰撞或动力学，移除其 inertial 不影响 TF 或 ros2_control 接口。
    """
    fixed_children = {
        joint.find("child").get("link", "")
        for joint in root.findall("joint")
        if joint.get("type") == "fixed" and joint.find("child") is not None
    }
    removed = 0
    for link in root.findall("link"):
        if link.get("name", "") not in fixed_children:
            continue
        if link.find("visual") is not None or link.find("collision") is not None:
            continue
        inertial = link.find("inertial")
        if inertial is not None:
            link.remove(inertial)
            removed += 1
    return removed


def set_inertial_origin_override(inertial: ET.Element, link_name: str) -> None:
    origin_values = LINK_INERTIAL_ORIGIN_OVERRIDES.get(link_name)
    if origin_values is None:
        return
    xyz, rpy = origin_values
    origin = inertial.find("origin")
    if origin is None:
        origin = add_element(inertial, "origin")
    origin.set("xyz", xyz)
    origin.set("rpy", rpy)


def set_inertial_values(inertial: ET.Element, values: tuple[str, str, str, str]) -> None:
    mass, ixx, iyy, izz = values
    mass_element = inertial.find("mass")
    if mass_element is None:
        mass_element = add_element(inertial, "mass")
    mass_element.set("value", mass)

    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = add_element(inertial, "inertia")
    inertia.set("ixx", ixx)
    inertia.set("iyy", iyy)
    inertia.set("izz", izz)
    for key in ("ixy", "ixz", "iyz"):
        if inertia.get(key) is None:
            inertia.set(key, "0")


def default_inertial_values(link: ET.Element) -> tuple[str, str, str, str]:
    link_name = link.get("name", "")
    if link_name in LINK_INERTIAL_OVERRIDES:
        return LINK_INERTIAL_OVERRIDES[link_name]

    has_geometry = link.find("visual") is not None or link.find("collision") is not None
    if has_geometry:
        return (
            DEFAULT_VISUAL_LINK_MASS,
            DEFAULT_VISUAL_LINK_INERTIA,
            DEFAULT_VISUAL_LINK_INERTIA,
            DEFAULT_VISUAL_LINK_INERTIA,
        )

    return FRAME_LINK_MASS, FRAME_LINK_INERTIA, FRAME_LINK_INERTIA, FRAME_LINK_INERTIA


def usd_safe_stem(stem: str) -> str:
    safe = USD_IDENTIFIER_CHAR_RE.sub("_", stem)
    if not safe or safe[0].isdigit():
        safe = f"mesh_{safe}"
    return safe


def ensure_mesh_alias(source: Path, alias_dir: Path, alias_cache: dict[Path, Path]) -> Path:
    source = source.resolve()
    if source in alias_cache:
        return alias_cache[source]

    alias_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = usd_safe_stem(source.stem)
    candidate = alias_dir / f"{safe_stem}{source.suffix}"
    index = 2
    while candidate.exists():
        try:
            if candidate.resolve() == source:
                alias_cache[source] = candidate
                return candidate
        except OSError:
            pass
        candidate = alias_dir / f"{safe_stem}_{index}{source.suffix}"
        index += 1

    try:
        candidate.symlink_to(source)
    except OSError:
        shutil.copy2(source, candidate)
    alias_cache[source] = candidate
    return candidate


def resolve_package_mesh_uris(root: ET.Element, alias_dir: Path) -> tuple[int, int]:
    resolved = 0
    aliases = 0
    alias_cache: dict[Path, Path] = {}
    for mesh in root.findall(".//mesh"):
        uri = mesh.get("filename", "")
        if not uri.startswith("package://"):
            continue
        package_and_path = uri.removeprefix("package://")
        package, separator, relative_path = package_and_path.partition("/")
        if not separator:
            raise RuntimeError(f"Invalid package URI in mesh: {uri}")
        try:
            mesh_path = Path(get_package_share_directory(package)) / relative_path
        except PackageNotFoundError as error:
            raise RuntimeError(f"Cannot resolve mesh package URI: {uri}") from error
        if not mesh_path.is_file():
            raise RuntimeError(f"Mesh URI resolves to a missing file: {mesh_path}")
        resolved_path = mesh_path.resolve()
        if not USD_IDENTIFIER_RE.match(resolved_path.stem):
            resolved_path = ensure_mesh_alias(resolved_path, alias_dir, alias_cache)
            aliases += 1
        mesh.set("filename", resolved_path.as_uri())
        resolved += 1
    return resolved, aliases


def visual_mesh_basename(visual: ET.Element) -> str:
    mesh = visual.find("./geometry/mesh")
    if mesh is None:
        return ""
    return Path(mesh.get("filename", "")).name


def set_visual_material(visual: ET.Element, name: str, rgba: str) -> bool:
    changed = False
    material = visual.find("material")
    if material is None:
        material = add_element(visual, "material", name=name)
        changed = True
    elif material.get("name") != name:
        material.set("name", name)
        changed = True

    color = material.find("color")
    if color is None:
        add_element(material, "color", rgba=rgba)
        changed = True
    elif color.get("rgba") != rgba:
        color.set("rgba", rgba)
        changed = True
    return changed


def apply_physics_materials(root: ET.Element) -> int:
    changed = 0
    root_materials = {material.get("name", ""): material for material in root.findall("material")}
    for name, config in PHYSICS_MATERIALS.items():
        material = root_materials.get(name)
        if material is None:
            material = add_element(root, "material", name=name)
            root_materials[name] = material
            changed += 1

        color = material.find("color")
        if color is None:
            add_element(material, "color", rgba=config["rgba"])
            changed += 1
        elif color.get("rgba") != config["rgba"]:
            color.set("rgba", config["rgba"])
            changed += 1

        rigid_body = material.find("isaac_rigid_body")
        if rigid_body is None:
            rigid_body = add_element(material, "isaac_rigid_body")
            changed += 1
        for key in ("static_friction", "dynamic_friction", "restitution"):
            if rigid_body.get(key) != config[key]:
                rigid_body.set(key, config[key])
                changed += 1
    return changed


def real_robot_material(link_name: str, visual: ET.Element) -> tuple[str, str] | None:
    mesh_name = visual_mesh_basename(visual).lower()

    if link_name == "lift_base_link":
        return "openflex_column_aluminum", REAL_ROBOT_COLUMN_ALUMINUM

    if link_name in ("base_link", "chest_link", "lift_carriage_link"):
        return "openflex_black", REAL_ROBOT_BLACK

    if link_name in ("left_link0_base", "right_link0_base"):
        return "openflex_graphite", REAL_ROBOT_GRAPHITE

    if link_name.endswith("_wheel_link"):
        return "openflex_rubber_black", REAL_ROBOT_RUBBER

    if link_name in ("mid360_link", "d435_link"):
        return "openflex_sensor_black", REAL_ROBOT_RUBBER

    if link_name.startswith("openarmx_"):
        if link_name.endswith("_finger") or mesh_name == "finger.dae":
            return "openflex_brushed_aluminum", REAL_ROBOT_BRUSHED_ALUMINUM
        return "openflex_black", REAL_ROBOT_BLACK

    if link_name.startswith("head_"):
        if mesh_name.startswith("rs00"):
            return "openflex_graphite", REAL_ROBOT_GRAPHITE
        if "camera" in mesh_name:
            return "openflex_sensor_black", REAL_ROBOT_RUBBER
        return "openflex_black", REAL_ROBOT_BLACK

    return None


def apply_real_robot_appearance(root: ET.Element) -> int:
    changed = 0
    for link in root.findall("link"):
        link_name = link.get("name", "")
        for visual in link.findall("visual"):
            material = real_robot_material(link_name, visual)
            if material is None:
                continue
            if set_visual_material(visual, *material):
                changed += 1
    return changed


def replace_wheel_collisions_with_cylinders(root: ET.Element) -> int:
    changed = 0
    for link in root.findall("link"):
        link_name = link.get("name", "")
        if not link_name.endswith("_wheel_link"):
            continue
        for collision in list(link.findall("collision")):
            link.remove(collision)
        collision = add_element(link, "collision")
        add_element(collision, "origin", xyz="0 0 0", rpy="1.5708 0 0")
        geometry = add_element(collision, "geometry")
        add_element(
            geometry,
            "cylinder",
            radius=ISAAC_WHEEL_COLLISION_RADIUS,
            length=ISAAC_WHEEL_COLLISION_LENGTH,
        )
        changed += 1
    return changed


def replace_high_poly_lift_collisions_with_boxes(root: ET.Element) -> int:
    changed = 0
    for link in root.findall("link"):
        proxy = ISAAC_LIFT_COLLISION_BOXES.get(link.get("name", ""))
        if proxy is None:
            continue

        origin_xyz, box_size = proxy
        for collision in list(link.findall("collision")):
            link.remove(collision)
        collision = add_element(link, "collision")
        add_element(collision, "origin", xyz=origin_xyz, rpy="0 0 0")
        geometry = add_element(collision, "geometry")
        add_element(geometry, "box", size=box_size)
        changed += 1
    return changed


def mechanical_joint_names(root: ET.Element) -> set[str]:
    return {
        joint.get("name", "")
        for joint in root.findall("joint")
        if joint.get("name") and joint.get("type") not in ("fixed", None)
    }


def drive_profile_for_joint(joint_name: str) -> dict[str, str] | None:
    if joint_name in WHEEL_JOINTS:
        return DRIVE_PROFILES.get("wheel")
    if joint_name in STEERING_JOINTS:
        return DRIVE_PROFILES.get("steering")
    if joint_name == "lift_joint":
        return DRIVE_PROFILES.get("lift")
    if joint_name in ARM_JOINTS:
        return DRIVE_PROFILES.get("arm")
    if joint_name in FINGER_JOINTS:
        return DRIVE_PROFILES.get("finger")
    if joint_name in HEAD_JOINTS:
        return DRIVE_PROFILES.get("head")
    return None


def apply_isaac_drive_profiles(root: ET.Element) -> int:
    changed = 0
    for joint in root.findall("joint"):
        joint_name = joint.get("name", "")
        profile = drive_profile_for_joint(joint_name)
        if profile is None:
            continue
        drive_api = joint.find("isaac_drive_api")
        if drive_api is None:
            drive_api = add_element(joint, "isaac_drive_api")
        for key, value in profile.items():
            drive_api.set(key, value)
        changed += 1
    return changed


def add_state_interfaces(joint: ET.Element, initial_value: str = "0.0") -> None:
    position = add_element(joint, "state_interface", name="position")
    add_text(position, "param", initial_value, name="initial_value")
    add_element(joint, "state_interface", name="velocity")
    add_element(joint, "state_interface", name="effort")


def add_topic_based_ros2_control(
    root: ET.Element,
    command_topic: str,
    state_topic: str,
) -> tuple[int, int]:
    existing_joints = mechanical_joint_names(root)
    system = add_element(root, "ros2_control", name="openflex", type="system")
    hardware = add_element(system, "hardware")
    add_text(hardware, "plugin", "topic_based_ros2_control/TopicBasedSystem")
    add_text(hardware, "param", command_topic, name="joint_commands_topic")
    add_text(hardware, "param", state_topic, name="joint_states_topic")
    add_text(hardware, "param", "true", name="sum_wrapped_joint_states")

    position_count = 0
    velocity_count = 0
    for joint_name in POSITION_JOINTS:
        if joint_name not in existing_joints:
            continue
        joint = add_element(system, "joint", name=joint_name)
        # Mimic fingers are state-only in ros2_control.  TopicBasedSystem
        # expands the command for the mimicked joint before publishing the
        # Isaac JointState command, so both physical fingers move together.
        if joint_name.endswith("finger_joint2"):
            add_text(joint, "param", joint_name.replace("finger_joint2", "finger_joint1"), name="mimic")
            add_text(joint, "param", "1.0", name="multiplier")
        else:
            add_element(joint, "command_interface", name="position")
        if joint_name in LIFT_VELOCITY_JOINTS:
            add_element(joint, "command_interface", name="velocity")
            velocity_count += 1
        if joint_name in EFFORT_JOINTS:
            add_element(joint, "command_interface", name="effort")
        add_state_interfaces(joint)
        position_count += 1

    for joint_name in WHEEL_JOINTS:
        if joint_name not in existing_joints:
            continue
        joint = add_element(system, "joint", name=joint_name)
        command = add_element(joint, "command_interface", name="velocity")
        add_text(command, "param", "-8.0", name="min")
        add_text(command, "param", "8.0", name="max")
        add_state_interfaces(joint)
        velocity_count += 1

    return position_count, velocity_count


def ensure_empty_link(root: ET.Element, link_name: str, parent_link: str, xyz: str = "0 0 0", rpy: str = "0 0 0") -> bool:
    links = {link.get("name", "") for link in root.findall("link")}
    if parent_link not in links:
        return False
    if link_name not in links:
        add_element(root, "link", name=link_name)
    joint_names = {joint.get("name", "") for joint in root.findall("joint")}
    joint_name = f"{link_name}_joint"
    if joint_name not in joint_names:
        joint = add_element(root, "joint", name=joint_name, type="fixed")
        add_element(joint, "parent", link=parent_link)
        add_element(joint, "child", link=link_name)
        add_element(joint, "origin", xyz=xyz, rpy=rpy)
    return True


def add_camera_sensor(
    isaac: ET.Element,
    link_name: str,
    sensor_type: str,
    topic: str,
    width: int,
    height: int,
    mount_xyz: str = "0 0 0",
    mount_rpy: str = "1.57079632679 0 -1.57079632679",
    render_group: str | None = None,
    camera_model: str = "d435i",
) -> None:
    sensor = add_element(isaac, "sensor", name=link_name, type=sensor_type)
    if render_group:
        add_text(sensor, "render_group", render_group)
    model = camera_model.strip().lower()
    add_text(sensor, "model", model)
    add_text(sensor, "topic", topic)
    add_element(sensor, "origin", xyz=mount_xyz, rpy=mount_rpy)
    image = add_element(sensor, "image")
    add_text(image, "width", str(width))
    add_text(image, "height", str(height))
    optics = CAMERA_OPTICS.get(model, CAMERA_OPTICS["d435i"])
    fov_h, fov_v = optics["depth" if sensor_type == "depth_camera" else "rgb"]
    add_text(sensor, "horizontal_fov_rad", f"{fov_h:.9f}")
    add_text(sensor, "vertical_fov_rad", f"{fov_v:.9f}")
    add_text(sensor, "horizontal_focal_length", "24.0")
    add_text(sensor, "vertical_focal_length", "24.0")
    add_text(sensor, "focus_distance", "8.0")
    add_text(sensor, "projection", "perspective")
    clip = add_element(sensor, "clip")
    add_text(clip, "near", "0.05")
    add_text(clip, "far", "8.0")


def add_lidar_sensor(isaac: ET.Element, link_name: str, topic: str, dimension: int, config: str) -> None:
    sensor = add_element(isaac, "sensor", name=link_name, type="lidar")
    add_text(sensor, "topic", topic)
    add_text(sensor, "sensor_dimension_num", str(dimension))
    add_text(sensor, "config", config)


def add_imu_sensor(isaac: ET.Element, link_name: str, topic: str) -> None:
    sensor = add_element(isaac, "sensor", name=link_name, type="imu")
    add_text(sensor, "topic", topic)


def override_mid360_mount(root: ET.Element, xyz: str, rpy: str) -> bool:
    """Override the fixed URDF mount while keeping the source xacro reusable."""
    joint = root.find("./joint[@name='mid360_joint']")
    if joint is None:
        return False
    origin = joint.find("origin")
    if origin is None:
        origin = add_element(joint, "origin")
    origin.set("xyz", " ".join(xyz.split()))
    origin.set("rpy", " ".join(rpy.split()))
    return True


def remove_head_camera_visual(root: ET.Element) -> int:
    removed = 0
    for link in root.findall("link"):
        for visual in list(link.findall("visual")):
            mesh = visual.find("./geometry/mesh")
            if mesh is None or "femto_bolt_camera" not in (mesh.get("filename") or ""):
                continue
            link.remove(visual)
            removed += 1
    return removed


def remove_link_visuals(root: ET.Element, link_name: str) -> int:
    link = root.find(f"./link[@name='{link_name}']")
    if link is None:
        return 0
    visuals = list(link.findall("visual"))
    for visual in visuals:
        link.remove(visual)
    return len(visuals)


def add_isaac_sensors(
    root: ET.Element,
    *,
    enabled: bool = True,
    image_width: int = 224,
    image_height: int = 224,
    sensor_profile: str = "data",
) -> int:
    profile = sensor_profile.strip().lower()
    if not enabled or profile in ("none", "teleop"):
        return 0
    enable_rgb = profile in ("rgb", "rgb_depth", "data")
    enable_depth = profile in ("rgb_depth", "data")
    enable_lidar = profile in ("lidar", "data")

    links = {link.get("name", "") for link in root.findall("link")}
    isaac = add_element(root, "isaac")
    added = 0

    # Keep one physical camera per mount.  RGB and depth entries deliberately
    # share the same render_group; launch_sensor.py then creates one Render
    # Product and two helpers instead of rendering the same view twice.
    camera_pairs = (
        (
            "base",
            "d435_color_optical_frame",
            "d435_depth_optical_frame",
            "color/image",
            "depth/image",
            CAMERA_MOUNTS["base"],
            "d435i",
        ),
        (
            "head",
            "head_yaw_link",
            "head_yaw_link",
            "head/color/image",
            "head/depth/image",
            CAMERA_MOUNTS["head"],
            "d435i",
        ),
        (
            "left_hand",
            "openarmx_left_hand",
            "openarmx_left_hand",
            "left/color/image",
            "left/depth/image",
            CAMERA_MOUNTS["left_hand"],
            "d405",
        ),
        (
            "right_hand",
            "openarmx_right_hand",
            "openarmx_right_hand",
            "right/color/image",
            "right/depth/image",
            CAMERA_MOUNTS["right_hand"],
            "d405",
        ),
    )
    for render_group, color_link, depth_link, color_topic, depth_topic, mount, camera_model in camera_pairs:
        if enable_rgb and color_link in links:
            add_camera_sensor(
                isaac,
                color_link,
                "camera",
                color_topic,
                image_width,
                image_height,
                *mount,
                render_group=render_group,
                camera_model=camera_model,
            )
            added += 1
        if enable_depth and depth_link in links:
            add_camera_sensor(
                isaac,
                depth_link,
                "depth_camera",
                depth_topic,
                image_width,
                image_height,
                *mount,
                render_group=render_group,
                camera_model=camera_model,
            )
            added += 1
    if enable_lidar and "livox_frame" in links:
        add_lidar_sensor(isaac, "livox_frame", "lidar", 3, LIVOX_APPROX_LIDAR_CONFIG)
        add_imu_sensor(isaac, "livox_frame", "/livox/imu")
        added += 2

    if added == 0:
        root.remove(isaac)
    return added


def build(
    source: Path,
    robot_name: str,
    command_topic: str,
    state_topic: str,
    keep_package_uris: bool,
    enable_head: str,
    mesh_alias_dir: Path,
    preserve_visual_materials: bool = False,
    enable_sensors: bool = True,
    camera_width: int = 224,
    camera_height: int = 224,
    sensor_profile: str = "data",
    mid360_xyz: str = "0.30 0.0 0.12",
    mid360_rpy: str = "-0.5236 0 -1.5708",
    hide_head_camera: bool = False,
    hide_lift_mast: bool = False,
) -> tuple[ET.ElementTree, dict[str, int]]:
    document = xacro.process_file(
        str(source),
        mappings={
            "use_fake_hardware": "true",
            "use_mock": "true",
            "enable_head": enable_head,
            "enable_sensors": "true" if enable_sensors else "false",
            "mid360_xyz": os.environ.get("OPENFLEX_MID360_XYZ", "0.30 0.0 0.12"),
            "mid360_rpy": os.environ.get("OPENFLEX_MID360_RPY", "-0.5236 0 -1.5708"),
        },
    )
    root = ET.fromstring(document.toxml())
    root.set("name", robot_name)
    hidden_head_camera_visuals = remove_head_camera_visual(root) if hide_head_camera else 0
    hidden_lift_mast_visuals = remove_link_visuals(root, "lift_base_link") if hide_lift_mast else 0
    override_mid360_mount(root, mid360_xyz, mid360_rpy)
    tree = ET.ElementTree(root)

    removed_ros2_control, removed_mujoco, removed_gazebo = remove_existing_simulation_blocks(root)
    if keep_package_uris:
        resolved_meshes = 0
        mesh_aliases = 0
    else:
        resolved_meshes, mesh_aliases = resolve_package_mesh_uris(root, mesh_alias_dir)
    position_joints, velocity_joints = add_topic_based_ros2_control(root, command_topic, state_topic)
    isaac_drive_apis = apply_isaac_drive_profiles(root)
    isaac_sensors = add_isaac_sensors(
        root,
        enabled=enable_sensors,
        image_width=camera_width,
        image_height=camera_height,
        sensor_profile=sensor_profile,
    )
    wheel_collisions = replace_wheel_collisions_with_cylinders(root)
    lift_collision_proxies = replace_high_poly_lift_collisions_with_boxes(root)
    added_inertials, overridden_inertials = add_default_inertials(root)
    fixed_frame_inertials_removed = remove_massless_fixed_frame_inertials(root)
    appearance_visuals = 0 if preserve_visual_materials else apply_real_robot_appearance(root)
    physics_materials = apply_physics_materials(root)

    ET.indent(tree, space="  ")
    return tree, {
        "removed_ros2_control": removed_ros2_control,
        "removed_mujoco": removed_mujoco,
        "removed_gazebo": removed_gazebo,
        "resolved_meshes": resolved_meshes,
        "mesh_aliases": mesh_aliases,
        "added_inertials": added_inertials,
        "overridden_inertials": overridden_inertials,
        "fixed_frame_inertials_removed": fixed_frame_inertials_removed,
        "isaac_drive_apis": isaac_drive_apis,
        "wheel_collisions": wheel_collisions,
        "lift_collision_proxies": lift_collision_proxies,
        "physics_materials": physics_materials,
        "position_joints": position_joints,
        "velocity_joints": velocity_joints,
        "isaac_sensors": isaac_sensors,
        "appearance_visuals": appearance_visuals,
        "hidden_head_camera_visuals": hidden_head_camera_visuals,
        "hidden_lift_mast_visuals": hidden_lift_mast_visuals,
    }


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if not args.source.is_file():
        print(f"Authoritative xacro does not exist: {args.source}", file=sys.stderr)
        return 1

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tree, summary = build(
            source=args.source,
            robot_name=args.robot_name,
            command_topic=args.command_topic,
            state_topic=args.state_topic,
            keep_package_uris=args.keep_package_uris,
            enable_head=args.enable_head,
            mesh_alias_dir=args.output.parent / f"{args.output.stem}_meshes",
            preserve_visual_materials=args.preserve_visual_materials,
            enable_sensors=not args.disable_sensors,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            sensor_profile="none" if args.disable_sensors else args.sensor_profile,
            mid360_xyz=args.mid360_xyz,
            mid360_rpy=args.mid360_rpy,
            hide_head_camera=args.hide_head_camera,
            hide_lift_mast=args.hide_lift_mast,
        )
    except Exception as error:
        print(f"Isaac URDF generation failed: {error}", file=sys.stderr)
        return 1

    tree.write(args.output, encoding="unicode", xml_declaration=False)
    root = tree.getroot()
    print(
        "Generated OpenFleX Isaac URDF: "
        f"links={len(root.findall('link'))} joints={len(root.findall('joint'))} "
        f"position_joints={summary['position_joints']} velocity_joints={summary['velocity_joints']} "
        f"isaac_drive_apis={summary['isaac_drive_apis']} "
        f"wheel_collisions={summary['wheel_collisions']} "
        f"lift_collision_proxies={summary['lift_collision_proxies']} "
        f"physics_materials={summary['physics_materials']} "
        f"isaac_sensors={summary['isaac_sensors']} resolved_meshes={summary['resolved_meshes']} "
        f"mesh_aliases={summary['mesh_aliases']} "
        f"default_inertials={summary['added_inertials']} "
        f"overridden_inertials={summary['overridden_inertials']} "
        f"fixed_frame_inertials_removed={summary['fixed_frame_inertials_removed']} "
        f"appearance_visuals={summary['appearance_visuals']} "
        f"hidden_head_camera_visuals={summary['hidden_head_camera_visuals']} "
        f"hidden_lift_mast_visuals={summary['hidden_lift_mast_visuals']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
