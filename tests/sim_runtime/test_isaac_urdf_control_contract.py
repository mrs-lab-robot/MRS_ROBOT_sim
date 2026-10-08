from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = (
    ROOT
    / "sim_runtime"
    / "ros2"
    / "openflex_isaac_sim"
    / "openflex_isaac_description"
    / "scripts"
    / "generate_isaac_urdf.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location("openflex_isaac_urdf_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _robot_with_controlled_joints(generator) -> ET.Element:
    root = ET.Element("robot", {"name": "openflex"})
    joint_names = (
        *generator.POSITION_JOINTS,
        *generator.WHEEL_JOINTS,
    )
    for name in joint_names:
        ET.SubElement(root, "joint", {"name": name, "type": "revolute"})
    return root


def test_legacy_isaac_controller_contract_has_command_interface_for_each_joint():
    generator = _load_generator()
    root = _robot_with_controlled_joints(generator)

    generator.add_topic_based_ros2_control(
        root,
        "/openflex/joint_command",
        "/openflex/joint_states",
    )

    controlled = root.find("ros2_control")
    assert controlled is not None
    joints = controlled.findall("joint")
    assert len(joints) == len(generator.POSITION_JOINTS) + len(generator.WHEEL_JOINTS)
    command_count = sum(len(joint.findall("command_interface")) for joint in joints)
    assert command_count >= len(joints)
    for mimic in ("openarmx_left_finger_joint2", "openarmx_right_finger_joint2"):
        assert not root.find(f"ros2_control/joint[@name='{mimic}']").findall(
            "command_interface"
        )

    for side in ("left", "right"):
        for index in range(1, 8):
            interfaces = root.find(
                f"ros2_control/joint[@name='openarmx_{side}_joint{index}']"
            ).findall("command_interface")
            assert [item.get("name") for item in interfaces] == ["position", "effort"]
