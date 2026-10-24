from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
GENERATOR = (
    ROOT
    / "sim_runtime"
    / "ros2"
    / "openflex_isaac_sim"
    / "openflex_isaac_description"
    / "scripts"
    / "generate_isaac_urdf.py"
)


def _module(name: str, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load_generator():
    """加载 URDF 生成器。

    生成器是 ROS 侧脚本（顶层 import ament_index_python / xacro），而本仓
    测试跑在 arena venv 里、没有 ROS 环境。照 tests/unit/sim_runtime/test_isaacsim_path.py
    的做法注入假模块，否则连 import 都过不了、断言一条也跑不到。
    """
    spec = importlib.util.spec_from_file_location("openflex_isaac_urdf_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    class _PackageNotFoundError(Exception):
        pass

    stubs = {
        "ament_index_python": _module("ament_index_python", __path__=[]),
        "ament_index_python.packages": _module(
            "ament_index_python.packages",
            PackageNotFoundError=_PackageNotFoundError,
            get_package_share_directory=lambda *a, **k: "/tmp",
            get_package_prefix=lambda *a, **k: "/tmp",
        ),
        "xacro": _module("xacro", process_file=lambda *a, **k: ""),
    }
    with patch.dict(sys.modules, stubs):
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
