from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = (
    REPO_ROOT
    / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_description/scripts/generate_isaac_urdf.py"
)
_SPEC = importlib.util.spec_from_file_location("isaac_urdf_generator", GENERATOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GENERATOR)


class IsaacArmCommandModeTest(unittest.TestCase):
    def test_arm_drives_match_the_known_good_isaac_profile(self) -> None:
        root = ET.Element("robot", name="test_robot")
        for joint_name in GENERATOR.ARM_JOINTS:
            ET.SubElement(root, "joint", name=joint_name, type="revolute")

        GENERATOR.apply_isaac_drive_profiles(root)

        drive_profiles = {
            joint.get("name"): joint.find("isaac_drive_api").attrib
            for joint in root.findall("joint")
        }

        self.assertEqual(set(drive_profiles), set(GENERATOR.ARM_JOINTS))
        for joint_name, profile in drive_profiles.items():
            with self.subTest(joint=joint_name):
                self.assertEqual(
                    profile,
                    {
                        "stiffness": "100000000.0",
                        "damping": "300000.0",
                        "joint_friction": "5.0",
                    },
                )

    def test_position_controlled_arm_joints_keep_legacy_effort_metadata(self) -> None:
        root = ET.Element("robot", name="test_robot")
        for joint_name in (*GENERATOR.POSITION_JOINTS, *GENERATOR.WHEEL_JOINTS):
            ET.SubElement(root, "joint", name=joint_name, type="revolute")

        GENERATOR.add_topic_based_ros2_control(
            root,
            command_topic="/openflex/joint_command",
            state_topic="/openflex/joint_states",
        )

        arm_joint_modes = {
            joint.get("name"): [
                interface.get("name")
                for interface in joint.findall("./command_interface")
            ]
            for joint in root.findall("./ros2_control/joint")
            if joint.get("name") in GENERATOR.ARM_JOINTS
        }

        self.assertEqual(set(arm_joint_modes), set(GENERATOR.ARM_JOINTS))
        for joint_name, modes in arm_joint_modes.items():
            with self.subTest(joint=joint_name):
                # The runtime command graph remains position-driven.  The
                # effort entry is compatibility metadata required by the
                # bundled legacy controller's flattened interface lookup.
                self.assertEqual(modes, ["position", "effort"])


if __name__ == "__main__":
    unittest.main()
