from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from mrs_robot_arena.teleoperation.action_mapper import JOINT_STATE_NAMES
from scripts.teleop_vr_openflex import _joint_state


class TeleopScriptHelpersTest(unittest.TestCase):
    def test_joint_state_reads_tensor_without_main_local_torch_binding(self) -> None:
        names = (*JOINT_STATE_NAMES, "unmapped_aux_joint")
        positions = torch.arange(len(names), dtype=torch.float32).unsqueeze(0)
        robot = SimpleNamespace(
            joint_names=names,
            data=SimpleNamespace(joint_pos=positions),
        )

        ordered, by_name = _joint_state(robot, "joint_pos")

        self.assertEqual(ordered, list(range(len(JOINT_STATE_NAMES))))
        self.assertEqual(by_name["unmapped_aux_joint"], float(len(names) - 1))


if __name__ == "__main__":
    unittest.main()
