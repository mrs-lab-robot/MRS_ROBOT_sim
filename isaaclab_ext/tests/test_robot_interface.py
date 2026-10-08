from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def test_contract_interface_uses_checkout_root_when_environment_override_is_unset():
    module = importlib.import_module("mrs_robot_lab.assets.robot_interface")
    with patch.dict(os.environ, {}, clear=True):
        module = importlib.reload(module)
    assert module.ROBOT_RESOLVER.repository_root == ROOT
    assert module.ROBOT_CONTRACT.robot_id == "openflex"


def test_action_slices_and_joint_limits_are_derived_from_the_robot_contract():
    module = importlib.import_module("mrs_robot_lab.assets.robot_interface")
    assert module.ACTION_DIMENSION == 22
    assert len(module.JOINT_STATE_NAMES) == 19
    assert module.ACTION_SLICES["base_twist_action"] == (0, 3)
    assert module.ACTION_SLICES["left_arm_action"] == (3, 7)
    assert module.ACTION_SLICES["right_arm_action"] == (10, 7)
    assert module.ACTION_SLICES["lift_action"] == (17, 1)
    assert module.ACTION_SLICES["head_action"] == (18, 2)
    assert module.ACTION_SLICES["left_gripper_action"] == (20, 1)
    assert module.ACTION_SLICES["right_gripper_action"] == (21, 1)
    assert module.JOINT_POSITION_LIMITS["lift_joint"] == (-0.65, 0.3)
    assert module.JOINT_POSITION_LIMITS["fl_steering_joint"] == (-1.5708, 1.5708)
