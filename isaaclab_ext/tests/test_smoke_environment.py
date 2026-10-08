from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mrs_robot_lab.environments.smoke.openflex_smoke import OpenFlexSmokeEnvironment


class OpenFlexSmokeEnvironmentTest(unittest.TestCase):
    def test_validated_joint_target_reaches_the_articulation(self):
        class FakeTorch:
            float32 = "torch.float32"

            @staticmethod
            def tensor(values, *, dtype, device):
                return {"values": values, "dtype": dtype, "device": device}

        class FakeArticulation:
            data = SimpleNamespace(joint_pos=SimpleNamespace(dtype="warp.float32"))
            device = "cuda:0"

            def find_joints(self, names, *, preserve_order):
                self.preserve_order = preserve_order
                return [7], names

            def set_joint_position_target(self, target, *, joint_ids):
                self.last_target = target
                self.last_joint_ids = joint_ids

        environment = OpenFlexSmokeEnvironment.__new__(OpenFlexSmokeEnvironment)
        environment._position_limits = {"arm_joint": (-1.0, 1.0)}
        environment._torch = FakeTorch()
        environment.robot = FakeArticulation()

        environment.set_joint_positions({"arm_joint": 0.25})

        self.assertEqual(
            environment.robot.last_target,
            {"values": [[0.25]], "dtype": FakeTorch.float32, "device": "cuda:0"},
        )
        self.assertEqual(environment.robot.last_joint_ids, [7])
        self.assertTrue(environment.robot.preserve_order)

    def test_articulation_joints_are_queried_after_simulation_reset(self):
        state = {"simulation_initialized": False, "joint_lookup_after_reset": False}
        joint_names = ("base_joint", "arm_joint")

        class FakeSimulationContext:
            def __init__(self, _cfg):
                pass

            def reset(self):
                state["simulation_initialized"] = True

        class FakeGroundPlaneCfg:
            func = staticmethod(lambda _prim_path, _cfg: None)

        class FakeRobotCfg:
            def copy(self):
                return SimpleNamespace()

        class FakeArticulation:
            def __init__(self, _cfg):
                pass

            def find_joints(self, names, *, preserve_order):
                if not state["simulation_initialized"]:
                    raise RuntimeError("find_joints called before SimulationContext.reset")
                state["joint_lookup_after_reset"] = True
                return list(range(len(names))), names

            def reset(self):
                pass

            def update(self, _dt):
                pass

        isaaclab = types.ModuleType("isaaclab")
        isaaclab.__path__ = []
        sim_module = types.ModuleType("isaaclab.sim")
        sim_module.SimulationCfg = lambda **kwargs: kwargs
        sim_module.GroundPlaneCfg = FakeGroundPlaneCfg
        sim_module.SimulationContext = FakeSimulationContext
        assets_module = types.ModuleType("isaaclab.assets")
        assets_module.Articulation = FakeArticulation
        isaaclab.sim = sim_module
        isaaclab.assets = assets_module

        lab_module = types.ModuleType("mrs_robot_lab")
        lab_module.__path__ = []
        lab_assets = types.ModuleType("mrs_robot_lab.assets")
        lab_assets.__path__ = []
        lab_runners = types.ModuleType("mrs_robot_lab.runners")
        lab_runners.__path__ = []
        robot_cfg_module = types.ModuleType("mrs_robot_lab.assets.mrs_robot_cfg")
        robot_cfg_module.JOINT_STATE_NAMES = joint_names
        robot_cfg_module.MRS_ROBOT_CFG = FakeRobotCfg()
        robot_interface_module = types.ModuleType("mrs_robot_lab.assets.robot_interface")
        robot_interface_module.JOINT_POSITION_LIMITS = {}
        robot_interface_module.SWERVE_CONFIG = SimpleNamespace(
            wheel_joint_names=(), max_wheel_speed=1.0, wheel_radius=0.1
        )
        joint_teleop_module = types.ModuleType("mrs_robot_lab.runners.joint_teleop")
        joint_teleop_module.validate_joint_targets = lambda targets, _limits: targets

        fake_modules = {
            "torch": types.ModuleType("torch"),
            "isaaclab": isaaclab,
            "isaaclab.sim": sim_module,
            "isaaclab.assets": assets_module,
            "mrs_robot_lab": lab_module,
            "mrs_robot_lab.assets": lab_assets,
            "mrs_robot_lab.runners": lab_runners,
            "mrs_robot_lab.assets.mrs_robot_cfg": robot_cfg_module,
            "mrs_robot_lab.assets.robot_interface": robot_interface_module,
            "mrs_robot_lab.runners.joint_teleop": joint_teleop_module,
        }

        with patch.dict(sys.modules, fake_modules):
            environment = OpenFlexSmokeEnvironment(enable_cameras=False)

        self.assertTrue(state["joint_lookup_after_reset"])
        self.assertEqual(environment.joint_names, joint_names)


if __name__ == "__main__":
    unittest.main()
