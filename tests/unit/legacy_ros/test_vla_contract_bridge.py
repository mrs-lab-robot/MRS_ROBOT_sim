"""Unit tests for the Isaac Sim LeRobot ROS contract bridge."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/scripts/vla_contract_bridge.py"
)
SPEC = importlib.util.spec_from_file_location("vla_contract_bridge_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VLA_CONTRACT_BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VLA_CONTRACT_BRIDGE)


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class LiftVlaContractTest(unittest.TestCase):
    @staticmethod
    def _bridge():
        bridge = SimpleNamespace(
            _lift_position_publisher=RecordingPublisher(),
            _lift_action_publisher=RecordingPublisher(),
            _lift_target_position=None,
            _latest_lift_position=0.12,
            _lift_velocity_mps=0.0,
            _last_velocity_command_time=None,
            _velocity_command_active=False,
            get_parameter=lambda name: SimpleNamespace(
                value={
                    "lift_min_position_m": -0.650,
                    "lift_max_position_m": 0.300,
                    "lift_max_velocity_mps": 0.100,
                }[name]
            ),
        )
        for method_name in (
            "_publish_lift_action",
            "_publish_lift_target",
            "_set_lift_velocity",
            "_stop_lift_velocity",
        ):
            setattr(
                bridge,
                method_name,
                lambda *args, name=method_name: getattr(
                    VLA_CONTRACT_BRIDGE.VlaContractBridge, name
                )(bridge, *args),
            )
        return bridge

    def test_lift_target_is_published_as_a_lerobot_action_as_well_as_a_sim_command(self):
        bridge = self._bridge()

        VLA_CONTRACT_BRIDGE.VlaContractBridge._publish_lift_target(bridge, 0.8)

        self.assertEqual(list(bridge._lift_position_publisher.messages[0].data), [0.300])
        self.assertEqual(bridge._lift_action_publisher.messages[0].data, 0.300)
        self.assertEqual(bridge._lift_target_position, 0.300)

    def test_idle_lift_joint_state_is_published_as_a_live_leader_action(self):
        bridge = self._bridge()
        joint_state = JointState()
        joint_state.name = ["lift_joint"]
        joint_state.position = [0.23]

        VLA_CONTRACT_BRIDGE.VlaContractBridge._on_joint_states(bridge, joint_state)

        self.assertEqual(bridge._lift_target_position, 0.23)
        self.assertEqual(bridge._lift_action_publisher.messages[0].data, 0.23)

    def test_vr_jog_input_drives_the_sim_lift_target(self):
        bridge = self._bridge()
        command = Float64()
        command.data = 0.05

        VLA_CONTRACT_BRIDGE.VlaContractBridge._on_lift_jog_command(bridge, command)

        self.assertTrue(bridge._velocity_command_active)
        self.assertEqual(bridge._lift_velocity_mps, 0.05)
        self.assertEqual(bridge._lift_target_position, 0.12)

    def test_vr_step_input_moves_and_records_the_clamped_lift_target(self):
        bridge = self._bridge()
        command = Float64MultiArray()
        command.data = [0.50]

        VLA_CONTRACT_BRIDGE.VlaContractBridge._on_lift_step_command(bridge, command)

        self.assertEqual(bridge._lift_target_position, 0.30)
        self.assertEqual(list(bridge._lift_position_publisher.messages[0].data), [0.30])
        self.assertEqual(bridge._lift_action_publisher.messages[0].data, 0.30)


if __name__ == "__main__":
    unittest.main()
