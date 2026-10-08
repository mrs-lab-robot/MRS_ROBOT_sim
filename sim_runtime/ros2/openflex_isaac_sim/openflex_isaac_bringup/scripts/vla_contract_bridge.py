#!/usr/bin/env python3
"""Expose the real-robot VLA odometry and lift-action contract in Isaac Sim."""

from __future__ import annotations

import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray


class VlaContractBridge(Node):
    def __init__(self) -> None:
        super().__init__("openflex_vla_contract_bridge")
        self.declare_parameter("odom_input_topic", "/odom")
        self.declare_parameter("odom_output_topic", "/fastlio2/lio_odom")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("lift_velocity_topic", "/velocity_controller/commands")
        self.declare_parameter("lift_position_topic", "/lift_position_controller/commands")
        self.declare_parameter(
            "lift_jog_topic", "/lift_manual_position_controller/jog_command"
        )
        self.declare_parameter(
            "lift_step_topic", "/lift_manual_position_controller/step_command"
        )
        self.declare_parameter(
            "lift_action_topic", "/lift_manual_position_controller/action_position"
        )
        self.declare_parameter("lift_min_position_m", -0.650)
        self.declare_parameter("lift_max_position_m", 0.300)
        self.declare_parameter("lift_max_velocity_mps", 0.100)
        self.declare_parameter("lift_command_timeout_s", 0.250)
        self.declare_parameter("lift_update_rate_hz", 50.0)

        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._latest_lift_position: float | None = None
        self._lift_target_position: float | None = None
        self._lift_velocity_mps = 0.0
        self._last_velocity_command_time: float | None = None
        self._last_update_time = time.monotonic()
        self._velocity_command_active = False

        self._odom_publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("odom_output_topic").value),
            10,
        )
        self._lift_position_publisher = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("lift_position_topic").value),
            10,
        )
        self._lift_action_publisher = self.create_publisher(
            Float64,
            str(self.get_parameter("lift_action_topic").value),
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_input_topic").value),
            self._odom_publisher.publish,
            10,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_states_topic").value),
            self._on_joint_states,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("lift_velocity_topic").value),
            self._on_lift_velocity,
            10,
        )
        self.create_subscription(
            Float64,
            str(self.get_parameter("lift_jog_topic").value),
            self._on_lift_jog_command,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            str(self.get_parameter("lift_step_topic").value),
            self._on_lift_step_command,
            10,
        )

        update_rate_hz = max(1.0, float(self.get_parameter("lift_update_rate_hz").value))
        self.create_timer(
            1.0 / update_rate_hz,
            self._update_lift_target,
            clock=self._steady_clock,
        )
        self.get_logger().info(
            "VLA contract bridge started: /odom -> /fastlio2/lio_odom; "
            "VR jog/step and /velocity_controller/commands -> lift targets; "
            "lift targets are exposed on /lift_manual_position_controller/action_position"
        )
        self.get_logger().warn(
            "/fastlio2/lio_odom is simulated wheel odometry compatibility output, "
            "not a FAST-LIO estimator result"
        )

    def _on_joint_states(self, message: JointState) -> None:
        try:
            index = message.name.index("lift_joint")
        except ValueError:
            return
        if index >= len(message.position):
            return
        self._latest_lift_position = float(message.position[index])
        if not self._velocity_command_active:
            self._lift_target_position = self._latest_lift_position
            self._publish_lift_action(self._latest_lift_position)

    def _on_lift_velocity(self, message: Float64MultiArray) -> None:
        if not message.data:
            self.get_logger().warn("Ignoring empty /velocity_controller/commands message")
            return
        self._set_lift_velocity(float(message.data[0]))

    def _on_lift_jog_command(self, message: Float64) -> None:
        self._set_lift_velocity(float(message.data))

    def _set_lift_velocity(self, requested_velocity: float) -> None:
        max_velocity = abs(float(self.get_parameter("lift_max_velocity_mps").value))
        self._lift_velocity_mps = max(-max_velocity, min(max_velocity, requested_velocity))
        self._last_velocity_command_time = time.monotonic()
        if abs(self._lift_velocity_mps) > 1.0e-9:
            if not self._velocity_command_active:
                self._lift_target_position = self._latest_lift_position
            self._velocity_command_active = True
        else:
            self._stop_lift_velocity()

    def _on_lift_step_command(self, message: Float64MultiArray) -> None:
        if not message.data:
            self.get_logger().warn("Ignoring empty /lift_manual_position_controller/step_command")
            return
        self._stop_lift_velocity()
        current_target = (
            self._lift_target_position
            if self._lift_target_position is not None
            else self._latest_lift_position
        )
        if current_target is None:
            self.get_logger().warn("Ignoring lift step because no lift joint state is available")
            return
        self._publish_lift_target(float(current_target) + float(message.data[0]))

    def _stop_lift_velocity(self) -> None:
        if self._velocity_command_active and self._lift_target_position is not None:
            self._publish_lift_target(self._lift_target_position)
        self._lift_velocity_mps = 0.0
        self._velocity_command_active = False

    def _update_lift_target(self) -> None:
        now = time.monotonic()
        elapsed = max(0.0, min(0.1, now - self._last_update_time))
        self._last_update_time = now
        if not self._velocity_command_active:
            return
        timeout = max(0.0, float(self.get_parameter("lift_command_timeout_s").value))
        if (
            self._last_velocity_command_time is None
            or (timeout > 0.0 and now - self._last_velocity_command_time > timeout)
        ):
            self._stop_lift_velocity()
            return
        if self._lift_target_position is None:
            self._lift_target_position = self._latest_lift_position
        if self._lift_target_position is None:
            return
        self._lift_target_position += self._lift_velocity_mps * elapsed
        self._publish_lift_target(self._lift_target_position)

    def _publish_lift_target(self, target: float) -> None:
        minimum = float(self.get_parameter("lift_min_position_m").value)
        maximum = float(self.get_parameter("lift_max_position_m").value)
        self._lift_target_position = max(minimum, min(maximum, float(target)))
        message = Float64MultiArray()
        message.data = [self._lift_target_position]
        self._lift_position_publisher.publish(message)
        self._publish_lift_action(self._lift_target_position)

    def _publish_lift_action(self, target: float) -> None:
        message = Float64()
        message.data = float(target)
        self._lift_action_publisher.publish(message)


def main() -> int:
    rclpy.init()
    node = VlaContractBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
