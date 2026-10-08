#!/usr/bin/python3
"""Wait until Isaac has produced simulation time and joint-state feedback."""

from __future__ import annotations

import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


def _time_ns(stamp: object) -> int:
    return int(getattr(stamp, "sec", 0)) * 1_000_000_000 + int(getattr(stamp, "nanosec", 0))


def _header_stamp_ns(message: object) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0
    return _time_ns(stamp)


@dataclass
class SimulationReadyState:
    clock_topic: str = "/clock"
    joint_states_topic: str = "/openflex/joint_states"
    require_nonzero_clock: bool = True
    require_nonzero_joint_stamp: bool = True
    clock_seen: bool = False
    joint_state_seen: bool = False
    last_clock_ns: int = 0
    last_joint_state_stamp_ns: int = 0

    def mark_clock(self, message: Clock) -> None:
        self.clock_seen = True
        self.last_clock_ns = _time_ns(message.clock)

    def mark_joint_state(self, message: JointState) -> None:
        self.joint_state_seen = True
        self.last_joint_state_stamp_ns = _header_stamp_ns(message)

    @property
    def ready(self) -> bool:
        return not self.issues()

    def issues(self) -> list[str]:
        issues: list[str] = []
        if not self.clock_seen:
            issues.append(f"{self.clock_topic}: no message received")
        elif self.require_nonzero_clock and self.last_clock_ns <= 0:
            issues.append(f"{self.clock_topic}: latest stamp is zero")

        if not self.joint_state_seen:
            issues.append(f"{self.joint_states_topic}: no message received")
        elif self.require_nonzero_joint_stamp and self.last_joint_state_stamp_ns <= 0:
            issues.append(f"{self.joint_states_topic}: latest header stamp is zero")
        return issues


class SimulationReadyWaiter(Node):
    def __init__(self) -> None:
        super().__init__("wait_for_sim_ready")
        self.declare_parameter("clock_topic", "/clock")
        self.declare_parameter("joint_states_topic", "/openflex/joint_states")
        self.declare_parameter("timeout_sec", 180.0)
        self.declare_parameter("log_period_sec", 5.0)
        self.declare_parameter("require_nonzero_clock", True)
        self.declare_parameter("require_nonzero_joint_stamp", True)

        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.log_period_sec = float(self.get_parameter("log_period_sec").value)
        self.state = SimulationReadyState(
            clock_topic=str(self.get_parameter("clock_topic").value),
            joint_states_topic=str(self.get_parameter("joint_states_topic").value),
            require_nonzero_clock=bool(self.get_parameter("require_nonzero_clock").value),
            require_nonzero_joint_stamp=bool(
                self.get_parameter("require_nonzero_joint_stamp").value
            ),
        )
        self.create_subscription(
            Clock,
            self.state.clock_topic,
            self.state.mark_clock,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            self.state.joint_states_topic,
            self.state.mark_joint_state,
            qos_profile_sensor_data,
        )

    def wait_until_ready(self) -> bool:
        deadline = time.monotonic() + self.timeout_sec
        last_log_time = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.ready:
                self.get_logger().info(
                    "Isaac simulation is ready: "
                    f"{self.state.clock_topic}={self.state.last_clock_ns}ns, "
                    f"{self.state.joint_states_topic}="
                    f"{self.state.last_joint_state_stamp_ns}ns"
                )
                return True

            now = time.monotonic()
            if last_log_time == 0.0 or now - last_log_time >= self.log_period_sec:
                self.get_logger().info(
                    "Waiting for Isaac simulation readiness: "
                    + "; ".join(self.state.issues())
                )
                last_log_time = now
        self.get_logger().error(
            f"Timed out after {self.timeout_sec:.1f}s waiting for Isaac simulation readiness: "
            + "; ".join(self.state.issues())
        )
        return False


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: SimulationReadyWaiter | None = None
    exit_code = 1
    try:
        node = SimulationReadyWaiter()
        if node.wait_until_ready():
            exit_code = 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
