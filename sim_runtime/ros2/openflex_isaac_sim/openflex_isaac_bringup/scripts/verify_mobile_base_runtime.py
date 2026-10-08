#!/usr/bin/python3
"""Runtime smoke test for the OpenFleX Isaac Sim 4W4S mobile base."""

from __future__ import annotations

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


WHEEL_JOINTS = (
    "fl_wheel_joint",
    "fr_wheel_joint",
    "bl_wheel_joint",
    "br_wheel_joint",
)


class MobileBaseRuntimeVerifier(Node):
    def __init__(self, require_ground_truth: bool, use_clock_duration: bool) -> None:
        super().__init__("verify_isaac_mobile_base_runtime")
        self.require_ground_truth = require_ground_truth
        self.use_clock_duration = use_clock_duration
        self.joint_state: JointState | None = None
        self.odom: Odometry | None = None
        self.ground_truth: Odometry | None = None
        self.initial_odom: Odometry | None = None
        self.initial_ground_truth: Odometry | None = None
        self.clock_time_sec: float | None = None

        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self.create_subscription(Odometry, "/simulator/ground_truth", self._on_ground_truth, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _on_clock(self, message: Clock) -> None:
        self.clock_time_sec = message.clock.sec + message.clock.nanosec / 1_000_000_000.0

    def _on_joint_state(self, message: JointState) -> None:
        self.joint_state = message

    def _on_odom(self, message: Odometry) -> None:
        self.odom = message
        if self.initial_odom is None:
            self.initial_odom = message

    def _on_ground_truth(self, message: Odometry) -> None:
        self.ground_truth = message
        if self.initial_ground_truth is None:
            self.initial_ground_truth = message

    def required_topics_ready(self) -> bool:
        ready = self.joint_state is not None and self.odom is not None
        if self.use_clock_duration:
            ready = ready and self.clock_time_sec is not None
        if self.require_ground_truth:
            ready = ready and self.ground_truth is not None
        return ready

    def publish_command(self, linear_x: float) -> None:
        command = Twist()
        command.linear.x = linear_x
        self.cmd_vel_pub.publish(command)

    def wheel_speed(self) -> float:
        if self.joint_state is None:
            return 0.0
        velocities = dict(zip(self.joint_state.name, self.joint_state.velocity))
        return max(abs(velocities.get(name, 0.0)) for name in WHEEL_JOINTS)

    def displacement(self, initial: Odometry | None, current: Odometry | None) -> float:
        if initial is None or current is None:
            return 0.0
        start = initial.pose.pose.position
        now = current.pose.pose.position
        return math.hypot(now.x - start.x, now.y - start.y)

    def command_elapsed_seconds(
        self,
        start_wall_sec: float,
        start_clock_sec: float | None,
        now_wall_sec: float | None = None,
    ) -> float:
        if (
            self.use_clock_duration
            and start_clock_sec is not None
            and self.clock_time_sec is not None
            and self.clock_time_sec >= start_clock_sec
        ):
            return self.clock_time_sec - start_clock_sec
        return (time.monotonic() if now_wall_sec is None else now_wall_sec) - start_wall_sec


def default_command_wall_timeout(command_seconds: float) -> float:
    return max(30.0, command_seconds * 30.0)


def spin_until(node: Node, predicate, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.20)
    parser.add_argument("--command-seconds", type=float, default=4.0)
    parser.add_argument("--command-wall-timeout", type=float, default=0.0)
    parser.add_argument("--required-wheel-speed", type=float, default=0.25)
    parser.add_argument("--required-odom-displacement", type=float, default=0.03)
    parser.add_argument("--required-ground-truth-displacement", type=float, default=0.03)
    parser.add_argument("--require-ground-truth", action="store_true")
    parser.add_argument("--use-clock-duration", dest="use_clock_duration", action="store_true", default=True)
    parser.add_argument("--no-use-clock-duration", dest="use_clock_duration", action="store_false")
    args = parser.parse_args()
    command_wall_timeout = args.command_wall_timeout or default_command_wall_timeout(args.command_seconds)

    rclpy.init()
    node = MobileBaseRuntimeVerifier(
        require_ground_truth=args.require_ground_truth,
        use_clock_duration=args.use_clock_duration,
    )
    try:
        if not spin_until(node, node.required_topics_ready, 20.0):
            required = "/joint_states and /odom"
            if args.use_clock_duration:
                required += " and /clock"
            if args.require_ground_truth:
                required += " and /simulator/ground_truth"
            raise RuntimeError(f"Timed out waiting for {required}")

        start_wall_sec = time.monotonic()
        start_clock_sec = node.clock_time_sec
        wall_deadline = start_wall_sec + command_wall_timeout
        max_wheel_speed = 0.0
        while rclpy.ok() and node.command_elapsed_seconds(start_wall_sec, start_clock_sec) < args.command_seconds:
            if time.monotonic() >= wall_deadline:
                elapsed = node.command_elapsed_seconds(start_wall_sec, start_clock_sec)
                time_source = "/clock" if args.use_clock_duration and start_clock_sec is not None else "wall"
                raise RuntimeError(
                    f"Timed out after {command_wall_timeout:.1f}s wall time while commanding for "
                    f"{args.command_seconds:.3f}s of {time_source} time; reached {elapsed:.3f}s"
                )
            node.publish_command(args.speed)
            rclpy.spin_once(node, timeout_sec=0.1)
            max_wheel_speed = max(max_wheel_speed, node.wheel_speed())

        node.publish_command(0.0)
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.05)

        odom_displacement = node.displacement(node.initial_odom, node.odom)
        ground_truth_displacement = node.displacement(node.initial_ground_truth, node.ground_truth)
        if max_wheel_speed < args.required_wheel_speed:
            raise RuntimeError(
                f"Wheel speed stayed below threshold: {max_wheel_speed:.3f} < "
                f"{args.required_wheel_speed:.3f} rad/s"
            )
        if odom_displacement < args.required_odom_displacement:
            raise RuntimeError(
                f"Wheel odometry displacement stayed below threshold: {odom_displacement:.3f} < "
                f"{args.required_odom_displacement:.3f} m"
            )
        if args.require_ground_truth and ground_truth_displacement < args.required_ground_truth_displacement:
            raise RuntimeError(
                f"Ground-truth displacement stayed below threshold: {ground_truth_displacement:.3f} < "
                f"{args.required_ground_truth_displacement:.3f} m"
            )

        print(
            {
                "wheel_speed_rad_s": max_wheel_speed,
                "wheel_odom_displacement_m": odom_displacement,
                "ground_truth_displacement_m": ground_truth_displacement,
                "odom_frame": node.odom.header.frame_id if node.odom else "",
                "odom_child_frame": node.odom.child_frame_id if node.odom else "",
            }
        )
        return 0
    except RuntimeError as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
