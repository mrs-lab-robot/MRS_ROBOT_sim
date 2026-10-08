"""Bridge existing ROS VR controller topics to and from an Arena UDP endpoint."""

from __future__ import annotations

import json
import math
import socket
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float32, Float64, Float64MultiArray

from mrs_robot_arena_bridge.aggregator import CommandAggregator
from mrs_teleoperation.camera_stream import (
    MAX_UDP_DATAGRAM_BYTES,
    CameraFrameAssembler,
    camera_topic_frame,
)


JOINT_NAMES = (
    "lift_joint",
    "openarmx_head_yaw_joint",
    "openarmx_head_pitch_joint",
    *(f"openarmx_left_joint{index}" for index in range(1, 8)),
    "openarmx_left_finger_joint1",
    *(f"openarmx_right_joint{index}" for index in range(1, 8)),
    "openarmx_right_finger_joint1",
)


class ArenaRelayNode(Node):
    def __init__(self) -> None:
        super().__init__("mrs_robot_arena_relay")
        self.declare_parameter("arena_host", "127.0.0.1")
        self.declare_parameter("command_port", 24102)
        self.declare_parameter("state_port", 24103)
        self.declare_parameter("camera_stream_port", 24104)
        self.declare_parameter("state_bind_host", "127.0.0.1")
        self.declare_parameter("camera_bind_host", "127.0.0.1")
        self.declare_parameter("control_rate_hz", 90.0)
        self.declare_parameter("deadman_timeout", 0.25)
        self.declare_parameter("control_mode", "vr")
        self.declare_parameter("vr_estop_topic", "/vr_estop_active")
        self.declare_parameter("left_pose_topic", "/pico_left_controller/pose")
        self.declare_parameter("right_pose_topic", "/pico_right_controller/pose")
        self.declare_parameter("left_grip_topic", "/pico_left_controller/grip")
        self.declare_parameter("right_grip_topic", "/pico_right_controller/grip")
        self.declare_parameter("left_trigger_topic", "/pico_left_controller/trigger")
        self.declare_parameter("right_trigger_topic", "/pico_right_controller/trigger")
        self.declare_parameter("left_joystick_x_topic", "/pico_left_controller/joystick_x")
        self.declare_parameter("left_joystick_y_topic", "/pico_left_controller/joystick_y")
        self.declare_parameter("right_joystick_x_topic", "/pico_right_controller/joystick_x")
        self.declare_parameter("right_joystick_y_topic", "/pico_right_controller/joystick_y")
        self.declare_parameter("left_arm_topic", "/left_forward_position_controller/commands")
        self.declare_parameter("right_arm_topic", "/right_forward_position_controller/commands")
        self.declare_parameter("head_topic", "/head_forward_position_controller/commands")
        self.declare_parameter("lift_jog_topic", "/lift_manual_position_controller/jog_command")
        self.declare_parameter("lift_position_topic", "/lift_position_controller/commands")

        get = lambda name: self.get_parameter(name).value
        self._arena_target = (str(get("arena_host")), int(get("command_port")))
        self._session_id = time.time_ns() & 0x7FFFFFFFFFFFFFFF
        self._aggregator = CommandAggregator(
            deadman_timeout=float(get("deadman_timeout")),
            session_id=self._session_id,
            control_mode=str(get("control_mode")),
        )
        self._tx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx_socket.setblocking(False)
        self._rx_socket.bind((str(get("state_bind_host")), int(get("state_port"))))
        self._camera_rx_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._camera_rx_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._camera_rx_socket.setblocking(False)
        self._camera_rx_socket.bind((str(get("camera_bind_host")), int(get("camera_stream_port"))))
        self._camera_assembler = CameraFrameAssembler()
        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._lift_action_publisher = self.create_publisher(
            Float64,
            "/lift_manual_position_controller/action_position",
            10,
        )
        self._camera_publishers = {
            camera_name: self.create_publisher(
                CompressedImage,
                f"/{camera_topic_frame(camera_name)}/color/image/compressed",
                1,
            )
            for camera_name in (
                "base_d435", "head_d435", "left_wrist_d405", "right_wrist_d405"
            )
        }

        self.create_subscription(Twist, "/cmd_vel", self._on_twist, 10)
        self.create_subscription(Float64MultiArray, str(get("left_arm_topic")), self._on_left_arm, 10)
        self.create_subscription(Float64MultiArray, str(get("right_arm_topic")), self._on_right_arm, 10)
        self.create_subscription(Float64MultiArray, str(get("head_topic")), self._on_head, 10)
        self.create_subscription(Float64, str(get("lift_jog_topic")), self._on_lift, 10)
        self.create_subscription(
            Float64MultiArray, str(get("lift_position_topic")), self._on_lift_position, 10
        )
        self.create_subscription(Bool, str(get("vr_estop_topic")), self._on_estop, 10)
        self.create_subscription(PoseStamped, str(get("left_pose_topic")), self._on_vr_pose, 10)
        self.create_subscription(PoseStamped, str(get("right_pose_topic")), self._on_vr_pose, 10)
        self.create_subscription(PoseStamped, str(get("left_pose_topic")), lambda msg: self._on_controller_pose("left", msg), 10)
        self.create_subscription(PoseStamped, str(get("right_pose_topic")), lambda msg: self._on_controller_pose("right", msg), 10)
        for side in ("left", "right"):
            self.create_subscription(Float32, str(get(f"{side}_grip_topic")), lambda msg, hand=side: self._on_hand_value(hand, "grip", msg), 10)
            self.create_subscription(Float32, str(get(f"{side}_trigger_topic")), lambda msg, hand=side: self._on_hand_value(hand, "trigger", msg), 10)
            self.create_subscription(Float32, str(get(f"{side}_joystick_x_topic")), lambda msg, hand=side: self._on_joystick_axis(hand, 0, msg), 10)
            self.create_subscription(Float32, str(get(f"{side}_joystick_y_topic")), lambda msg, hand=side: self._on_joystick_axis(hand, 1, msg), 10)
        for topic, button_name in (
            ("/pico_right_controller/button_a", "right_a"),
            ("/pico_right_controller/button_b", "right_b"),
            ("/pico_left_controller/button_x", "left_x"),
            ("/pico_left_controller/button_y", "left_y"),
        ):
            self.create_subscription(Bool, topic, lambda msg, name=button_name: self._aggregator.update_button(name, msg.data), 10)
        self._timer = self.create_timer(1.0 / float(get("control_rate_hz")), self._tick)
        self.get_logger().info(
            f"Arena relay ready: command={self._arena_target}, state={self._rx_socket.getsockname()}, "
            f"camera={self._camera_rx_socket.getsockname()}, session={self._session_id}"
        )

    def _on_vr_pose(self, _message: PoseStamped) -> None:
        self._aggregator.update_vr_pose(now=time.monotonic())

    def _on_controller_pose(self, side: str, message: PoseStamped) -> None:
        pose = message.pose
        values = (
            pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
        )
        try:
            self._aggregator.update_controller_pose(side, values)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_hand_value(self, side: str, field: str, message: Float32) -> None:
        try:
            self._aggregator.update_hand_value(side, field, message.data)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_joystick_axis(self, side: str, axis: int, message: Float32) -> None:
        current = getattr(self._aggregator, f"{side}_joystick") or (0.0, 0.0)
        values = [current[0], current[1]]
        values[axis] = float(message.data)
        try:
            self._aggregator.update_joystick(side, values[0], values[1])
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_estop(self, message: Bool) -> None:
        self._aggregator.update_estop(message.data)

    def _on_twist(self, message: Twist) -> None:
        try:
            self._aggregator.update_base_twist(
                message.linear.x,
                message.linear.y,
                message.angular.z,
                now=time.monotonic(),
            )
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_left_arm(self, message: Float64MultiArray) -> None:
        self._update_arm("left", message.data)

    def _on_right_arm(self, message: Float64MultiArray) -> None:
        self._update_arm("right", message.data)

    def _update_arm(self, side: str, values) -> None:
        try:
            self._aggregator.update_arm(side, values)
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f"Ignoring malformed {side} arm target: {error}")

    def _on_head(self, message: Float64MultiArray) -> None:
        try:
            self._aggregator.update_head(message.data)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_lift(self, message: Float64) -> None:
        try:
            self._aggregator.update_lift_velocity(message.data, now=time.monotonic())
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _on_lift_position(self, message: Float64MultiArray) -> None:
        if len(message.data) != 1:
            self.get_logger().warning("Ignoring malformed lift target: expected one position")
            return
        try:
            self._aggregator.update_lift_position(message.data[0])
        except ValueError as error:
            self.get_logger().warning(str(error))

    def _tick(self) -> None:
        frame = self._aggregator.command_frame(now=time.monotonic(), source_time_ns=time.time_ns())
        if frame.lift_action_position is not None:
            self._publish_lift_action(frame.lift_action_position[0])
        payload = json.dumps(frame.to_dict(), separators=(",", ":")).encode("utf-8")
        try:
            self._tx_socket.sendto(payload, self._arena_target)
        except OSError as error:
            self.get_logger().warning(f"Arena command UDP send failed: {error}")
        self._poll_sim_state()
        self._poll_camera_frames()

    def _poll_camera_frames(self) -> None:
        for _ in range(256):
            try:
                packet, _address = self._camera_rx_socket.recvfrom(
                    MAX_UDP_DATAGRAM_BYTES
                )
            except BlockingIOError:
                return
            frame = self._camera_assembler.feed(packet)
            if frame is None:
                continue
            camera_name, _frame_id, jpeg = frame
            message = CompressedImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = camera_topic_frame(camera_name)
            message.format = "jpeg"
            message.data = jpeg
            self._camera_publishers[camera_name].publish(message)

    def _poll_sim_state(self) -> None:
        newest = None
        while True:
            try:
                packet, _address = self._rx_socket.recvfrom(8192)
            except BlockingIOError:
                break
            try:
                state = json.loads(packet.decode("utf-8"))
                positions = tuple(float(value) for value in state["joint_position"])
                velocities = tuple(float(value) for value in state["joint_velocity"])
                if len(positions) != len(JOINT_NAMES) or len(velocities) != len(JOINT_NAMES):
                    continue
                if not all(math.isfinite(value) for value in positions + velocities):
                    continue
                newest = (int(state.get("seq", 0)), positions, velocities)
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        if newest is None:
            return
        _sequence, positions, velocities = newest
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(JOINT_NAMES)
        message.position = list(positions)
        message.velocity = list(velocities)
        self._joint_state_pub.publish(message)
        lift_action_position = self._aggregator.update_lift_state(
            positions[0], now=time.monotonic()
        )
        self._publish_lift_action(lift_action_position)

    def _publish_lift_action(self, position: float) -> None:
        message = Float64()
        message.data = float(position)
        self._lift_action_publisher.publish(message)

    def destroy_node(self) -> bool:
        self._tx_socket.close()
        self._rx_socket.close()
        self._camera_rx_socket.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ArenaRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
