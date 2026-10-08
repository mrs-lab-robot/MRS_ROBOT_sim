from collections import deque
import json
from types import SimpleNamespace
from unittest.mock import patch

from builtin_interfaces.msg import Time

from mrs_robot_arena_bridge import relay_node
from mrs_robot_arena_bridge.aggregator import CommandAggregator
from mrs_teleoperation.camera_stream import CameraFrameAssembler, encode_camera_frame


class DatagramSocket:
    def __init__(self, packets):
        self.packets = deque(packets)

    def recvfrom(self, _size):
        if not self.packets:
            raise BlockingIOError
        return self.packets.popleft(), ("127.0.0.1", 12345)


class PublishedImage:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.format = ""
        self.data = b""


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class TransmitSocket:
    def sendto(self, _payload, _target):
        return None


def test_relay_reassembles_and_publishes_arena_jpeg_as_compressed_image():
    jpeg = b"\xff\xd8arena-camera-frame\xff\xd9"
    packets = encode_camera_frame(
        "base_d435", frame_id=7, sim_time_ns=900_000_000,
        jpeg_bytes=jpeg, chunk_size=5,
    )
    node = object.__new__(relay_node.ArenaRelayNode)
    node._camera_rx_socket = DatagramSocket(packets)
    node._camera_assembler = CameraFrameAssembler()
    publisher = Publisher()
    node._camera_publishers = {"base_d435": publisher}
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: "ros-clock-stamp")
    )

    with patch.object(relay_node, "CompressedImage", PublishedImage):
        node._poll_camera_frames()

    assert len(publisher.messages) == 1
    image = publisher.messages[0]
    assert image.header.stamp == "ros-clock-stamp"
    assert image.header.frame_id == "cam_base"
    assert image.format == "jpeg"
    assert image.data == jpeg


def test_relay_publishes_idle_lift_action_from_simulated_joint_state():
    positions = [0.123] + [0.0] * 18
    velocities = [0.0] * 19
    state_packet = json.dumps({
        "seq": 3,
        "joint_position": positions,
        "joint_velocity": velocities,
    }).encode("utf-8")
    node = object.__new__(relay_node.ArenaRelayNode)
    node._rx_socket = DatagramSocket([state_packet])
    node._joint_state_pub = Publisher()
    node._aggregator = CommandAggregator()
    action_publisher = Publisher()
    node._lift_action_publisher = action_publisher
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time())
    )

    node._poll_sim_state()

    assert len(node._joint_state_pub.messages) == 1
    assert len(action_publisher.messages) == 1
    assert action_publisher.messages[0].data == 0.123


def test_relay_publishes_integrated_lift_target_for_lerobot_action_labels():
    aggregator = CommandAggregator()
    aggregator.update_lift_state(0.123, now=10.0)
    aggregator.update_lift_velocity(0.05, now=10.0)
    node = object.__new__(relay_node.ArenaRelayNode)
    node._aggregator = aggregator
    node._lift_action_publisher = Publisher()
    node._tx_socket = TransmitSocket()
    node._arena_target = ("127.0.0.1", 24102)
    node._rx_socket = DatagramSocket([])
    node._camera_rx_socket = DatagramSocket([])
    node._camera_assembler = CameraFrameAssembler()
    node._camera_publishers = {}

    with patch.object(relay_node.time, "monotonic", return_value=10.1):
        node._tick()

    assert len(node._lift_action_publisher.messages) == 1
    assert abs(node._lift_action_publisher.messages[0].data - 0.128) < 1e-9
