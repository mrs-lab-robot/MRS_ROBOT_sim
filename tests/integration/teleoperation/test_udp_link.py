from __future__ import annotations

import json
import socket
import unittest
from unittest.mock import Mock

from mrs_teleoperation.protocol import CommandFrame
from mrs_teleoperation.udp_link import UdpTeleopLink


class UdpTeleopLinkTest(unittest.TestCase):
    def test_receives_latest_valid_command_and_sends_simulation_state(self) -> None:
        with UdpTeleopLink(command_port=0, state_target=("127.0.0.1", 0)) as link:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addCleanup(sender.close)
            sender.sendto(json.dumps(CommandFrame(seq=2, deadman=True).to_dict()).encode(), link.command_address)
            frame = link.receive_latest()
            self.assertEqual(frame.seq, 2)

            state_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addCleanup(state_receiver.close)
            state_receiver.bind(("127.0.0.1", 0))
            link.state_target = state_receiver.getsockname()
            link.send_state({"type": "state", "seq": 9, "joint_position": [0.0] * 19})
            payload, _ = state_receiver.recvfrom(4096)
            self.assertEqual(json.loads(payload)["seq"], 9)

    def test_overflowing_command_datagram_is_dropped_without_raising(self) -> None:
        with UdpTeleopLink(command_port=0, state_target=("127.0.0.1", 0)) as link:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addCleanup(sender.close)
            oversized = {
                "magic": "MRSAT",
                "version": 1,
                "type": "command",
                "seq": 1,
                "base_twist": [10**400, 0.0, 0.0],
            }
            sender.sendto(json.dumps(oversized).encode(), link.command_address)

            self.assertIsNone(link.receive_latest())

    def test_oversized_trailing_bytes_are_not_hidden_by_receive_buffer(self) -> None:
        with UdpTeleopLink(command_port=0, state_target=("127.0.0.1", 0)) as link:
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.addCleanup(sender.close)
            valid_prefix = json.dumps(CommandFrame(seq=3, deadman=True).to_dict()).encode()
            sender.sendto(valid_prefix + (b" " * 9000) + b"invalid", link.command_address)

            self.assertIsNone(link.receive_latest())

    def test_deeply_nested_command_datagram_is_dropped_without_stopping_receive(self) -> None:
        link = object.__new__(UdpTeleopLink)
        link._command_socket = Mock()
        link._command_socket.recvfrom.side_effect = [
            (("[" * 1100 + "0" + "]" * 1100).encode(), ("127.0.0.1", 1)),
            (json.dumps(CommandFrame(seq=4, deadman=True).to_dict()).encode(), ("127.0.0.1", 1)),
            BlockingIOError,
        ]

        frame = link.receive_latest()

        self.assertEqual(frame.seq, 4)


if __name__ == "__main__":
    unittest.main()
