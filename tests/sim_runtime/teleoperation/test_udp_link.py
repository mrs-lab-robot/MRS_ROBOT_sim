from __future__ import annotations

import json
import socket
import unittest

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


if __name__ == "__main__":
    unittest.main()
