"""Non-blocking localhost UDP transport between Arena and the ROS relay."""

from __future__ import annotations

import json
import socket
from typing import Any

from mrs_teleoperation.protocol import CommandFrame, ProtocolError


class UdpTeleopLink:
    def __init__(
        self,
        *,
        command_host: str = "127.0.0.1",
        command_port: int = 24102,
        state_target: tuple[str, int] = ("127.0.0.1", 24103),
    ) -> None:
        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._command_socket.setblocking(False)
        self._command_socket.bind((command_host, command_port))
        self.command_address = self._command_socket.getsockname()
        self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_target = state_target

    def receive_latest(self) -> CommandFrame | None:
        latest: CommandFrame | None = None
        while True:
            try:
                datagram, _ = self._command_socket.recvfrom(8192)
            except BlockingIOError:
                break
            try:
                payload = json.loads(datagram.decode("utf-8"))
                frame = CommandFrame.from_dict(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ProtocolError, AttributeError):
                continue
            if latest is None or (frame.session_id, frame.seq) > (latest.session_id, latest.seq):
                latest = frame
        return latest

    def send_state(self, payload: dict[str, Any]) -> None:
        body = {"magic": "MRSAT", "version": 1, **payload}
        self._state_socket.sendto(json.dumps(body, separators=(",", ":")).encode("utf-8"), self.state_target)

    def close(self) -> None:
        self._command_socket.close()
        self._state_socket.close()

    def __enter__(self) -> "UdpTeleopLink":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
