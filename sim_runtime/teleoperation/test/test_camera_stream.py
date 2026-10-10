from __future__ import annotations

import unittest


try:
    from mrs_teleoperation.camera_stream import (
        CameraFrameAssembler,
        camera_topic_frame,
        encode_camera_frame,
    )
    from mrs_teleoperation import camera_stream as _camera_stream
    PreviewPeerRegistry = getattr(_camera_stream, "PreviewPeerRegistry", None)
    validate_camera_chunk_size = getattr(_camera_stream, "validate_camera_chunk_size", None)
except ImportError:
    CameraFrameAssembler = None
    PreviewPeerRegistry = None
    camera_topic_frame = None
    encode_camera_frame = None
    validate_camera_chunk_size = None


class CameraStreamTest(unittest.TestCase):
    def test_fragmented_frames_reassemble_out_of_order(self):
        self.assertIsNotNone(encode_camera_frame)
        self.assertIsNotNone(CameraFrameAssembler)
        jpeg = bytes(range(251)) * 7
        packets = encode_camera_frame(
            "head_d435", frame_id=42, sim_time_ns=987654321, jpeg_bytes=jpeg,
            chunk_size=333,
        )
        assembler = CameraFrameAssembler()

        frames = [frame for packet in reversed(packets) if (frame := assembler.feed(packet))]

        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0], ("head_d435", 987654321, jpeg))

    def test_single_datagram_frame_and_ros_camera_mapping(self):
        packet, = encode_camera_frame(
            "left_wrist_d405", frame_id=1, sim_time_ns=2, jpeg_bytes=b"jpeg",
        )

        self.assertEqual(
            CameraFrameAssembler().feed(packet),
            ("left_wrist_d405", 2, b"jpeg"),
        )
        self.assertEqual(camera_topic_frame("left_wrist_d405"), "cam_left")
        self.assertEqual(camera_topic_frame("right_wrist_d405"), "cam_right")

    def test_unknown_camera_and_oversized_udp_chunk_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "相机"):
            encode_camera_frame("unknown", frame_id=1, sim_time_ns=2, jpeg_bytes=b"x")
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            encode_camera_frame(
                "base_d435", frame_id=1, sim_time_ns=2, jpeg_bytes=b"x",
                chunk_size=65508,
            )

    def test_preview_registration_records_peer_and_expires_to_fallback(self):
        registry = PreviewPeerRegistry(lease_s=3.0)
        peer = ("10.20.30.40", 51842)
        fallback = ("192.0.2.10", 24104)

        self.assertTrue(registry.register(b"MRSR1", peer, now=10.0))
        self.assertEqual(registry.active_peer, peer)
        self.assertEqual(registry.target(fallback, now=12.999), peer)
        self.assertEqual(registry.target(fallback, now=13.0), fallback)
        self.assertIsNone(registry.active_peer)

    def test_non_registration_datagrams_are_ignored_without_lease_refresh(self):
        registry = PreviewPeerRegistry(lease_s=3.0)
        peer = ("10.20.30.40", 51842)
        fallback = ("192.0.2.10", 24104)
        self.assertTrue(registry.register(b"MRSR1", peer, now=10.0))

        for packet in (b"MRSR", b"MRSR0", b"MRSR1extra", b"", bytearray(b"MRSR1")):
            self.assertFalse(registry.register(packet, ("10.0.0.2", 12), now=12.5))

        self.assertEqual(registry.target(fallback, now=12.999), peer)
        self.assertEqual(registry.target(fallback, now=13.0), fallback)

    def test_switching_active_preview_peer_reports_one_new_activation(self):
        registry = PreviewPeerRegistry(lease_s=3.0)
        first_peer = ("10.20.30.40", 51842)
        second_peer = ("10.20.30.41", 51843)

        self.assertTrue(registry.register(b"MRSR1", first_peer, now=10.0))
        self.assertFalse(registry.register(b"MRSR1", first_peer, now=11.0))
        self.assertTrue(registry.register(b"MRSR1", second_peer, now=12.0))
        self.assertEqual(registry.active_peer, second_peer)

    def test_camera_chunk_size_cli_bounds_and_udp_datagram_limit(self):
        self.assertEqual(validate_camera_chunk_size(200), 200)
        self.assertEqual(validate_camera_chunk_size(65507), 65507)
        for invalid_size in (199, 65508):
            with self.assertRaisesRegex(ValueError, "camera_chunk_size"):
                validate_camera_chunk_size(invalid_size)

        packets = encode_camera_frame(
            "base_d435", frame_id=1, sim_time_ns=2,
            jpeg_bytes=b"x" * 65510, chunk_size=65507,
        )
        self.assertEqual([len(packet) for packet in packets], [65507, 53])


if __name__ == "__main__":
    unittest.main()
