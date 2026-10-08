from __future__ import annotations

import unittest


try:
    from mrs_teleoperation.camera_stream import (
        CameraFrameAssembler,
        camera_topic_frame,
        encode_camera_frame,
    )
except ImportError:
    CameraFrameAssembler = None
    camera_topic_frame = None
    encode_camera_frame = None


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
                chunk_size=65507,
            )


if __name__ == "__main__":
    unittest.main()
