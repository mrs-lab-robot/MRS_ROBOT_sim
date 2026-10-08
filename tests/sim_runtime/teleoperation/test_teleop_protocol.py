from __future__ import annotations

import unittest

from mrs_teleoperation.protocol import CommandFrame, ProtocolError


class CommandFrameTest(unittest.TestCase):
    def test_round_trips_named_commands_and_preserves_sequence(self) -> None:
        frame = CommandFrame(
            seq=17,
            source_time_ns=1234,
            deadman=True,
            estop=False,
            base_twist=(0.2, -0.1, 0.3),
            left_arm_position=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            left_controller_pose=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
            lift_position=(0.12,),
        )

        decoded = CommandFrame.from_dict(frame.to_dict())

        self.assertEqual(decoded.seq, 17)
        self.assertEqual(decoded.base_twist, (0.2, -0.1, 0.3))
        self.assertEqual(decoded.left_arm_position, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        self.assertEqual(decoded.left_controller_pose, (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0))
        self.assertEqual(decoded.lift_position, (0.12,))
        self.assertIsNone(decoded.right_arm_position)

    def test_rejects_wrong_vector_size_and_non_finite_values(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "left_arm_position.*7"):
            CommandFrame.from_dict({"version": 1, "seq": 1, "left_arm_position": [0.0] * 6})

        with self.assertRaisesRegex(ProtocolError, "finite"):
            CommandFrame.from_dict({"version": 1, "seq": 1, "base_twist": [0.0, float("nan"), 0.0]})


if __name__ == "__main__":
    unittest.main()
