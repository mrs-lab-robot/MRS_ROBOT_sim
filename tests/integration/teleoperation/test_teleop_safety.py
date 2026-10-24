from __future__ import annotations

import unittest

from mrs_teleoperation.protocol import CommandFrame
from mrs_teleoperation.safety import CommandWatchdog


class CommandWatchdogTest(unittest.TestCase):
    def test_stale_estop_or_deadman_release_disables_motion(self) -> None:
        watchdog = CommandWatchdog(timeout_seconds=0.25)
        watchdog.accept(CommandFrame(seq=1, deadman=True), now=10.0)
        self.assertTrue(watchdog.motion_enabled(now=10.1))
        self.assertFalse(watchdog.motion_enabled(now=10.251))

        watchdog.accept(CommandFrame(seq=2, deadman=True, estop=True), now=11.0)
        self.assertFalse(watchdog.motion_enabled(now=11.01))

    def test_old_or_duplicate_sequence_is_rejected(self) -> None:
        watchdog = CommandWatchdog(timeout_seconds=0.25)
        self.assertTrue(watchdog.accept(CommandFrame(seq=5, deadman=True), now=1.0))
        self.assertFalse(watchdog.accept(CommandFrame(seq=5, deadman=True), now=1.1))
        self.assertFalse(watchdog.accept(CommandFrame(seq=4, deadman=True), now=1.2))
        self.assertEqual(watchdog.latest.seq, 5)

    def test_rejects_non_finite_or_non_positive_timeout(self) -> None:
        for invalid_timeout in (float("inf"), float("-inf"), float("nan"), 0.0, -0.25):
            with self.subTest(timeout=invalid_timeout):
                with self.assertRaises(ValueError):
                    CommandWatchdog(timeout_seconds=invalid_timeout)

    def test_rejects_non_finite_clock_timestamps(self) -> None:
        for timestamp in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(timestamp=timestamp):
                watchdog = CommandWatchdog(timeout_seconds=0.25)

                self.assertFalse(
                    watchdog.accept(CommandFrame(seq=1, deadman=True), now=timestamp)
                )
                self.assertFalse(watchdog.motion_enabled(now=timestamp))

    def test_clock_regression_or_future_command_timestamp_disables_motion(self) -> None:
        watchdog = CommandWatchdog(timeout_seconds=0.25)
        self.assertTrue(watchdog.accept(CommandFrame(seq=1, deadman=True), now=10.0))

        self.assertFalse(watchdog.motion_enabled(now=9.9))
        self.assertFalse(watchdog.accept(CommandFrame(seq=2, deadman=True), now=9.9))
        self.assertTrue(watchdog.accept(CommandFrame(seq=2, deadman=True), now=100.0))
        self.assertFalse(watchdog.motion_enabled(now=10.1))

    def test_backward_clock_after_expiration_cannot_resurrect_deadman(self) -> None:
        watchdog = CommandWatchdog(timeout_seconds=0.25)
        self.assertTrue(watchdog.accept(CommandFrame(seq=1, deadman=True), now=10.0))

        self.assertFalse(watchdog.motion_enabled(now=20.0))
        self.assertFalse(watchdog.motion_enabled(now=10.1))
        self.assertFalse(watchdog.accept(CommandFrame(seq=2, deadman=True), now=10.2))
        self.assertFalse(watchdog.motion_enabled(now=10.3))


if __name__ == "__main__":
    unittest.main()
