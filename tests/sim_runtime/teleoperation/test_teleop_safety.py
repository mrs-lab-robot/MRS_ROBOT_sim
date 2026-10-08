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


if __name__ == "__main__":
    unittest.main()
