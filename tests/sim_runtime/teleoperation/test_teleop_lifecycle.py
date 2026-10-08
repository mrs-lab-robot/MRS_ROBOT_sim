from __future__ import annotations

import unittest

from mrs_teleoperation.lifecycle import format_runtime_exit


class TeleopLifecycleTest(unittest.TestCase):
    def test_exit_diagnostic_distinguishes_kit_shutdown_from_vr_timeout(self) -> None:
        message = format_runtime_exit(
            first_app_running=False,
            first_app_exiting=False,
            app_running=False,
            app_exiting=False,
            sim_steps=100,
            received_vr=False,
            pending_exception_type=None,
            pending_exception_message=None,
        )

        self.assertIn("reason=kit_stopped", message)
        self.assertIn("sim_steps=100", message)
        self.assertIn("received_vr=false", message)
        self.assertIn("first_app_running=false", message)
        self.assertNotIn("reason=vr_timeout", message)

    def test_exit_diagnostic_reports_explicit_kit_shutdown(self) -> None:
        message = format_runtime_exit(
            first_app_running=True,
            first_app_exiting=False,
            app_running=False,
            app_exiting=True,
            sim_steps=15,
            received_vr=True,
            pending_exception_type="SystemExit",
            pending_exception_message="Kit requested exit",
        )

        self.assertIn("reason=kit_shutdown_requested", message)
        self.assertIn("received_vr=true", message)
        self.assertIn("pending_exception=SystemExit", message)
        self.assertIn("exception_message=Kit requested exit", message)


if __name__ == "__main__":
    unittest.main()
