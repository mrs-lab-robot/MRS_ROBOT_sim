from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest

import torch

from mrs_robot_arena.teleoperation.action_mapper import JOINT_STATE_NAMES
from scripts.teleop_vr_openflex import (
    _ARENA_DISABLE_FABRIC,
    _camera_step_roles,
    _frame_interval_for_hz,
    _add_positive_finite_float_option,
    _joint_state,
)


class TeleopScriptHelpersTest(unittest.TestCase):
    def test_joint_state_reads_tensor_without_main_local_torch_binding(self) -> None:
        names = (*JOINT_STATE_NAMES, "unmapped_aux_joint")
        positions = torch.arange(len(names), dtype=torch.float32).unsqueeze(0)
        robot = SimpleNamespace(
            joint_names=names,
            data=SimpleNamespace(joint_pos=positions),
        )

        ordered, by_name = _joint_state(robot, "joint_pos")

        self.assertEqual(ordered, list(range(len(JOINT_STATE_NAMES))))
        self.assertEqual(by_name["unmapped_aux_joint"], float(len(names) - 1))

    def test_arena_builder_keeps_fabric_enabled_for_mid360_runtime(self) -> None:
        self.assertFalse(_ARENA_DISABLE_FABRIC)

    def test_frame_interval_converts_rate_to_positive_step_count(self) -> None:
        self.assertEqual(_frame_interval_for_hz(0.01, 15.0), 7)
        self.assertEqual(_frame_interval_for_hz(0.01, 30.0), 3)
        self.assertEqual(_frame_interval_for_hz(1.0, 0.5), 2)
        self.assertEqual(_frame_interval_for_hz(0.1, 1000.0), 1)

    def test_frame_interval_handles_subnormal_rate_without_overflow(self) -> None:
        interval = _frame_interval_for_hz(0.1, 1e-320)

        self.assertIsInstance(interval, int)
        self.assertGreater(interval, 0)

    def test_frame_interval_rejects_non_positive_or_non_finite_rate(self) -> None:
        for bad_dt in (0.0, -0.01, float("inf"), float("nan")):
            with self.subTest(step_dt=bad_dt):
                with self.assertRaises(ValueError):
                    _frame_interval_for_hz(bad_dt, 15.0)
        for bad_hz in (0.0, -15.0, float("inf"), float("nan")):
            with self.subTest(hz=bad_hz):
                with self.assertRaises(ValueError):
                    _frame_interval_for_hz(0.01, bad_hz)

    def test_camera_roles_decouple_stream_cadence_from_record_cadence(self) -> None:
        # record every 2 steps, stream every 3 steps
        self.assertEqual(_camera_step_roles(3, 2, 3), (True, False, True))
        self.assertEqual(_camera_step_roles(4, 2, 3), (True, True, False))
        self.assertEqual(_camera_step_roles(6, 2, 3), (True, True, True))
        self.assertEqual(_camera_step_roles(1, 2, 3), (False, False, False))

    def test_cli_positive_finite_float_option_rejects_invalid_watchdog_timeouts(self) -> None:
        parser = argparse.ArgumentParser()
        _add_positive_finite_float_option(
            parser, "--watchdog-timeout", default=0.25
        )
        self.assertEqual(
            parser.parse_args(["--watchdog-timeout", "0.5"]).watchdog_timeout,
            0.5,
        )

        for invalid in ("0", "-0.1", "inf", "-inf", "nan"):
            with self.subTest(value=invalid):
                with self.assertRaises(SystemExit) as error:
                    parser.parse_args(["--watchdog-timeout", invalid])
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
