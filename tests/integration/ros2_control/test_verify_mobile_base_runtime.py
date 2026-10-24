#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


REPO_DIR = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_DIR / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_bringup"
SCRIPT_PATH = PACKAGE_DIR / "scripts" / "verify_mobile_base_runtime.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_mobile_base_runtime", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class VerifyMobileBaseRuntimeTest(unittest.TestCase):
    def test_required_topics_include_clock_when_using_clock_duration(self) -> None:
        runtime = load_module()
        verifier = object.__new__(runtime.MobileBaseRuntimeVerifier)
        verifier.require_ground_truth = False
        verifier.use_clock_duration = True
        verifier.joint_state = SimpleNamespace()
        verifier.odom = SimpleNamespace()
        verifier.ground_truth = None
        verifier.clock_time_sec = None

        self.assertFalse(verifier.required_topics_ready())
        verifier.clock_time_sec = 1.25
        self.assertTrue(verifier.required_topics_ready())

    def test_required_topics_can_fallback_to_wall_duration_without_clock(self) -> None:
        runtime = load_module()
        verifier = object.__new__(runtime.MobileBaseRuntimeVerifier)
        verifier.require_ground_truth = False
        verifier.use_clock_duration = False
        verifier.joint_state = SimpleNamespace()
        verifier.odom = SimpleNamespace()
        verifier.ground_truth = None
        verifier.clock_time_sec = None

        self.assertTrue(verifier.required_topics_ready())

    def test_command_elapsed_seconds_prefers_clock_duration(self) -> None:
        runtime = load_module()
        verifier = object.__new__(runtime.MobileBaseRuntimeVerifier)
        verifier.use_clock_duration = True
        verifier.clock_time_sec = 13.5

        self.assertAlmostEqual(
            verifier.command_elapsed_seconds(
                start_wall_sec=100.0,
                start_clock_sec=10.0,
            ),
            3.5,
        )

    def test_command_elapsed_seconds_falls_back_to_wall_duration(self) -> None:
        runtime = load_module()
        verifier = object.__new__(runtime.MobileBaseRuntimeVerifier)
        verifier.use_clock_duration = True
        verifier.clock_time_sec = None

        self.assertAlmostEqual(
            verifier.command_elapsed_seconds(
                start_wall_sec=100.0,
                start_clock_sec=None,
                now_wall_sec=123.0,
            ),
            23.0,
        )

    def test_default_command_wall_timeout_allows_slow_simulation(self) -> None:
        runtime = load_module()
        self.assertEqual(runtime.default_command_wall_timeout(8.0), 240.0)
        self.assertEqual(runtime.default_command_wall_timeout(0.5), 30.0)


if __name__ == "__main__":
    unittest.main()
