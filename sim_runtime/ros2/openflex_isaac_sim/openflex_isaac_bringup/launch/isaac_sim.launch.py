#!/usr/bin/env python3
"""Reject the incomplete pre-Isaac-Sim-6 bringup entry point."""


def generate_launch_description():
    raise RuntimeError(
        "isaac_sim.launch.py is a retired, incomplete launcher: it did not start "
        "Isaac Sim and depended on placeholder sensor publishers. Use the "
        "validated Isaac Sim 6.0 entry point `ros2 launch "
        "openflex_isaac_bringup sim.launch.py` instead."
    )
