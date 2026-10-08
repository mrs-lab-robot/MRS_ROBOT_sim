#!/usr/bin/env python3
"""Reject the retired bridge launch, which had no simulator transport."""


def generate_launch_description():
    raise RuntimeError(
        "The legacy bridge.launch.py has no Isaac Sim transport and is disabled. "
        "Use `ros2 launch openflex_isaac_bringup sim.launch.py` for the supported "
        "Isaac Sim 6.0 robot/control path."
    )
