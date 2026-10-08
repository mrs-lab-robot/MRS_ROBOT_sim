#!/usr/bin/env python3
"""Reject the retired launch that started placeholder sensor publishers."""


def generate_launch_description():
    raise RuntimeError(
        "The legacy sensors.launch.py starts publishers that only emitted "
        "fabricated camera, LiDAR, and IMU data. Start the Isaac Sim 6.0 core "
        "with `ros2 launch openflex_isaac_bringup sim.launch.py`; create real "
        "sensor resources from its Sensors page or runtime sensor-control API."
    )
