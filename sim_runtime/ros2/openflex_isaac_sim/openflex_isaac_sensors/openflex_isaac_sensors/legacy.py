"""Guardrails for retired ROS sensor-publisher placeholders."""


def reject_placeholder_publisher(sensor_name: str) -> None:
    """Stop legacy entry points from publishing fabricated sensor readings."""
    raise RuntimeError(
        f"The legacy {sensor_name} publisher is disabled because it only emitted "
        "fabricated placeholder data. Start the Isaac Sim 6.0 runtime with "
        "`ros2 launch openflex_isaac_bringup sim.launch.py`, then create real "
        "sensor resources from the Sensors page or the runtime sensor-control API."
    )
