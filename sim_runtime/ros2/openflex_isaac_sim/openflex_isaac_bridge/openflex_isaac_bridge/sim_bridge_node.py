#!/usr/bin/env python3
"""Retired bridge scaffold; no Isaac Sim transport was implemented."""


def _reject_unimplemented_bridge():
    raise RuntimeError(
        "The legacy IsaacSimBridgeNode is disabled: it did not forward commands "
        "or control the simulator and could report reset/pause success without "
        "performing either operation. Start the supported Isaac Sim 6.0 runtime "
        "with `ros2 launch openflex_isaac_bringup sim.launch.py`."
    )


class IsaacSimBridgeNode:
    """Compatibility name that refuses to advertise a nonfunctional bridge."""

    def __init__(self, *args, **kwargs):
        _reject_unimplemented_bridge()


def main(args=None):
    _reject_unimplemented_bridge()


if __name__ == "__main__":
    main()
