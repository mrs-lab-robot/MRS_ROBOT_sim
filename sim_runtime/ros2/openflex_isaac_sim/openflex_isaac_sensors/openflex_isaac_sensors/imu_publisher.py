#!/usr/bin/env python3
"""Retired IMU publisher entry point; it never connected to Isaac Sim."""

from .legacy import reject_placeholder_publisher


class ImuPublisher:
    """Compatibility name that refuses to create a fake IMU stream."""

    def __init__(self, *args, **kwargs):
        reject_placeholder_publisher("IMU")


def main(args=None):
    reject_placeholder_publisher("IMU")


if __name__ == "__main__":
    main()
