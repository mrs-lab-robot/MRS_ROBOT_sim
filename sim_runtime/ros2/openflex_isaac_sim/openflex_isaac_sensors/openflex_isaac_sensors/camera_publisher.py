#!/usr/bin/env python3
"""Retired camera publisher entry point; it never connected to Isaac Sim."""

from .legacy import reject_placeholder_publisher


class CameraPublisher:
    """Compatibility name that refuses to create a fake camera stream."""

    def __init__(self, *args, **kwargs):
        reject_placeholder_publisher("camera")


def main(args=None):
    reject_placeholder_publisher("camera")


if __name__ == "__main__":
    main()
