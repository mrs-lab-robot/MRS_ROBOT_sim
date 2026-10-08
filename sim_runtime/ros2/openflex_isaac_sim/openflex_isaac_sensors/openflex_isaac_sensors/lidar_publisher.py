#!/usr/bin/env python3
"""Retired LiDAR publisher entry point; it never connected to Isaac Sim."""

from .legacy import reject_placeholder_publisher


class LidarPublisher:
    """Compatibility name that refuses to create a fake point cloud."""

    def __init__(self, *args, **kwargs):
        reject_placeholder_publisher("LiDAR")


def main(args=None):
    reject_placeholder_publisher("LiDAR")


if __name__ == "__main__":
    main()
