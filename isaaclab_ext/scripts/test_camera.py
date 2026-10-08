#!/usr/bin/env python3
"""Run the Isaac Lab camera smoke test, including RGB and depth output."""

from __future__ import annotations

import sys

from spawn_robot import main


if __name__ == "__main__":
    main(sys.argv[1:])
