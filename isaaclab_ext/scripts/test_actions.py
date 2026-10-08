#!/usr/bin/env python3
"""Validate contract-derived Isaac Lab action groups and apply a bounded joint probe."""

from __future__ import annotations

import sys

from spawn_robot import main


if __name__ == "__main__":
    main(["--no-cameras", "--validate-actions", *sys.argv[1:]])
