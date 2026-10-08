#!/usr/bin/env python3
"""Short source-tree smoke entry point for the standalone rig."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

target = Path(__file__).with_name("realsense_standalone.py")
sys.argv[0] = str(target)
sys.argv.extend(["--camera", "both", "--steps", "120"])
runpy.run_path(str(target), run_name="__main__")
