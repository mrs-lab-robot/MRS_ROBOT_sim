"""Versioned HDF5 container identifiers for new and legacy episodes."""

CURRENT_EPISODE_FORMAT = "mrs_robot_capture_v1"
LEGACY_EPISODE_FORMATS = frozenset({"mrs_robot_arena_teleop_v1"})
SUPPORTED_EPISODE_FORMATS = LEGACY_EPISODE_FORMATS | {CURRENT_EPISODE_FORMAT}

__all__ = [
    "CURRENT_EPISODE_FORMAT",
    "LEGACY_EPISODE_FORMATS",
    "SUPPORTED_EPISODE_FORMATS",
]
