"""Compatibility import for the resolver moved to ``mrs_robot_lab``."""

from mrs_robot_lab.assets.asset_resolver import AssetResolutionError, AssetResolver

__all__ = ["AssetResolutionError", "AssetResolver"]
