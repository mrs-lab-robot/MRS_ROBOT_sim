"""Resolve canonical robot resources from the unified MRS_ROBOT_sim checkout."""

from __future__ import annotations

import os
from pathlib import Path


class AssetResolutionError(FileNotFoundError):
    """A configured simulation asset or contract could not be found."""


class AssetResolver:
    """Resolve canonical runtime resources for the Isaac Lab adapter."""

    _ASSETS = {
        "openflex_robot_usd": Path("sim_runtime/assets/robots/openflex_robot.usda"),
        "openflex_tabletop_usd": Path("sim_runtime/assets/environments/openflex_tabletop.usda"),
        "openflex_transport_box_usd": Path("sim_runtime/assets/environments/openflex_pick_cube.usda"),
        "openflex_embodiment_contract": Path(
            "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        ),
        "openflex_sensor_params": Path("sim_runtime/config/sensors/sensors.isaac.yaml"),
        "openflex_camera_mounts": Path(
            "sim_runtime/config/sensors/realsense/realsense_robot_mounts.yaml"
        ),
        "openflex_lidar_mounts": Path("sim_runtime/config/sensors/mid360/mid360_robot_mount.yaml"),
        "openflex_swerve_controller": Path(
            "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/config/controllers.isaac.mobile_base.yaml"
        ),
    }

    def __init__(self, repository_root: str | Path | None = None) -> None:
        configured_root = repository_root or os.environ.get("MRS_ROBOT_SIM_ROOT")
        if not configured_root:
            raise AssetResolutionError(
                "MRS_ROBOT_sim repository root is required; pass repository_root or set "
                "MRS_ROBOT_SIM_ROOT."
            )
        self.repository_root = Path(configured_root).expanduser().resolve()

    def resolve(self, asset_name: str) -> Path:
        """Return an existing canonical resource path by its stable contract name."""

        relative_path = self._ASSETS.get(asset_name)
        if relative_path is None:
            raise AssetResolutionError(f"unknown asset: {asset_name}")

        path = (self.repository_root / relative_path).resolve()
        if not path.is_relative_to(self.repository_root):
            raise AssetResolutionError(f"asset escapes MRS_ROBOT_sim repository: {asset_name}")
        if not path.is_file():
            raise AssetResolutionError(f"asset {asset_name} was not found: {path}")
        return path
