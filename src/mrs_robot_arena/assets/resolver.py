"""Resolve canonical robot resources from the sibling MRS_ROBOT_sim checkout."""

from __future__ import annotations

import os
from pathlib import Path


class AssetResolutionError(FileNotFoundError):
    """A configured simulation asset or contract could not be found."""


class AssetResolver:
    """Find canonical OpenFleX assets without copying them into this repository."""

    _ASSETS = {
        "openflex_robot_usd": Path("isaac_sim_core/assets/robots/openflex_robot.usda"),
        "openflex_embodiment_contract": Path(
            "ros2_pkgs/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        ),
        "openflex_sensor_params": Path("isaac_sim_core/config/sensor_params/sensors.isaac.yaml"),
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
