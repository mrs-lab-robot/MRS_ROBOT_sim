from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mrs_robot_arena.assets.resolver import AssetResolutionError, AssetResolver


class AssetResolverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "isaac_sim_core/assets/robots").mkdir(parents=True)
        (self.root / "isaac_sim_core/assets/robots/openflex_robot.usda").touch()
        contract = self.root / (
            "ros2_pkgs/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml"
        )
        contract.parent.mkdir(parents=True)
        contract.touch()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_resolves_robot_usd_and_embodiment_contract_from_sim_repository(self) -> None:
        resolver = AssetResolver(self.root)

        self.assertEqual(
            resolver.resolve("openflex_robot_usd"),
            self.root / "isaac_sim_core/assets/robots/openflex_robot.usda",
        )
        self.assertEqual(
            resolver.resolve("openflex_embodiment_contract"),
            self.root
            / "ros2_pkgs/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml",
        )

    def test_uses_environment_override_when_root_is_not_passed(self) -> None:
        with patch.dict(os.environ, {"MRS_ROBOT_SIM_ROOT": str(self.root)}):
            resolver = AssetResolver()

        self.assertEqual(
            resolver.resolve("openflex_robot_usd"),
            self.root / "isaac_sim_core/assets/robots/openflex_robot.usda",
        )

    def test_reports_missing_root_instead_of_guessing_a_machine_path(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(AssetResolutionError, "MRS_ROBOT_SIM_ROOT"):
                AssetResolver()

    def test_reports_unknown_and_missing_assets_clearly(self) -> None:
        resolver = AssetResolver(self.root)

        with self.assertRaisesRegex(AssetResolutionError, "unknown asset"):
            resolver.resolve("made_up_asset")

        (self.root / "isaac_sim_core/assets/robots/openflex_robot.usda").unlink()
        with self.assertRaisesRegex(AssetResolutionError, "openflex_robot_usd"):
            resolver.resolve("openflex_robot_usd")


if __name__ == "__main__":
    unittest.main()
