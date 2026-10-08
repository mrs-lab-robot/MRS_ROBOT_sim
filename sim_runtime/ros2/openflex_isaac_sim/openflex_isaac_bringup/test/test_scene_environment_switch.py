"""Regression tests for replacing GUI-managed Isaac Sim environments."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import unittest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "start_robot_control_sim.py"
)
SPEC = importlib.util.spec_from_file_location("start_robot_control_sim", SCRIPT_PATH)
START_SIM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(START_SIM)


class _Prim:
    def __init__(self, name: str, references=None):
        self.name = name
        self.references = references or _References()

    def GetName(self) -> str:
        return self.name

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return f"/World/{self.name}"

    def GetReferences(self):
        return self.references


class _References:
    def __init__(self):
        self.paths = []

    def AddReference(self, path):
        self.paths.append(path)
        return not str(path).endswith("_FAIL")


class _World:
    def __init__(self, children):
        self.children = list(children)

    def IsValid(self) -> bool:
        return True

    def GetChildren(self):
        return list(self.children)


class _Stage:
    def __init__(self, children):
        self.world = _World(children)

    def GetPrimAtPath(self, path: str):
        if path == "/World":
            return self.world
        name = path.rsplit("/", 1)[-1]
        return next((prim for prim in self.world.children if prim.name == name), None)

    def RemovePrim(self, path: str) -> None:
        name = path.rsplit("/", 1)[-1]
        self.world.children = [
            prim for prim in self.world.children if prim.name != name
        ]

    def DefinePrim(self, path: str, _type_name: str):
        name = path.rsplit("/", 1)[-1]
        prim = self.GetPrimAtPath(path)
        if prim is None:
            prim = _Prim(name)
            self.world.children.append(prim)
        return prim

    def RenamePrim(self, source: str, target: str):
        prim = self.GetPrimAtPath(source)
        if prim is None or self.GetPrimAtPath(target) is not None:
            return False
        prim.name = target.rsplit("/", 1)[-1]
        return True


class _PromotionFailureStage(_Stage):
    def RenamePrim(self, source: str, target: str):
        if source.startswith("/World/MRS_Staging_") and target == "/World/MRS_NewCube":
            return False
        return super().RenamePrim(source, target)


class SceneEnvironmentSwitchTest(unittest.TestCase):
    def test_twenty_environment_switches_leave_only_the_latest_profile(self):
        stage = _Stage((_Prim("openflex"), _Prim("Ground"), _Prim("DomeLight")))

        def add_usd(asset):
            stage.DefinePrim(f"/World/{asset['prim_name']}", "Xform")
            return {"success": True}

        profiles = (
            [
                {"prim_name": "MRS_NavigationObstacles"},
                {"prim_name": "MRS_Waypoint"},
            ],
            [
                {"prim_name": "MRS_Tabletop"},
                {"prim_name": "MRS_TargetCube"},
            ],
        )
        for index in range(20):
            expected = {asset["prim_name"] for asset in profiles[index % 2]}
            result = START_SIM._replace_scene_environment(
                stage,
                profiles[index % 2],
                add_usd,
                lambda _prim, _asset: {"success": True},
            )
            self.assertTrue(result["success"])
            world_names = {prim.name for prim in stage.world.children}
            self.assertTrue(expected.issubset(world_names))
            self.assertFalse(any(name.startswith("MRS_Staging_") for name in world_names))
            self.assertFalse(any(name.startswith("MRS_Backup_") for name in world_names))
            self.assertEqual(
                {name for name in world_names if name.startswith("MRS_")},
                expected,
            )
            self.assertTrue({"openflex", "Ground", "DomeLight"}.issubset(world_names))

    def test_failed_import_keeps_old_environment_and_removes_all_staging_prims(self):
        stage = _Stage((_Prim("openflex"), _Prim("MRS_OldTable")))

        def add_usd(asset):
            stage.DefinePrim(f"/World/{asset['prim_name']}", "Xform")
            return {
                "success": not asset.get("usd_path", "").endswith("_FAIL"),
                "message": "mock import failure",
            }

        with self.assertRaisesRegex(RuntimeError, "MRS_BrokenCube"):
            START_SIM._replace_scene_environment(
                stage,
                [
                    {"prim_name": "MRS_NewTable", "usd_path": "/assets/table.usd"},
                    {"prim_name": "MRS_BrokenCube", "usd_path": "/assets/cube_FAIL"},
                ],
                add_usd,
                lambda _prim, _asset: {"success": True},
            )

        self.assertEqual(
            {prim.name for prim in stage.world.children},
            {"openflex", "MRS_OldTable"},
        )

    def test_failed_promotion_rolls_back_old_environment_and_removes_staging_prims(self):
        stage = _PromotionFailureStage((_Prim("openflex"), _Prim("MRS_OldTable")))

        def add_usd(asset):
            stage.DefinePrim(f"/World/{asset['prim_name']}", "Xform")
            return {"success": True}

        with self.assertRaisesRegex(RuntimeError, "切换环境prim名称"):
            START_SIM._replace_scene_environment(
                stage,
                [
                    {"prim_name": "MRS_NewTable"},
                    {"prim_name": "MRS_NewCube"},
                ],
                add_usd,
                lambda _prim, _asset: {"success": True},
            )

        self.assertEqual(
            {prim.name for prim in stage.world.children},
            {"openflex", "MRS_OldTable"},
        )

    def test_switching_from_navigation_to_tabletop_removes_startup_stage_environment(self):
        repository_root = Path(__file__).resolve().parents[5]
        stage_text = (
            repository_root
            / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/config/empty_stage.usd"
        ).read_text(encoding="utf-8")
        startup_world_children = re.findall(
            r'^    def\s+\w+\s+"([^"]+)"', stage_text, flags=re.MULTILINE
        )
        stage = _Stage([_Prim("openflex"), *(_Prim(name) for name in startup_world_children)])

        def add_usd(asset):
            stage.world.children.append(_Prim(asset["prim_name"]))
            return {"success": True}

        START_SIM._replace_scene_environment(
            stage,
            [
                {"prim_name": "MRS_NavigationObstacles"},
                {"prim_name": "MRS_Waypoint"},
            ],
            add_usd,
            lambda prim, _asset: {"success": True},
        )
        result = START_SIM._replace_scene_environment(
            stage,
            [
                {"prim_name": "MRS_Tabletop"},
                {"prim_name": "MRS_TargetCube"},
            ],
            add_usd,
            lambda prim, _asset: {"success": True},
        )

        self.assertEqual(
            result["data"]["removed"],
            ["MRS_NavigationObstacles", "MRS_Waypoint"],
        )
        self.assertEqual(
            {prim.name for prim in stage.world.children},
            {"openflex", "DomeLight", "Ground", "MRS_Tabletop", "MRS_TargetCube"},
        )

    def test_switch_removes_old_environment_but_preserves_robot_and_base_stage(self):
        stage = _Stage((
            _Prim("openflex"),
            _Prim("Ground"),
            _Prim("TargetBoxFront"),
            _Prim("TargetBoxSide"),
            _Prim("TargetWall"),
            _Prim("MRS_Waypoint"),
        ))

        def add_usd(asset):
            stage.world.children.append(_Prim(asset["prim_name"]))
            return {"success": True}

        result = START_SIM._replace_scene_environment(
            stage,
            [
                {"prim_name": "MRS_Tabletop"},
                {"prim_name": "MRS_TargetCube"},
            ],
            add_usd,
            lambda prim, _asset: {"success": True},
        )

        self.assertEqual(result["data"]["loaded"], ["MRS_Tabletop", "MRS_TargetCube"])
        self.assertEqual(
            result["data"]["removed"],
            ["MRS_Waypoint", "TargetBoxFront", "TargetBoxSide", "TargetWall"],
        )
        self.assertEqual(
            {prim.name for prim in stage.world.children},
            {"openflex", "Ground", "MRS_Tabletop", "MRS_TargetCube"},
        )

    def test_reloading_same_environment_replaces_instead_of_stacking_prims(self):
        stage = _Stage((_Prim("openflex"), _Prim("MRS_TargetCube")))
        updates = []

        def add_usd(asset):
            stage.DefinePrim(f"/World/{asset['prim_name']}", "Xform")
            return {"success": True}

        def update_usd(prim, asset):
            updates.append((prim.GetName(), asset["x"]))
            return {"success": True}

        for x in (0.8, 1.1):
            START_SIM._replace_scene_environment(
                stage,
                [{"prim_name": "MRS_TargetCube", "x": x}],
                add_usd,
                update_usd,
            )

        self.assertEqual([x for _, x in updates], [0.8, 1.1])
        self.assertEqual(
            [prim.name for prim in stage.world.children], ["openflex", "MRS_TargetCube"]
        )

    def test_new_gui_environment_prim_receives_scale_after_usd_reference_is_added(self):
        stage = _Stage((_Prim("openflex"),))
        updates = []

        def add_usd(asset):
            stage.DefinePrim(f"/World/{asset['prim_name']}", "Xform")
            return {"success": True}

        def update_usd(prim, asset):
            updates.append((prim.GetName(), asset["scale"]))
            return {"success": True}

        START_SIM._replace_scene_environment(
            stage,
            [{"prim_name": "MRS_ArenaTable", "scale": [1.0, 1.0, 0.7]}],
            add_usd,
            update_usd,
        )

        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0][1], [1.0, 1.0, 0.7])
        self.assertEqual(
            [prim.name for prim in stage.world.children], ["openflex", "MRS_ArenaTable"]
        )

    def test_invalid_scale_is_rejected_before_importing_any_environment_asset(self):
        stage = _Stage((_Prim("openflex"),))
        attempted = []

        def add_usd(asset):
            attempted.append(asset["prim_name"])
            stage.world.children.append(_Prim(asset["prim_name"]))
            return {"success": True}

        with self.assertRaisesRegex(ValueError, "scale"):
            START_SIM._replace_scene_environment(
                stage,
                [{"prim_name": "MRS_ArenaTable", "scale": [1.0, 0.0, 1.0]}],
                add_usd,
                lambda _prim, _asset: {"success": True},
            )

        self.assertEqual(attempted, [])
        self.assertEqual([prim.name for prim in stage.world.children], ["openflex"])


if __name__ == "__main__":
    unittest.main()
