"""测试SceneSpec加载器：YAML → Isaac Sim USD场景实例化"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from types import ModuleType
from unittest.mock import patch


class MockReference:
    """模拟USD Reference"""

    def __init__(self):
        self.added_refs: list[str] = []

    def AddReference(self, asset_path: str) -> bool:
        """返回True表示成功添加引用"""
        if asset_path and not asset_path.endswith("_FAIL"):
            self.added_refs.append(asset_path)
            return True
        return False


class MockXformable:
    """模拟UsdGeom.Xformable"""

    def __init__(self):
        self.ops: dict[str, Any] = {}

    def AddTranslateOp(self):
        class TranslateOp:
            def __init__(self, parent):
                self.parent = parent

            def Set(self, value):
                self.parent.ops["translate"] = tuple(value)

        return TranslateOp(self)

    def AddRotateXYZOp(self):
        class RotateOp:
            def __init__(self, parent):
                self.parent = parent

            def Set(self, value):
                self.parent.ops["rotate_xyz"] = tuple(value)

        return RotateOp(self)

    def AddScaleOp(self):
        class ScaleOp:
            def __init__(self, parent):
                self.parent = parent

            def Set(self, value):
                self.parent.ops["scale"] = tuple(value)

        return ScaleOp(self)

    def AddOrientOp(self):
        class OrientOp:
            def __init__(self, parent):
                self.parent = parent

            def Set(self, value):
                # value应该是Quatf或Quatd
                self.parent.ops["orient"] = value

        return OrientOp(self)


class MockPrim:
    def __init__(self, path: str, valid: bool = True, stage=None):
        self._path = path
        self._valid = valid
        self._name = path.rsplit("/", 1)[-1]
        self._stage = stage
        self._references = MockReference()
        self._xformable = MockXformable()

    def GetPath(self) -> str:
        return self._path

    def GetName(self) -> str:
        return self._name

    def IsValid(self) -> bool:
        return self._valid

    def GetReferences(self) -> MockReference:
        return self._references

    def GetXformable(self) -> MockXformable:
        return self._xformable

    def GetChildren(self):
        if self._stage is None:
            return []
        prefix = self._path.rstrip("/") + "/"
        children = []
        for path, prim in self._stage.prims.items():
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                children.append(prim)
        return children


class MockNamespaceEdit:
    @staticmethod
    def Rename(path: str, name: str):
        return ("rename", str(path), name)


class MockBatchNamespaceEdit:
    def __init__(self):
        self.edits = []

    def Add(self, edit):
        self.edits.append(edit)


class MockLayer:
    def __init__(self, stage):
        self.stage = stage

    def Apply(self, edits):
        for operation, old_path, new_name in edits.edits:
            old_prim = self.stage.prims.get(old_path)
            if old_prim is None:
                return False
            parent = old_path.rsplit("/", 1)[0]
            new_path = f"{parent}/{new_name}"
            if new_path in self.stage.prims:
                return False
            affected = [
                path for path in self.stage.prims
                if path == old_path or path.startswith(old_path + "/")
            ]
            for path in affected:
                suffix = path[len(old_path):]
                moved = self.stage.prims.pop(path)
                moved._path = new_path + suffix
                moved._name = moved._path.rsplit("/", 1)[-1]
                self.stage.prims[moved._path] = moved
        return True


class MockEditTarget:
    def __init__(self, stage):
        self.layer = MockLayer(stage)

    def GetLayer(self):
        return self.layer


class MockStage:
    def __init__(self):
        self.prims: dict[str, MockPrim] = {
            "/World": MockPrim("/World", stage=self),
            "/World/openflex": MockPrim("/World/openflex", stage=self),
            "/World/Ground": MockPrim("/World/Ground", stage=self),
            "/World/DomeLight": MockPrim("/World/DomeLight", stage=self),
        }
        self.removed: list[str] = []
        self.defined: list[tuple[str, str]] = []
        self.edit_target = MockEditTarget(self)

    def GetPrimAtPath(self, path: str) -> MockPrim:
        return self.prims.get(path, MockPrim(path, valid=False))

    def DefinePrim(self, path: str, type_name: str) -> MockPrim:
        self.defined.append((path, type_name))
        prim = MockPrim(path, stage=self)
        self.prims[path] = prim
        return prim

    def RemovePrim(self, path: Any) -> None:
        path_str = str(path) if not isinstance(path, str) else path
        self.removed.append(path_str)
        for prim_path in list(self.prims):
            if prim_path == path_str or prim_path.startswith(path_str.rstrip("/") + "/"):
                self.prims.pop(prim_path, None)

    def GetEditTarget(self):
        return self.edit_target


def fake_pxr_modules():
    class FakeVec3d(tuple):
        def __new__(cls, *values):
            return tuple.__new__(cls, values)

    class FakeGf:
        Vec3d = FakeVec3d

        @staticmethod
        def Quatd(real, imaginary):
            return (real, tuple(imaginary))

    class FakeUsdGeom:
        @staticmethod
        def Xformable(prim):
            return prim.GetXformable()

    class FakeSdf:
        Path = str
        NamespaceEdit = MockNamespaceEdit
        BatchNamespaceEdit = MockBatchNamespaceEdit

    pxr = ModuleType("pxr")
    pxr.Gf = FakeGf
    pxr.UsdGeom = FakeUsdGeom
    pxr.Sdf = FakeSdf
    return {"pxr": pxr}


class TestSceneSpecLoader(unittest.TestCase):
    def test_load_scene_spec_from_yaml(self):
        """测试从YAML加载SceneSpec数据结构"""
        from openflex_isaac_contract.scene_spec import SceneSpec, ObjectPlacement

        spec = SceneSpec(
            spec_version="1.0",
            scene_id="test_scene",
            description="测试场景",
            base_stage_usd="/path/to/stage.usd",
            robot_spawn_pose=((0.0, 0.0, 0.25), (0.0, 0.0, 0.0, 1.0)),
            objects=[
                ObjectPlacement(
                    object_id="table_01",
                    object_type="furniture",
                    position=(1.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    usd_path="/assets/table.usd",
                )
            ],
        )

        self.assertEqual(spec.scene_id, "test_scene")
        self.assertEqual(len(spec.objects), 1)
        self.assertEqual(spec.objects[0].object_id, "table_01")

    def test_scene_loader_initializes_with_stage(self):
        """测试SceneLoader初始化需要stage"""
        from openflex_isaac_bringup.scene_loader import SceneLoader

        stage = MockStage()
        loader = SceneLoader(stage)

        self.assertIsNotNone(loader.stage)

    def test_clear_scene_preserves_robot_and_ground(self):
        """清理只移除加载器拥有的场景，不碰同级外部prim"""
        from openflex_isaac_bringup.scene_loader import SceneLoader

        stage = MockStage()
        stage.prims["/World/ExternalCamera"] = MockPrim("/World/ExternalCamera", stage=stage)
        stage.DefinePrim("/World/MRSRobotScene", "Xform")
        stage.DefinePrim("/World/MRSRobotScene/OldObject", "Xform")

        loader = SceneLoader(stage)
        removed = loader.clear_scene_objects()

        self.assertIn("/World/MRSRobotScene", removed)
        self.assertIn("/World/ExternalCamera", stage.prims)
        self.assertIn("/World/openflex", stage.prims)
        self.assertIn("/World/Ground", stage.prims)

    def test_instantiate_object_creates_prim(self):
        """测试实例化ObjectPlacement创建prim"""
        from openflex_isaac_bringup.scene_loader import SceneLoader
        from openflex_isaac_contract.scene_spec import ObjectPlacement

        stage = MockStage()
        loader = SceneLoader(stage)

        obj = ObjectPlacement(
            object_id="box_01",
            object_type="prop",
            position=(1.0, 2.0, 0.5),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )

        with patch.dict("sys.modules", fake_pxr_modules()):
            result = loader.instantiate_object(obj)

        self.assertTrue(result["success"])
        self.assertEqual(result["prim_path"], "/World/MRSRobotScene/objects/box_01")
        xform = stage.GetPrimAtPath(result["prim_path"]).GetXformable()
        self.assertEqual(xform.ops["translate"], (1.0, 2.0, 0.5))
        self.assertEqual(xform.ops["orient"], (1.0, (0.0, 0.0, 0.0)))

    def _write_scene(self, scene_id: str, object_id: str, usd_path=None) -> Path:
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(
                {
                    "spec_version": "1.0",
                    "scene_id": scene_id,
                    "description": scene_id,
                    "base_stage_usd": "/stages/empty.usd",
                    "robot_spawn_pose": {
                        "position": [0.0, 0.0, 0.25],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                    "objects": [
                        {
                            "object_id": object_id,
                            "object_type": "prop",
                            "position": [1.0, 0.0, 0.0],
                            "rotation": [0.0, 0.0, 0.0, 1.0],
                            "usd_path": usd_path,
                        }
                    ],
                },
                f,
            )
            return Path(f.name)

    def test_scene_switch_replaces_loader_owned_content_without_residue(self):
        """反复切换场景不残留旧物体，也不删除机器人或外部prim"""
        from openflex_isaac_bringup.scene_loader import SceneLoader

        stage = MockStage()
        stage.prims["/World/ExternalCamera"] = MockPrim("/World/ExternalCamera", stage=stage)
        loader = SceneLoader(stage)
        paths = [
            self._write_scene("scene_a", "table"),
            self._write_scene("scene_b", "box"),
        ]
        try:
            with patch.dict("sys.modules", fake_pxr_modules()):
                for _ in range(20):
                    for path, expected, stale in ((paths[0], "table", "box"), (paths[1], "box", "table")):
                        result = loader.load_scene(path)
                        self.assertTrue(result["success"], result)
                        self.assertTrue(
                            stage.GetPrimAtPath(f"/World/MRSRobotScene/objects/{expected}").IsValid()
                        )
                        self.assertFalse(
                            stage.GetPrimAtPath(f"/World/MRSRobotScene/objects/{stale}").IsValid()
                        )
                        self.assertIn("/World/openflex", stage.prims)
                        self.assertIn("/World/ExternalCamera", stage.prims)
                        env = stage.GetPrimAtPath("/World/MRSRobotScene/Environment")
                        self.assertIn("/stages/empty.usd", env.GetReferences().added_refs)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

    def test_failed_scene_load_keeps_previous_active_scene(self):
        """新场景构建失败时保留原活动场景，且清理临时内容"""
        from openflex_isaac_bringup.scene_loader import SceneLoader

        stage = MockStage()
        loader = SceneLoader(stage)
        good = self._write_scene("good", "kept")
        bad = self._write_scene("bad", "broken", "/assets/table_FAIL")
        try:
            with patch.dict("sys.modules", fake_pxr_modules()):
                first = loader.load_scene(good)
                self.assertTrue(first["success"], first)
                failure = loader.load_scene(bad)
            self.assertFalse(failure["success"])
            self.assertTrue(
                stage.GetPrimAtPath("/World/MRSRobotScene/objects/kept").IsValid()
            )
            self.assertFalse(
                stage.GetPrimAtPath("/World/__MRSRobotSceneStaging/objects/broken").IsValid()
            )
        finally:
            good.unlink(missing_ok=True)
            bad.unlink(missing_ok=True)

    def test_load_scene_replaces_existing_objects(self):
        """测试加载新场景时替换现有物体"""
        from openflex_isaac_bringup.scene_loader import SceneLoader

        stage = MockStage()
        stage.prims["/World/OldTable"] = MockPrim("/World/OldTable", stage=stage)

        loader = SceneLoader(stage)

        # 创建临时测试YAML
        import tempfile
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "spec_version": "1.0",
                    "scene_id": "kitchen",
                    "description": "Kitchen scene",
                    "base_stage_usd": "/stages/empty.usd",
                    "robot_spawn_pose": {
                        "position": [0.0, 0.0, 0.25],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                    "objects": [
                        {
                            "object_id": "counter_01",
                            "object_type": "furniture",
                            "position": [1.5, 0.0, 0.0],
                            "rotation": [0.0, 0.0, 0.0, 1.0],
                        }
                    ],
                },
                f,
            )
            temp_path = Path(f.name)

        try:
            with patch.dict("sys.modules", fake_pxr_modules()):
                result = loader.load_scene(temp_path)

            self.assertTrue(result["success"])
            self.assertEqual(result["scene_id"], "kitchen")
            self.assertIn("/World/OldTable", stage.prims)
            self.assertTrue(stage.GetPrimAtPath("/World/MRSRobotScene/objects/counter_01").IsValid())
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
