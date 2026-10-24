from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

from pxr import Gf, Usd, UsdGeom


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROBOT_ASSET = REPO_ROOT / "sim_runtime" / "assets" / "robots" / "openflex_robot.usda"
ROBOT_ASSET = Path(os.environ.get("OPENFLEX_TEST_ROBOT_USD", SOURCE_ROBOT_ASSET))
NORMALIZER_PATH = (
    REPO_ROOT
    / "tools"
    / "model_converter"
    / "robot_assets"
    / "normalize_usd_negative_scale.py"
)
_NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "openflex_usd_negative_scale_normalizer", NORMALIZER_PATH
)
if _NORMALIZER_SPEC is None or _NORMALIZER_SPEC.loader is None:
    raise RuntimeError(f"cannot load USD scale normalizer: {NORMALIZER_PATH}")
_NORMALIZER_MODULE = importlib.util.module_from_spec(_NORMALIZER_SPEC)
_NORMALIZER_SPEC.loader.exec_module(_NORMALIZER_MODULE)
normalize_usd_negative_scale = _NORMALIZER_MODULE.normalize_usd_negative_scale
LEFT_LINK1_VISUAL = (
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/"
    "openarmx_left_link1/openarmx_left_link1_visual"
)
RIGHT_LINK1_VISUAL = (
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/"
    "openarmx_right_link1/openarmx_right_link1_visual"
)


class UsdNegativeScaleNormalizationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.stage = Usd.Stage.Open(str(ROBOT_ASSET))
        self.assertIsNotNone(self.stage, f"cannot open robot USD: {ROBOT_ASSET}")

    def test_robot_asset_has_no_negative_scale_ops(self) -> None:
        offenders: list[str] = []
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Xformable):
                continue
            for operation in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if operation.GetOpType() != UsdGeom.XformOp.TypeScale:
                    continue
                scale = operation.Get()
                if scale is not None and any(float(component) < 0.0 for component in scale):
                    offenders.append(f"{prim.GetPath()}: {tuple(scale)}")

        self.assertEqual([], offenders, "negative scale still exists in the runtime robot USD")

    def test_left_link1_mirror_is_baked_into_vertices(self) -> None:
        """左臂 link1 必须是右臂的 Y 镜像（已烘焙进顶点）。

        注意：**不要求左右顶点数相等**。aaaba9e 重导网格时把左臂 40 个重复顶点焊掉了
        （2141 -> 2101），面数/拓扑不变、镜像几何不变；严格等长 + 逐索引比对会把这种
        「焊接 + 镜像」误判成回归。这里改成几何容差比对：每个左顶点到镜像右点云的
        最近距离、以及面集合一致性。
        """

        import numpy as np

        left_prim = self.stage.GetPrimAtPath(LEFT_LINK1_VISUAL)
        right_prim = self.stage.GetPrimAtPath(RIGHT_LINK1_VISUAL)
        self.assertTrue(left_prim.IsA(UsdGeom.Mesh))
        self.assertTrue(right_prim.IsA(UsdGeom.Mesh))

        left = UsdGeom.Mesh(left_prim)
        right = UsdGeom.Mesh(right_prim)
        left_points = np.asarray(left.GetPointsAttr().Get(), dtype=float)
        right_points = np.asarray(right.GetPointsAttr().Get(), dtype=float)
        self.assertGreater(len(left_points), 0)
        self.assertGreater(len(right_points), 0)

        # 右臂点云做 Y 镜像（URDF 里是 scale="1 -1 1"）
        mirrored_right = right_points * np.array([1.0, -1.0, 1.0])

        # 每个左顶点到镜像右点云的最近距离
        chunk = 256
        worst = 0.0
        for offset in range(0, len(left_points), chunk):
            block = left_points[offset : offset + chunk]
            distances = np.linalg.norm(block[:, None, :] - mirrored_right[None, :, :], axis=2)
            worst = max(worst, float(distances.min(axis=1).max()))
        self.assertLess(worst, 1e-5, f"left link1 顶点与镜像右臂的最大最近距离 {worst:.3e} m")

        # 面数必须一致（焊接只删重复顶点，不改三角面）
        left_counts = left.GetFaceVertexCountsAttr().Get()
        right_counts = right.GetFaceVertexCountsAttr().Get()
        self.assertEqual(len(right_counts), len(left_counts))
        self.assertEqual(sum(left_counts), sum(right_counts))

    def test_mirrored_link1_has_consistent_face_varying_normals(self) -> None:
        """镜像后的 link1 必须只有一套自洽的法线。

        资产约定是 `primvars:normals`（faceVarying 平面着色，棱角不被糊成平滑面），
        不再写 built-in `normals`（vertex）。历史遗留的 built-in 那组是焊接前的旧数据
        （2141 条 vs 2101 个顶点、索引错位），已删除；这里守住三条：
        不再出现 built-in 法线、primvars 条数与面索引一致、每条单位长且与绕序同向。
        """

        import numpy as np

        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(LEFT_LINK1_VISUAL))
        self.assertEqual(
            [],
            list(mesh.GetNormalsAttr().Get() or []),
            "built-in vertex normals should be gone; asset uses primvars:normals (faceVarying)",
        )

        primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("normals")
        self.assertIsNotNone(primvar, "link1 must carry primvars:normals")
        self.assertEqual(UsdGeom.Tokens.faceVarying, primvar.GetInterpolation())

        normals = np.asarray(primvar.Get(), dtype=float)
        indices = np.asarray(primvar.GetIndices() or [], dtype=int)
        self.assertGreater(len(normals), 0)
        self.assertTrue(
            np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5),
            "primvars:normals must be unit length",
        )

        face_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int).reshape(-1, 3)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        if len(indices):
            per_corner = normals[indices].reshape(-1, 3, 3)
        else:
            self.assertEqual(len(normals), len(face_indices) * 3)
            per_corner = normals.reshape(-1, 3, 3)
        face_normals = per_corner.mean(axis=1)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12

        a, b, c = (points[face_indices[:, i]] for i in range(3))
        geometric = np.cross(b - a, c - a)
        geometric /= np.linalg.norm(geometric, axis=1, keepdims=True) + 1e-12
        agreeing = int((np.einsum("ij,ij->i", geometric, face_normals) > 0).sum())
        self.assertEqual(
            len(face_indices),
            agreeing,
            f"primvars:normals must follow winding; only {agreeing}/{len(face_indices)} do",
        )

    def test_normalization_preserves_mesh_world_vertices_and_topology(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        parent = UsdGeom.Xform.Define(stage, "/Robot/left_link1").GetPrim()
        parent_xform = UsdGeom.Xformable(parent)
        scale_op = parent_xform.AddScaleOp()
        scale_op.Set(Gf.Vec3f(1.0, -1.0, 1.0))

        mesh = UsdGeom.Mesh.Define(stage, "/Robot/left_link1/visual")
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
        mesh.CreateExtentAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 1.0, 0.0)])

        points_before = mesh.GetPointsAttr().Get()
        topology_before = (
            mesh.GetFaceVertexCountsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),
            mesh.GetOrientationAttr().Get(),
        )
        world_before = [
            UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()).Transform(point)
            for point in points_before
        ]

        changed_ops, changed_meshes = normalize_usd_negative_scale(stage)

        self.assertEqual((1, 1), (changed_ops, changed_meshes))
        self.assertEqual((1.0, 1.0, 1.0), tuple(scale_op.Get()))
        points_after = mesh.GetPointsAttr().Get()
        expected_points = [
            Gf.Vec3f(0.0, 0.0, 0.0),
            Gf.Vec3f(1.0, 0.0, 0.0),
            Gf.Vec3f(0.0, -1.0, 0.0),
        ]
        for actual, expected in zip(points_after, expected_points):
            self.assertLessEqual((actual - expected).GetLength(), 1e-6)

        cache_after = UsdGeom.XformCache()
        world_after = [
            cache_after.GetLocalToWorldTransform(mesh.GetPrim()).Transform(point)
            for point in points_after
        ]
        for before, after in zip(world_before, world_after):
            self.assertLessEqual((before - after).GetLength(), 1e-6)

        topology_after = (
            mesh.GetFaceVertexCountsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),
            mesh.GetOrientationAttr().Get(),
        )
        self.assertEqual(topology_before, topology_after)
        self.assertEqual(
            [(0.0, 0.0, 1.0)] * 3,
            [tuple(normal) for normal in mesh.GetNormalsAttr().Get() or []],
        )


if __name__ == "__main__":
    unittest.main()
