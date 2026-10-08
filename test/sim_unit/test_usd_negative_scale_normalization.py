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
        left_prim = self.stage.GetPrimAtPath(LEFT_LINK1_VISUAL)
        right_prim = self.stage.GetPrimAtPath(RIGHT_LINK1_VISUAL)
        self.assertTrue(left_prim.IsA(UsdGeom.Mesh))
        self.assertTrue(right_prim.IsA(UsdGeom.Mesh))

        left = UsdGeom.Mesh(left_prim)
        right = UsdGeom.Mesh(right_prim)
        left_points = left.GetPointsAttr().Get()
        right_points = right.GetPointsAttr().Get()
        self.assertEqual(len(right_points), len(left_points))
        self.assertGreater(len(left_points), 0)

        mismatch_count = 0
        first_mismatches = []
        for index, (left_point, right_point) in enumerate(zip(left_points, right_points)):
            if any(
                abs(actual - expected) > 1e-6
                for actual, expected in (
                    (float(left_point[0]), float(right_point[0])),
                    (float(left_point[1]), -float(right_point[1])),
                    (float(left_point[2]), float(right_point[2])),
                )
            ):
                mismatch_count += 1
                if len(first_mismatches) < 3:
                    first_mismatches.append((index, tuple(left_point), tuple(right_point)))

        self.assertEqual(
            0,
            mismatch_count,
            "left link1 vertices are not a baked Y-mirror of the right; "
            f"mismatch_count={mismatch_count}, first={first_mismatches}",
        )

    def test_mirrored_link1_has_authored_vertex_normals(self) -> None:
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(LEFT_LINK1_VISUAL))
        points = mesh.GetPointsAttr().Get() or []
        normals = mesh.GetNormalsAttr().Get() or []

        self.assertEqual(UsdGeom.Tokens.vertex, mesh.GetNormalsInterpolation())
        self.assertEqual(
            len(points),
            len(normals),
            "mirrored mesh must have stable vertex normals after scale baking",
        )
        self.assertGreater(len(normals), 0)
        self.assertTrue(
            all(abs(normal.GetLength() - 1.0) < 1e-5 for normal in normals),
            "authored vertex normals must be normalized",
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
