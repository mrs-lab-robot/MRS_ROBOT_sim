#!/usr/bin/env python3
"""Bake negative USD scale operations into mesh data.

The OpenFlex robot USD uses negative scale to mirror several left-side meshes.
Some Isaac Sim/USD rendering paths flicker on these mirrored transforms. This
utility preserves each mesh's world-space vertices while replacing negative
scale components with positive values.

Run with Isaac Sim's ``python.sh`` so the bundled ``pxr`` bindings are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def _has_negative_component(value: object) -> bool:
    return value is not None and any(float(component) < 0.0 for component in value)


def _vec3_like(value: object, components: object) -> object:
    return type(value)(*(float(component) for component in components))


def _mesh_descendants(root: Usd.Prim) -> list[Usd.Prim]:
    return [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh)]


def _compute_vertex_normals(mesh: UsdGeom.Mesh, points: object) -> list[Gf.Vec3f]:
    face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
    face_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
    if sum(face_counts) != len(face_indices):
        raise ValueError(f"invalid face topology on {mesh.GetPath()}")

    point_vectors = [Gf.Vec3d(*point) for point in points]
    accumulated = [Gf.Vec3d(0.0) for _ in point_vectors]
    offset = 0
    for face_count in face_counts:
        face = face_indices[offset : offset + face_count]
        offset += face_count
        if face_count < 3:
            continue
        if any(index < 0 or index >= len(point_vectors) for index in face):
            raise ValueError(f"face index out of range on {mesh.GetPath()}")

        origin = point_vectors[face[0]]
        for triangle_index in range(1, face_count - 1):
            first_index = face[triangle_index]
            second_index = face[triangle_index + 1]
            area_normal = (point_vectors[first_index] - origin).GetCross(
                point_vectors[second_index] - origin
            )
            if area_normal.GetLength() <= 1e-12:
                continue
            for vertex_index in (face[0], first_index, second_index):
                accumulated[vertex_index] += area_normal

    normals = []
    for normal in accumulated:
        if normal.GetLength() <= 1e-12:
            normals.append(Gf.Vec3f(0.0, 0.0, 1.0))
        else:
            normal.Normalize()
            normals.append(Gf.Vec3f(float(normal[0]), float(normal[1]), float(normal[2])))
    return normals


def _bake_delta_into_mesh(mesh_prim: Usd.Prim, delta: Gf.Matrix4d) -> None:
    mesh = UsdGeom.Mesh(mesh_prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get()
    normals_attr = mesh.GetNormalsAttr()
    normals = normals_attr.Get()
    if points and not normals:
        normals = _compute_vertex_normals(mesh, points)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    if points:
        transformed_points = [
            _vec3_like(point, delta.Transform(Gf.Vec3d(*point))) for point in points
        ]
        points_attr.Set(transformed_points)

        extent_attr = mesh.GetExtentAttr()
        if extent_attr.Get() is not None:
            components = list(zip(*(tuple(point) for point in transformed_points)))
            extent_attr.Set(
                [
                    _vec3_like(
                        points[0],
                        (min(components[0]), min(components[1]), min(components[2])),
                    ),
                    _vec3_like(
                        points[0],
                        (max(components[0]), max(components[1]), max(components[2])),
                    ),
                ]
            )

    if normals:
        normal_delta = delta.GetInverse().GetTranspose()
        transformed_normals = []
        for normal in normals:
            transformed = normal_delta.TransformDir(Gf.Vec3d(*normal))
            if transformed.GetLength() > 0.0:
                transformed.Normalize()
            transformed_normals.append(_vec3_like(normal, transformed))
        normals_attr.Set(transformed_normals)


def normalize_usd_negative_scale(stage: Usd.Stage) -> tuple[int, int]:
    """Normalize static negative scale ops, returning (ops, meshes) changed.

    Mesh points are transformed by the difference between their pre- and
    post-normalization world matrices. This keeps world-space geometry and
    face topology unchanged; only the local point data and scale signs change.
    Animated negative-scale operations are rejected rather than approximated.
    """

    targets: list[tuple[int, str, str]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        for operation in UsdGeom.Xformable(prim).GetOrderedXformOps():
            if operation.GetOpType() != UsdGeom.XformOp.TypeScale:
                continue
            values = [operation.Get()]
            time_samples = operation.GetAttr().GetTimeSamples()
            if time_samples:
                values.extend(operation.Get(time) for time in time_samples)
                if any(_has_negative_component(value) for value in values):
                    raise ValueError(
                        f"animated negative scale is unsupported: {prim.GetPath()} "
                        f"({operation.GetOpName()})"
                    )
                continue
            if _has_negative_component(values[0]):
                targets.append(
                    (
                        prim.GetPath().pathElementCount,
                        str(prim.GetPath()),
                        operation.GetOpName(),
                    )
                )

    # Work from inner transforms outward so each bake sees the current mesh
    # points and does not invalidate a not-yet-processed descendant transform.
    targets.sort(reverse=True)
    changed_ops = 0
    changed_meshes: set[str] = set()

    for _, prim_path, operation_name in targets:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        operation = next(
            (op for op in xformable.GetOrderedXformOps() if op.GetOpName() == operation_name),
            None,
        )
        if operation is None:
            continue
        old_scale = operation.Get()
        if not _has_negative_component(old_scale):
            continue

        meshes = _mesh_descendants(prim)
        before_cache = UsdGeom.XformCache()
        before_world = {
            str(mesh_prim.GetPath()): before_cache.GetLocalToWorldTransform(mesh_prim)
            for mesh_prim in meshes
        }

        operation.Set(
            _vec3_like(
                old_scale,
                (abs(float(component)) for component in old_scale),
            )
        )

        after_cache = UsdGeom.XformCache()
        for mesh_prim in meshes:
            mesh_path = str(mesh_prim.GetPath())
            after_world = after_cache.GetLocalToWorldTransform(mesh_prim)
            try:
                point_delta = before_world[mesh_path] * after_world.GetInverse()
            except Exception as exc:
                raise ValueError(
                    f"cannot bake scale for singular mesh transform: {mesh_path}"
                ) from exc
            _bake_delta_into_mesh(mesh_prim, point_delta)
            changed_meshes.add(mesh_path)
        changed_ops += 1

    return changed_ops, len(changed_meshes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="source USD stage")
    parser.add_argument("output", type=Path, help="normalized USD output")
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.source))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {args.source}")

    normalized_ops, transformed_meshes = normalize_usd_negative_scale(stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage.Flatten().Export(str(args.output))
    print(f"exported={args.output}")
    print(f"negative_scale_ops_normalized={normalized_ops}")
    print(f"meshes_transformed={transformed_meshes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
