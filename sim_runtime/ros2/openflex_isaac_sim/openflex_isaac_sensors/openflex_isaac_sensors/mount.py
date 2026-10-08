"""Generic parent-prim and local-pose mounting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class LocalPose:
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "LocalPose":
        value = value or {}
        translation = tuple(float(x) for x in value.get("translation_m", (0.0, 0.0, 0.0)))
        quaternion = tuple(float(x) for x in value.get("quaternion_wxyz", (1.0, 0.0, 0.0, 0.0)))
        if len(translation) != 3 or len(quaternion) != 4:
            raise ValueError("local pose must contain 3 translation and 4 quaternion values")
        norm = sum(x * x for x in quaternion) ** 0.5
        if norm == 0.0:
            raise ValueError("local pose quaternion cannot be zero")
        return cls(translation, tuple(x / norm for x in quaternion))


def resolve_mount_prim_path(parent_prim_path: str, camera_name: str, explicit_mount: str | None = None) -> str:
    if not parent_prim_path.startswith("/"):
        raise ValueError("parent_prim_path must be an absolute USD path")
    if explicit_mount:
        if not explicit_mount.startswith("/"):
            raise ValueError("explicit mount path must be an absolute USD path")
        return explicit_mount
    clean_name = "".join(char if char.isalnum() or char == "_" else "_" for char in camera_name)
    return f"{parent_prim_path.rstrip('/')}/{clean_name}_Mount"


def apply_local_pose(stage: object, prim_path: str, pose: LocalPose) -> None:
    """Apply a WXYZ pose to a USD Xform; imports pxr only inside Isaac Sim."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
    xform = UsdGeom.Xformable(prim)
    translate_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
    orient_op = next((op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient), None)
    translate_op = translate_op or xform.AddTranslateOp()
    orient_op = orient_op or xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*pose.translation_m))
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
        orient_op.Set(Gf.Quatf(pose.quaternion_wxyz[0], Gf.Vec3f(*pose.quaternion_wxyz[1:])))
    else:
        orient_op.Set(Gf.Quatd(pose.quaternion_wxyz[0], Gf.Vec3d(*pose.quaternion_wxyz[1:])))
