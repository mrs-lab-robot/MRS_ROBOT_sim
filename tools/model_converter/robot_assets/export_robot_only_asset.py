#!/usr/bin/env python3
"""Export the current OpenFlex USD without camera/LiDAR payloads.

This script must run with Isaac Sim's ``python.sh`` because the bundled USD
Python bindings are not available in the system interpreter.
"""

from __future__ import annotations

import argparse
from pathlib import Path


SENSOR_PRIMS = (
    "/openarmx_integrated/Geometry/base_link/Sensors",
    "/openarmx_integrated/Geometry/base_link/d435_link",
    "/openarmx_integrated/Geometry/base_link/mid360_link",
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/head_pitch_link/head_yaw_link/Sensors",
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/head_pitch_link/head_yaw_link/femto_bolt_camera",
    "/openarmx_integrated/Materials/d435_color",
    "/openarmx_integrated/Materials/mid360_color",
)

MOUNT_PRIMS = (
    "/openarmx_integrated/Geometry/base_link/CameraMount",
    "/openarmx_integrated/Geometry/base_link/Mid360Mount",
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/head_pitch_link/head_yaw_link/HeadCameraMount",
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/openarmx_left_link1/openarmx_left_link2/openarmx_left_link3/openarmx_left_link4/openarmx_left_link5/openarmx_left_link6/openarmx_left_link7/OpenFlexRuntimeLinks/openarmx_left_hand/LeftWristCameraMount",
    "/openarmx_integrated/Geometry/base_link/lift_carriage_link/openarmx_right_link1/openarmx_right_link2/openarmx_right_link3/openarmx_right_link4/openarmx_right_link5/openarmx_right_link6/openarmx_right_link7/OpenFlexRuntimeLinks/openarmx_right_hand/RightWristCameraMount",
)

MOUNT_POSES = {
    "/openarmx_integrated/Geometry/base_link/CameraMount": (
        (0.36, 0.0, 0.055),
        (0.70710677, 0.0, -0.70710677, 0.0),
    ),
    "/openarmx_integrated/Geometry/base_link/Mid360Mount": (
        (0.3, 0.0, 0.12),
        (0.6830127, -0.1830127, 0.1830127, -0.6830127),
    ),
}


def _set_mount_pose(stage: object, path: str, pose: tuple[tuple[float, ...], tuple[float, ...]]) -> None:
    from pxr import Gf, UsdGeom

    translation, quaternion = pose
    prim = stage.DefinePrim(path, "Xform")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:]))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from pxr import Usd
        from normalize_usd_negative_scale import normalize_usd_negative_scale

        stage = Usd.Stage.Open(str(args.source))
        if stage is None:
            raise RuntimeError(f"failed to open USD: {args.source}")

        removed = []
        for path in SENSOR_PRIMS:
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
                removed.append(path)

        created = []
        for path in MOUNT_PRIMS:
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Xform")
                created.append(path)
            if path in MOUNT_POSES:
                _set_mount_pose(stage, path, MOUNT_POSES[path])

        normalized_ops, transformed_meshes = normalize_usd_negative_scale(stage)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        flat = stage.Flatten()
        flat.Export(str(args.output))

        check = Usd.Stage.Open(str(args.output))
        remaining = [
            str(prim.GetPath())
            for prim in check.Traverse()
            if prim.GetTypeName() in ("Camera", "OmniLidar")
        ]
        if remaining:
            raise RuntimeError(f"sensor prims remain in robot asset: {remaining}")

        print(f"exported={args.output}")
        print(f"removed={len(removed)}")
        print(f"mounts_created={len(created)}")
        print(f"negative_scale_ops_normalized={normalized_ops}")
        print(f"meshes_transformed={transformed_meshes}")
        print(f"camera_or_lidar_remaining={len(remaining)}")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
