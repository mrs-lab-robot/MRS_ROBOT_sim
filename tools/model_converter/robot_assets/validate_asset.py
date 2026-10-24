#!/usr/bin/env python3
"""Validate that the released OpenFleX robot USD is sensor-free."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


REQUIRED_MOUNT_SUFFIXES = (
    "/Geometry/base_link/CameraMount",
    "/Geometry/base_link/Mid360Mount",
    "/Geometry/base_link/lift_carriage_link/head_pitch_link/head_yaw_link/HeadCameraMount",
    "/LeftWristCameraMount",
    "/RightWristCameraMount",
)

EXPECTED_MOUNT_POSES = {
    "/Geometry/base_link/CameraMount": (
        (0.36, 0.0, 0.055),
        (0.70710677, 0.0, -0.70710677, 0.0),
    ),
    "/Geometry/base_link/Mid360Mount": (
        (0.3, 0.0, 0.12),
        (0.6830127, -0.1830127, 0.1830127, -0.6830127),
    ),
}

FORBIDDEN_NAMES = (
    "d435",
    "femto_bolt_camera",
    "openflexsensors",
    "ros2_camera",
    "lidar_ros2_graph",
)


def _values_close(actual: tuple[float, ...], expected: tuple[float, ...], tolerance: float = 1e-5) -> bool:
    return len(actual) == len(expected) and all(
        abs(actual_value - expected_value) <= tolerance
        for actual_value, expected_value in zip(actual, expected)
    )


def _mount_pose_error(prim: object, expected: tuple[tuple[float, ...], tuple[float, ...]]) -> str | None:
    translation_attr = prim.GetAttribute("xformOp:translate")
    orient_attr = prim.GetAttribute("xformOp:orient")
    translation = translation_attr.Get() if translation_attr else None
    orientation = orient_attr.Get() if orient_attr else None
    if translation is None or orientation is None:
        return "missing translate/orient transform"

    actual_translation = tuple(float(value) for value in translation)
    actual_quaternion = (
        float(orientation.GetReal()),
        *(float(value) for value in orientation.GetImaginary()),
    )
    expected_translation, expected_quaternion = expected
    quaternion_matches = _values_close(actual_quaternion, expected_quaternion) or _values_close(
        actual_quaternion, tuple(-value for value in expected_quaternion)
    )
    if not _values_close(actual_translation, expected_translation) or not quaternion_matches:
        return f"expected={expected} actual={(actual_translation, actual_quaternion)}"
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "asset",
        type=Path,
        nargs="?",
        default=repo_root / "sim_runtime" / "assets" / "robots" / "openflex_robot.usda",
    )
    args = parser.parse_args()

    gpu_check = subprocess.run(
        ["nvidia-smi", "-L"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if gpu_check.returncode != 0:
        print("BLOCKED: NVIDIA driver is unavailable; Isaac USD validation was not run")
        return 2

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    result = 0
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(args.asset))
        if stage is None:
            raise RuntimeError(f"cannot open USD: {args.asset}")

        prims = list(stage.Traverse())
        sensor_types = [
            str(prim.GetPath())
            for prim in prims
            if prim.GetTypeName() in {"Camera", "OmniLidar"}
        ]
        forbidden_paths = [
            str(prim.GetPath())
            for prim in prims
            if any(name in str(prim.GetPath()).lower() for name in FORBIDDEN_NAMES)
        ]
        paths = [str(prim.GetPath()) for prim in prims]
        missing_mounts = [
            suffix for suffix in REQUIRED_MOUNT_SUFFIXES if not any(path.endswith(suffix) for path in paths)
        ]
        invalid_mount_poses = {}
        for suffix, expected_pose in EXPECTED_MOUNT_POSES.items():
            matching_prim = next((prim for prim in prims if str(prim.GetPath()).endswith(suffix)), None)
            if matching_prim is not None:
                error = _mount_pose_error(matching_prim, expected_pose)
                if error:
                    invalid_mount_poses[suffix] = error

        if sensor_types or forbidden_paths or missing_mounts or invalid_mount_poses:
            if sensor_types:
                print(f"sensor prims remain: {sensor_types}", flush=True)
            if forbidden_paths:
                print(f"forbidden sensor paths remain: {forbidden_paths}", flush=True)
            if missing_mounts:
                print(f"required mounts missing: {missing_mounts}", flush=True)
            if invalid_mount_poses:
                print(f"invalid mount poses: {invalid_mount_poses}", flush=True)
            result = 1
        else:
            print(f"PASS asset={args.asset}", flush=True)
            print(
                f"prim_count={len(prims)} sensor_prims=0 "
                f"required_mounts={len(REQUIRED_MOUNT_SUFFIXES)}",
                flush=True,
            )
    finally:
        app.close()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
