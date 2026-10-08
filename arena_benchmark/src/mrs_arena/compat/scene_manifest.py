"""Convert a relation-solved Arena scene into the classic Isaac Sim importer schema."""

from __future__ import annotations

import math
import re
from typing import Any


def quaternion_xyzw_to_euler_xyz(quaternion: tuple[float, ...] | list[float]) -> tuple[float, float, float]:
    """Convert Arena's xyzw quaternion to roll, pitch, yaw in radians."""
    if len(quaternion) != 4:
        raise ValueError("Arena asset rotation quaternion must contain four values")
    values = tuple(float(value) for value in quaternion)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Arena asset rotation quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("Arena asset rotation quaternion cannot be zero")
    x, y, z, w = (value / norm for value in values)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def _single_pose(asset: Any) -> Any | None:
    pose = asset.get_initial_pose() if hasattr(asset, "get_initial_pose") else getattr(asset, "initial_pose", None)
    if pose is None:
        return None
    poses = getattr(pose, "poses", None)
    if poses:
        return poses[0]
    midpoint = getattr(pose, "get_midpoint", None)
    if callable(midpoint):
        return midpoint()
    return pose


def _asset_pose(asset: Any, x_offset_m: float) -> dict[str, float]:
    pose = _single_pose(asset)
    if pose is None:
        position = (0.0, 0.0, 0.0)
        rotation = (0.0, 0.0, 0.0, 1.0)
    else:
        position = tuple(float(value) for value in pose.position_xyz)
        rotation = tuple(float(value) for value in pose.rotation_xyzw)
    if len(position) != 3 or any(not math.isfinite(value) for value in position):
        raise ValueError(f"Arena asset {getattr(asset, 'name', '<unnamed>')} has an invalid position")
    roll, pitch, yaw = quaternion_xyzw_to_euler_xyz(rotation)
    return {
        "x": position[0] + x_offset_m,
        "y": position[1],
        "z": position[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not normalized:
        raise ValueError("Arena asset id cannot be converted to a USD prim name")
    return normalized


def _asset_scale(asset: Any) -> list[float] | None:
    scale = getattr(asset, "scale", None)
    if scale is None:
        return None
    if not isinstance(scale, (list, tuple)) or len(scale) != 3:
        raise ValueError(f"Arena asset {getattr(asset, 'name', '<unnamed>')} has an invalid scale")
    values = [float(value) for value in scale]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"Arena asset {getattr(asset, 'name', '<unnamed>')} has an invalid scale")
    return values


def _task_asset_ids(graph_spec: Any) -> tuple[str | None, str | None]:
    target_id = None
    destination_id = None
    task = getattr(graph_spec, "task", None)
    for subtask in getattr(task, "subtasks", ()) or ():
        params = getattr(subtask, "params", {}) or {}
        if target_id is None:
            for key in ("pick_up_object", "lift_object", "object", "target_object", "placeable_object"):
                value = params.get(key)
                if isinstance(value, str):
                    target_id = value
                    break
        if destination_id is None:
            for key in ("destination_location", "destination", "container", "goal_object"):
                value = params.get(key)
                if isinstance(value, str):
                    destination_id = value
                    break
    return target_id, destination_id


def build_openflex_scene_task_manifest(
    arena_env: Any,
    graph_spec: Any,
    *,
    scene_id: str,
    task_id: str,
    source_scene_yaml: str,
    source_task_yaml: str | None,
    x_offset_m: float = 1.05,
) -> dict[str, Any]:
    """Return scene/task dictionaries compatible with the shared GUI and LeRobot page.

    ``arena_env`` must already have had Arena's relation solver applied.  The
    environment is shifted in X as one rigid layout so the desktop scene stays
    in front of the OpenFlex base while preserving all solved relative poses.
    """
    offset = float(x_offset_m)
    if not math.isfinite(offset):
        raise ValueError("Arena scene X offset must be finite")

    background_spec = graph_spec.background
    background_ids = {
        str(getattr(background_spec, "id", "")),
        str(getattr(background_spec, "registry_name", "")),
    }
    object_ids = {
        str(getattr(item, "id", ""))
        for item in getattr(graph_spec, "objects", ()) or ()
    }
    target_id, destination_id = _task_asset_ids(graph_spec)
    scene_assets = []
    prim_by_asset_id: dict[str, str] = {}
    seen_prim_names: set[str] = set()

    for asset in arena_env.scene.assets.values():
        usd_path = getattr(asset, "usd_path", None)
        if not isinstance(usd_path, str) or not usd_path.strip():
            continue
        asset_name = str(getattr(asset, "name", "")).strip()
        if not asset_name:
            continue
        is_background = (
            asset_name in background_ids
            or type(asset).__name__ == "Background"
            or "background" in (getattr(asset, "tags", None) or [])
        )
        if is_background:
            asset_id = str(getattr(background_spec, "id", asset_name))
            role = "background"
        else:
            asset_id = asset_name
            role = "task_object" if asset_id == target_id else "object"
            if asset_id == destination_id:
                role = "destination"
        prim_name = f"MRS_Arena_{_safe_id(asset_id)}"
        if prim_name in seen_prim_names:
            raise ValueError(f"Arena assets map to a duplicate USD prim name: {prim_name}")
        seen_prim_names.add(prim_name)
        descriptor = {
            "asset": usd_path.strip(),
            "prim_name": prim_name,
            "role": role,
            "pose": _asset_pose(asset, offset),
            "randomization": {"enabled": False},
        }
        scale = _asset_scale(asset)
        if scale is not None:
            descriptor["scale"] = scale
        scene_assets.append(descriptor)
        prim_by_asset_id[asset_id] = prim_name
        if is_background:
            prim_by_asset_id[str(getattr(background_spec, "id", asset_name))] = prim_name

    if not scene_assets:
        raise ValueError("Arena scene contains no standalone USD assets for classic Isaac Sim")
    backgrounds = [asset for asset in scene_assets if asset["role"] == "background"]
    if not backgrounds:
        raise ValueError("Arena scene USD background could not be identified")

    if target_id not in prim_by_asset_id:
        target_id = next(
            (asset_id for asset_id in object_ids if asset_id in prim_by_asset_id),
            None,
        )
    if target_id is None:
        target_id = next(
            (asset_id for asset_id in prim_by_asset_id if asset_id not in background_ids),
            None,
        )
    target_component = prim_by_asset_id.get(target_id or "")
    if target_component is None:
        target_component = backgrounds[0]["prim_name"]

    task = getattr(graph_spec, "task", None)
    task_description = str(
        getattr(task, "description", "") or getattr(arena_env, "name", task_id)
    ).strip()
    scene_name = str(getattr(arena_env, "name", "") or getattr(graph_spec, "env_name", scene_id))
    return {
        "scene": {
            "schema_version": 1,
            "id": scene_id,
            "name": scene_name,
            "description": f"IsaacLab-Arena 场景；对象相对位姿由 Arena 关系求解器生成。",
            "backend": "isaacsim",
            "source": {
                "project": "IsaacLab-Arena",
                "scene_yaml": source_scene_yaml,
                "task_yaml": source_task_yaml,
                "placement": "Arena relation solver; whole scene shifted in +X for OpenFlex clearance",
            },
            "assets": scene_assets,
        },
        "task": {
            "schema_version": 1,
            "id": task_id,
            "name": task_description,
            "environment_id": scene_id,
            "target_component": target_component,
            "collection_label": task_description,
            "description": task_description,
            "goal": task_description,
            "required_cameras": ["head", "left", "right"],
            "source": {
                "project": "IsaacLab-Arena",
                "scene_yaml": source_scene_yaml,
                "task_yaml": source_task_yaml,
            },
        },
    }
