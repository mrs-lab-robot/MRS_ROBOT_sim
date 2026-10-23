"""First-party Arena task registry for the canonical OpenFlex embodiment.

The registry is intentionally small and explicit. Arena graph YAMLs remain
available for upstream benchmark tasks, while these profiles point directly at
locally implemented OpenFlex task environments and never duplicate USD assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True)
class OpenFlexTaskProfile:
    task_id: str
    scene_id: str
    display_name: str
    description: str
    factory: str
    randomization_keys: tuple[str, ...] = ()


_PROFILES = (
    OpenFlexTaskProfile(
        task_id="openflex_empty_scene",
        scene_id="openflex_empty_scene",
        display_name="OpenFlex 干净场景（无干扰）",
        description="只有机器人+地面+灯光的空场景，用于隔离物体对控制的干扰。",
        factory="make_openflex_empty_environment",
        randomization_keys=(),
    ),
    OpenFlexTaskProfile(
        task_id="openflex_dual_arm_box_transport",
        scene_id="openflex_box_transport",
        display_name="OpenFlex 双臂搬箱",
        description="双臂抓取箱子、搬运到目标位置并释放。",
        factory="make_openflex_dual_arm_box_environment",
        randomization_keys=("objects", "light"),
    ),
    OpenFlexTaskProfile(
        task_id="openflex_pick_cube",
        scene_id="openflex_pick_cube",
        display_name="OpenFlex 抓取方块",
        description="从桌面抓取方块并完成抬升。",
        factory="make_openflex_pick_cube_environment",
        randomization_keys=("objects", "light"),
    ),
    OpenFlexTaskProfile(
        task_id="openflex_navigation_to_goal",
        scene_id="openflex_navigation",
        display_name="OpenFlex 导航到目标",
        description="在 Arena 导航场景中驶向目标位姿。",
        factory="make_openflex_navigation_environment",
    ),
)

_PROFILE_BY_ID = {profile.task_id: profile for profile in _PROFILES}


def list_openflex_task_profiles() -> tuple[OpenFlexTaskProfile, ...]:
    return _PROFILES


def get_openflex_task_profile(task_id: str) -> OpenFlexTaskProfile:
    try:
        return _PROFILE_BY_ID[str(task_id).strip()]
    except KeyError as error:
        raise KeyError(f"unknown OpenFlex task: {task_id}") from error


def make_openflex_task_environment(
    task_id: str,
    *,
    dataset_dir: str = "outputs/datasets",
    enable_cameras: bool = False,
) -> Any:
    """Build one registered task after Isaac Sim/Kit has been initialized."""

    profile = get_openflex_task_profile(task_id)
    from mrs_arena.environments import openflex_smoke

    if profile.factory == "make_openflex_navigation_environment":
        from mrs_arena.environments.openflex_navigation import (
            make_openflex_navigation_environment,
        )

        return make_openflex_navigation_environment(enable_cameras=enable_cameras)
    factory: Callable[..., Any] = getattr(openflex_smoke, profile.factory)
    return factory(dataset_dir, enable_cameras=enable_cameras)
