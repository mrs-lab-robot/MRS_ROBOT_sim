"""Construct an Arena environment from the shared flat-navigation YAML pair."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.configclass import combine_configclass_instances

from mrs_robot_lab.environments.learning.configuration import load_task_configuration

from mrs_arena.embodiments.openflex_navigation import OpenFlexNavigationEmbodiment
from mrs_arena.tasks.openflex_navigation_task import (
    OpenFlexNavigationTask,
    default_navigation_scene_path,
    default_navigation_task_path,
)
from mrs_arena.tasks.navigation_logic import navigation_parameters_from_task


@configclass
class _NavigationGroundAndLightCfg:
    """Local procedural ground avoids Nucleus/network access in reproducible runs."""

    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.32)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75)),
    )


class _OpenFlexNavigationScene(Scene):
    def get_scene_cfg(self):
        return combine_configclass_instances(
            "OpenFlexNavigationSceneCfg",
            super().get_scene_cfg(),
            _NavigationGroundAndLightCfg(),
        )


def make_openflex_navigation_environment(
    *,
    task_spec_path: str | Path | None = None,
    scene_spec_path: str | Path | None = None,
    enable_cameras: bool = False,
) -> IsaacLabArenaEnvironment:
    """Build the Arena navigation task using shared target, reset, and timing data."""
    runtime = load_task_configuration(
        task_spec_path or default_navigation_task_path(),
        scene_spec_path=scene_spec_path or default_navigation_scene_path(),
        backend="arena",
    )
    task = OpenFlexNavigationTask(runtime)
    parameters = navigation_parameters_from_task(runtime.task)
    robot_pose = runtime.task.initial_state.get("robot_pose", {})
    fallback_position, fallback_rotation = runtime.scene.robot_spawn_pose
    position = robot_pose.get("position", fallback_position)
    if len(position) != 3:
        raise ValueError("navigation initial robot position must contain x, y and z")
    if "yaw_rad" in robot_pose:
        yaw = float(robot_pose["yaw_rad"])
        rotation = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
    else:
        rotation = fallback_rotation
    initial_pose = Pose(
        position_xyz=tuple(float(value) for value in position),
        rotation_xyzw=tuple(float(value) for value in rotation),
    )
    embodiment = OpenFlexNavigationEmbodiment(
        parameters=parameters,
        initial_pose=initial_pose,
        enable_cameras=enable_cameras,
    )
    return IsaacLabArenaEnvironment(
        name="openflex_navigation_to_goal",
        scene=_OpenFlexNavigationScene(),
        embodiment=embodiment,
        task=task,
    )
