"""No-task OpenFlex environment used to verify the Arena embodiment contract."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.configclass import combine_configclass_instances

from mrs_arena.embodiments.openflex import OpenFlexEmbodiment


@configclass
class _GroundAndLightCfg:
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75)),
    )


class _OpenFlexSmokeScene(Scene):
    def get_scene_cfg(self):
        return combine_configclass_instances(
            "OpenFlexSmokeSceneCfg",
            super().get_scene_cfg(),
            _GroundAndLightCfg(),
        )


def _visual_box_cfg(name: str, size: tuple[float, float, float], position: tuple[float, float, float], color):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


@configclass
class _PickCubeTableVisualCfg:
    tabletop: AssetBaseCfg = _visual_box_cfg(
        "pick_cube_table_visual_top", (0.8, 1.5, 0.04), (0.65, 0.0, 0.62), (0.48, 0.29, 0.14)
    )
    front_left_leg: AssetBaseCfg = _visual_box_cfg(
        "pick_cube_table_visual_leg_fl", (0.05, 0.05, 0.62), (0.32, -0.65, 0.31), (0.34, 0.20, 0.09)
    )
    front_right_leg: AssetBaseCfg = _visual_box_cfg(
        "pick_cube_table_visual_leg_fr", (0.05, 0.05, 0.62), (0.98, -0.65, 0.31), (0.34, 0.20, 0.09)
    )
    rear_left_leg: AssetBaseCfg = _visual_box_cfg(
        "pick_cube_table_visual_leg_rl", (0.05, 0.05, 0.62), (0.32, 0.65, 0.31), (0.34, 0.20, 0.09)
    )
    rear_right_leg: AssetBaseCfg = _visual_box_cfg(
        "pick_cube_table_visual_leg_rr", (0.05, 0.05, 0.62), (0.98, 0.65, 0.31), (0.34, 0.20, 0.09)
    )


class _OpenFlexPickCubeScene(_OpenFlexSmokeScene):
    def __init__(self, task):
        super().__init__([task.table, task.cube])

    def get_scene_cfg(self):
        return combine_configclass_instances(
            "OpenFlexPickCubeSceneCfg",
            super().get_scene_cfg(),
            _PickCubeTableVisualCfg(),
        )


class _OpenFlexDualArmBoxScene(_OpenFlexSmokeScene):
    def __init__(self, task):
        super().__init__([task.table, task.box])


def make_openflex_smoke_environment(*, enable_cameras: bool = False) -> IsaacLabArenaEnvironment:
    """Build a single-robot Arena environment with no task, rewards, or metrics."""

    return IsaacLabArenaEnvironment(
        name="openflex_no_task_smoke",
        scene=_OpenFlexSmokeScene(),
        embodiment=OpenFlexEmbodiment(enable_cameras=enable_cameras),
        task=None,
    )


def make_openflex_task_smoke_environment(dataset_dir: str) -> IsaacLabArenaEnvironment:
    """Build the registered one-step task and direct metric output to a caller-owned directory."""

    from mrs_arena.tasks.openflex_smoke_task import OpenFlexSmokeTask

    def configure_recording(cfg):
        cfg.recorders.dataset_export_dir_path = dataset_dir
        cfg.recorders.dataset_filename = "openflex_task_smoke"
        return cfg

    return IsaacLabArenaEnvironment(
        name="openflex_task_smoke",
        scene=_OpenFlexSmokeScene(),
        embodiment=OpenFlexEmbodiment(enable_cameras=False),
        task=OpenFlexSmokeTask(),
        env_cfg_callback=configure_recording,
    )


def make_openflex_pick_cube_environment(dataset_dir: str) -> IsaacLabArenaEnvironment:
    """Build the physical tabletop cube-lift task without any VR dependency."""

    from mrs_arena.tasks.openflex_pick_cube_task import OpenFlexPickCubeTask

    task = OpenFlexPickCubeTask()

    def configure_recording(cfg):
        cfg.recorders.dataset_export_dir_path = dataset_dir
        cfg.recorders.dataset_filename = "openflex_pick_cube"
        return cfg

    return IsaacLabArenaEnvironment(
        name="openflex_pick_cube",
        scene=_OpenFlexPickCubeScene(task),
        embodiment=OpenFlexEmbodiment(enable_cameras=False),
        task=task,
        env_cfg_callback=configure_recording,
    )


def make_openflex_dual_arm_box_environment(dataset_dir: str) -> IsaacLabArenaEnvironment:
    """Build the bilateral carry/place task with actual filtered finger contacts."""
    from mrs_arena.tasks.dual_arm_box_transport import DualArmBoxTransportTask

    task = DualArmBoxTransportTask()

    def configure_recording(cfg):
        cfg.recorders.dataset_export_dir_path = dataset_dir
        cfg.recorders.dataset_filename = "openflex_dual_arm_box_transport"
        return cfg

    return IsaacLabArenaEnvironment(
        name="openflex_dual_arm_box_transport",
        scene=_OpenFlexDualArmBoxScene(task),
        embodiment=OpenFlexEmbodiment(enable_cameras=False),
        task=task,
        env_cfg_callback=configure_recording,
    )
