"""OpenFlex tabletop cube-lifting task built from Arena's procedural assets."""

from __future__ import annotations

from isaaclab_arena.assets.object_library import ProceduralCube, ProceduralTable
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.tasks.lift_object_task import LiftObjectTask
from isaaclab_arena.utils.pose import Pose


TABLE_POSE = Pose(position_xyz=(0.65, 0.0, 0.62))
CUBE_POSE = Pose(position_xyz=(0.40, 0.0, 0.69))
PICK_CUBE_LIFT_DELTA_Z = 0.25
PICK_CUBE_POSITION_TOLERANCE = 0.04


@register_task
class OpenFlexPickCubeTask(LiftObjectTask):
    """Lift the 200 g cube from the tabletop by 25 cm to complete a pick."""

    def __init__(self):
        self.table = ProceduralTable(
            instance_name="pick_cube_table",
            prim_path="{ENV_REGEX_NS}/PickCubeTable",
            initial_pose=TABLE_POSE,
        )
        self.cube = ProceduralCube(
            instance_name="pick_cube",
            prim_path="{ENV_REGEX_NS}/PickCube",
            initial_pose=CUBE_POSE,
        )
        super().__init__(
            lift_object=self.cube,
            background_scene=self.table,
            episode_length_s=12.0,
            goal_position_delta_xyz=(0.0, 0.0, PICK_CUBE_LIFT_DELTA_Z),
            goal_position_tolerance=PICK_CUBE_POSITION_TOLERANCE,
        )
        self.task_description = "Pick up the cube from the table and lift it 0.25 m."

    def get_mimic_env_cfg(self, embodiment_name: str):
        """This first task uses the shared OpenFlex actions, not Mimic data generation."""
        return None
