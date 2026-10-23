"""Clean no-distraction task for OpenFlex: empty scene, long horizon, no success.

The purpose is to isolate the robot's own control (base drive, arm IK, sensors)
from scene objects such as tables, boxes, or walls.  The environment only
contains ground + dome light + the robot articulation.

There is intentionally no success predicate: the episode simply times out after
a long horizon so teleoperation / capture sessions can run unattended.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase


@configclass
class OpenFlexEmptyTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class OpenFlexEmptyTask(TaskBase):
    """Long-horizon task with no success condition and no scene objects."""

    def __init__(self):
        super().__init__(
            episode_length_s=3600.0,
            task_description=(
                "Clean empty scene: validate the robot's own control loop "
                "without any table, box, or wall interference."
            ),
        )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return OpenFlexEmptyTerminationsCfg()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self):
        # No success metric: this task has no success predicate, so a success
        # rate recorder would fail its pre-reset assertion.
        return []
