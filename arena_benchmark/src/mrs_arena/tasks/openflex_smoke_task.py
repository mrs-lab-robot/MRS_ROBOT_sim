"""Deterministic one-step task for validating Arena task and metric wiring."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase


def smoke_task_success(env, minimum_steps: int = 1):
    """Succeed after a real environment step, independent of robot motion."""

    return env.episode_length_buf >= minimum_steps


@configclass
class OpenFlexSmokeTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = TerminationTermCfg(func=smoke_task_success, params={"minimum_steps": 1})


@register_task
class OpenFlexSmokeTask(TaskBase):
    """One-step success task used only for the layered integration smoke test."""

    def __init__(self):
        super().__init__(episode_length_s=10.0, task_description="Advance OpenFlex by one safe step")

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return OpenFlexSmokeTerminationsCfg()

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
