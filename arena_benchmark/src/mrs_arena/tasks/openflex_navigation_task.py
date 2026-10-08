"""Arena task adapter for the shared flat-ground point-navigation contract."""

from __future__ import annotations

from pathlib import Path
import os

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass

from mrs_robot_lab.environments.learning.configuration import (
    TaskRuntimeConfiguration,
    load_task_configuration,
)

from mrs_arena.adapters.navigation_observation import navigation_state_tensors
from mrs_arena.tasks.navigation_logic import (
    NavigationParameters,
    advance_success_hold,
    navigation_goal_mask,
    navigation_parameters_from_task,
)


def default_navigation_task_path() -> Path:
    configured_root = os.environ.get("MRS_ROBOT_SIM_ROOT")
    sim_root = Path(configured_root).expanduser().resolve() if configured_root else Path(__file__).resolve().parents[4]
    return sim_root / "sim_runtime/config/tasks/navigation_to_goal.yaml"


def default_navigation_scene_path() -> Path:
    configured_root = os.environ.get("MRS_ROBOT_SIM_ROOT")
    sim_root = Path(configured_root).expanduser().resolve() if configured_root else Path(__file__).resolve().parents[4]
    return sim_root / "sim_runtime/config/scenes/flat_navigation.yaml"


def navigation_task_success(
    env,
    *,
    target_xy: tuple[float, float],
    target_yaw_rad: float,
    position_tolerance_m: float,
    heading_tolerance_rad: float,
    hold_seconds: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate successful episodes after a continuous in-tolerance hold."""
    robot = env.scene[asset_cfg.name]
    position_xy, yaw, _velocity_world_xy, _angular_velocity_z = navigation_state_tensors(
        robot,
        env.scene.env_origins,
    )
    parameters = NavigationParameters(
        target_xy=target_xy,
        target_yaw_rad=target_yaw_rad,
        position_tolerance_m=position_tolerance_m,
        heading_tolerance_rad=heading_tolerance_rad,
        hold_seconds=hold_seconds,
        episode_length_s=float(env.max_episode_length) * float(env.step_dt),
    )
    inside_goal = navigation_goal_mask(position_xy, yaw, parameters)
    held = getattr(env, "_mrs_navigation_success_hold", None)
    if held is None:
        held = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    held, success = advance_success_hold(
        held,
        inside_goal,
        dt=float(env.step_dt),
        required_seconds=hold_seconds,
    )
    env._mrs_navigation_success_hold = held
    return success


def reset_navigation_success_hold(env, env_ids: torch.Tensor | None = None) -> None:
    """Clear the per-environment success timer whenever Arena resets an episode."""
    held = getattr(env, "_mrs_navigation_success_hold", None)
    if held is None:
        held = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._mrs_navigation_success_hold = held
    if env_ids is None:
        held.zero_()
    else:
        held[env_ids] = 0.0


@register_task
class OpenFlexNavigationTask(TaskBase):
    """Evaluate the same position, heading, hold-time, and timeout criteria as Lab."""

    def __init__(
        self,
        runtime_configuration: TaskRuntimeConfiguration | None = None,
        *,
        task_spec_path: str | Path | None = None,
        scene_spec_path: str | Path | None = None,
    ):
        if runtime_configuration is None:
            runtime_configuration = load_task_configuration(
                task_spec_path or default_navigation_task_path(),
                scene_spec_path=scene_spec_path or default_navigation_scene_path(),
                backend="arena",
            )
        self.runtime_configuration = runtime_configuration
        self.parameters = navigation_parameters_from_task(runtime_configuration.task)
        super().__init__(
            episode_length_s=self.parameters.episode_length_s,
            task_description=runtime_configuration.task.description,
        )
        params = {
            "target_xy": self.parameters.target_xy,
            "target_yaw_rad": self.parameters.target_yaw_rad,
            "position_tolerance_m": self.parameters.position_tolerance_m,
            "heading_tolerance_rad": self.parameters.heading_tolerance_rad,
            "hold_seconds": self.parameters.hold_seconds,
            "asset_cfg": SceneEntityCfg("robot"),
        }
        self._termination_cfg = make_configclass(
            "OpenFlexNavigationTerminationsCfg",
            [
                (
                    "time_out",
                    TerminationTermCfg,
                    TerminationTermCfg(func=mdp.time_out, time_out=True),
                ),
                (
                    "success",
                    TerminationTermCfg,
                    TerminationTermCfg(func=navigation_task_success, params=params),
                ),
            ],
        )()
        self._events_cfg = make_configclass(
            "OpenFlexNavigationEventsCfg",
            [
                (
                    "reset_navigation_success_hold",
                    EventTermCfg,
                    EventTermCfg(func=reset_navigation_success_hold, mode="reset"),
                ),
            ],
        )()

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return self._termination_cfg

    def get_events_cfg(self):
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
