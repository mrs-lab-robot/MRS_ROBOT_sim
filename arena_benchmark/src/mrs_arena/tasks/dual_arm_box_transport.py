"""Arena task adapter for fixed-base bilateral box carry, place, and release."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import torch
import isaaclab.envs.mdp as mdp
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass
from isaaclab_arena.utils.pose import Pose

from mrs_robot_lab.assets.asset_resolver import AssetResolver
from mrs_arena.tasks.transport_logic import (
    create_transport_progress_state,
    update_transport_progress,
)


TASK_INITIAL_BOX_POSITION = (0.65, 0.0, 0.80)
TASK_TARGET_BOX_POSITION = (0.90, 0.0, 0.80)
LEFT_GRIPPER_FINGERS = (
    "openarmx_left_left_finger",
    "openarmx_left_right_finger",
)
RIGHT_GRIPPER_FINGERS = (
    "openarmx_right_left_finger",
    "openarmx_right_right_finger",
)
CONTACT_FORCE_THRESHOLD_N = 0.1


def _build_finger_contact_path(arm_side: str, finger_name: str) -> str:
    """构建完整的 finger USD 路径，基于离线验证的层级结构。

    完整路径：Robot/Geometry/base_link/lift_carriage_link/
             openarmx_{side}_link1/.../openarmx_{side}_link7/<finger>
    """
    link_chain = "/".join(f"openarmx_{arm_side}_link{i}" for i in range(1, 8))
    return (
        f"{{ENV_REGEX_NS}}/Robot/Geometry/base_link/lift_carriage_link/"
        f"{link_chain}/{finger_name}"
    )


@register_task
class DualArmBoxTransportTask(TaskBase):
    """Transport the 0.2 kg box using both grippers, then place and release it.

    The success predicate requires a measured bilateral contact, a 10 cm lift,
    20 cm of carry, placement within the configured target tolerance, release,
    and a one-second settled hold. The sensor paths assume the canonical
    OpenFlex USD finger-link names and must be physically validated on Kit.
    """

    def __init__(
        self,
        *,
        episode_length_s: float = 60.0,
        initial_position: tuple[float, float, float] = TASK_INITIAL_BOX_POSITION,
        target_position: tuple[float, float, float] = TASK_TARGET_BOX_POSITION,
        min_lift_height_m: float = 0.10,
        min_carry_distance_m: float = 0.20,
        position_tolerance_m: float = 0.05,
        max_linear_speed_mps: float = 0.02,
        stable_seconds: float = 1.0,
        contact_force_threshold_n: float = CONTACT_FORCE_THRESHOLD_N,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=(
                "With the base fixed, grasp the box with both arms, lift it, "
                "carry it to the target, place it, release, and let it settle."
            ),
        )
        self.initial_position = _vector3(initial_position, "initial_position")
        self.target_position = _vector3(target_position, "target_position")
        self.min_lift_height_m = _positive(min_lift_height_m, "min_lift_height_m")
        self.min_carry_distance_m = _positive(min_carry_distance_m, "min_carry_distance_m")
        self.position_tolerance_m = _positive(position_tolerance_m, "position_tolerance_m")
        self.max_linear_speed_mps = _positive(max_linear_speed_mps, "max_linear_speed_mps", allow_zero=True)
        self.stable_seconds = _positive(stable_seconds, "stable_seconds")
        self.contact_force_threshold_n = _positive(
            contact_force_threshold_n, "contact_force_threshold_n"
        )

        resolver = AssetResolver()
        self.table = Object(
            name="box_transport_table",
            prim_path="{ENV_REGEX_NS}/BoxTransportTable",
            object_type=ObjectType.BASE,
            usd_path=str(resolver.resolve("openflex_tabletop_usd")),
            initial_pose=Pose(position_xyz=(0.85, 0.0, 0.0)),
        )
        self.box = Object(
            name="transport_box",
            prim_path="{ENV_REGEX_NS}/TransportBox",
            object_type=ObjectType.BASE,
            usd_path=str(resolver.resolve("openflex_transport_box_usd")),
            scale=(2.0, 2.0, 2.0),
            initial_pose=Pose(position_xyz=self.initial_position),
        )
        self.box.object_cfg.spawn.activate_contact_sensors = True
        self.box_rigid_body_path = f"{self.box.prim_path}/PickCube"
        self.box_rigid_body_cfg = RigidObjectCfg(
            prim_path=self.box_rigid_body_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self.initial_position,
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.termination_cfg = self._make_termination_cfg()
        self.events_cfg = self._make_events_cfg()
        self.scene_config = self._make_scene_cfg()

    def _make_scene_cfg(self):
        sensors = [(
            "transport_box_rigid_body",
            RigidObjectCfg,
            self.box_rigid_body_cfg,
        )]
        for side, fingers in (("left", LEFT_GRIPPER_FINGERS), ("right", RIGHT_GRIPPER_FINGERS)):
            filter_paths = [_build_finger_contact_path(side, finger) for finger in fingers]
            contact_sensor_cfg = ContactSensorCfg(
                prim_path=self.box_rigid_body_path,
                filter_prim_paths_expr=filter_paths,
                history_length=1,
                update_period=0.0,
                track_air_time=False,
            )
            sensors.append((
                f"transport_box_{side}_contact",
                ContactSensorCfg,
                contact_sensor_cfg,
            ))
        return make_configclass("DualArmBoxTransportSensorsCfg", sensors)()

    def _make_termination_cfg(self):
        success = TerminationTermCfg(
            func=bilateral_box_transport_success,
            params={
                "object_cfg": SceneEntityCfg("transport_box_rigid_body"),
                "left_contact_cfg": SceneEntityCfg("transport_box_left_contact"),
                "right_contact_cfg": SceneEntityCfg("transport_box_right_contact"),
                "initial_position": self.initial_position,
                "target_position": self.target_position,
                "min_lift_height_m": self.min_lift_height_m,
                "min_carry_distance_m": self.min_carry_distance_m,
                "position_tolerance_m": self.position_tolerance_m,
                "max_linear_speed_mps": self.max_linear_speed_mps,
                "stable_seconds": self.stable_seconds,
                "contact_force_threshold_n": self.contact_force_threshold_n,
            },
        )
        box_dropped = TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": 0.55,
                "asset_cfg": SceneEntityCfg("transport_box_rigid_body"),
            },
        )
        return DualArmBoxTransportTerminationsCfg(success=success, box_dropped=box_dropped)

    def _make_events_cfg(self):
        reset = EventTermCfg(func=reset_transport_progress, mode="reset")
        return make_configclass(
            "DualArmBoxTransportEventsCfg",
            [("reset_transport_progress", EventTermCfg, reset)],
        )()

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        # Mimic needs calibrated bimanual end-effector states and subtask labels.
        # Do not expose a placeholder Mimic config as if it were accepted.
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self):
        return get_viewer_cfg_look_at_object(
            lookat_object=self.box,
            offset=(-1.5, -1.5, 1.5),
        )


@configclass
class DualArmBoxTransportTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    box_dropped: TerminationTermCfg = MISSING


def bilateral_box_transport_success(
    env: Any,
    *,
    object_cfg: SceneEntityCfg,
    left_contact_cfg: SceneEntityCfg,
    right_contact_cfg: SceneEntityCfg,
    initial_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    min_lift_height_m: float,
    min_carry_distance_m: float,
    position_tolerance_m: float,
    max_linear_speed_mps: float,
    stable_seconds: float,
    contact_force_threshold_n: float,
) -> torch.Tensor:
    box = env.scene[object_cfg.name]
    box_position_world = _torch(box.data.root_pos_w)
    origins = _torch(env.scene.env_origins)
    box_position = box_position_world - origins
    box_linear_speed = torch.linalg.vector_norm(_torch(box.data.root_lin_vel_w)[:, :3], dim=-1)
    left_contact = _filtered_contact_mask(env.scene[left_contact_cfg.name], contact_force_threshold_n)
    right_contact = _filtered_contact_mask(env.scene[right_contact_cfg.name], contact_force_threshold_n)

    state = getattr(env, "_mrs_box_transport_progress", None)
    if state is None:
        state = create_transport_progress_state(env.num_envs, device=env.device)
        env._mrs_box_transport_progress = state
    success, phase = update_transport_progress(
        state,
        box_position=box_position,
        box_linear_speed=box_linear_speed,
        initial_position=torch.tensor(initial_position, dtype=box_position.dtype, device=env.device).expand_as(box_position),
        target_position=target_position,
        left_contact=left_contact,
        right_contact=right_contact,
        dt=float(env.step_dt),
        min_lift_height_m=min_lift_height_m,
        min_carry_distance_m=min_carry_distance_m,
        position_tolerance_m=position_tolerance_m,
        max_linear_speed_mps=max_linear_speed_mps,
        stable_seconds=stable_seconds,
    )
    if hasattr(env, "extras"):
        env.extras["task_phase"] = phase
    return success


def reset_transport_progress(env: Any, env_ids: torch.Tensor | None = None) -> None:
    state = getattr(env, "_mrs_box_transport_progress", None)
    if state is None:
        env._mrs_box_transport_progress = create_transport_progress_state(
            env.num_envs, device=env.device
        )
        return
    if env_ids is None:
        env_ids = slice(None)
    for value in state.values():
        value[env_ids] = 0


def _filtered_contact_mask(sensor: Any, threshold_n: float) -> torch.Tensor:
    force_matrix = sensor.data.force_matrix_w
    if force_matrix is None:
        raise RuntimeError(f"filtered contact force is unavailable for sensor {sensor}")
    force_matrix = _torch(force_matrix)
    if force_matrix.ndim != 4 or force_matrix.shape[-1] != 3:
        raise RuntimeError(f"unexpected filtered force matrix shape {tuple(force_matrix.shape)}")
    force_norm = torch.linalg.vector_norm(force_matrix, dim=-1)
    return force_norm.amax(dim=(1, 2)) >= threshold_n


def _torch(value: Any) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = tuple(float(component) for component in value)
    if not all(torch.isfinite(torch.tensor(component)).item() for component in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    number = float(value)
    lower_ok = number >= 0 if allow_zero else number > 0
    if not torch.isfinite(torch.tensor(number)).item() or not lower_ok:
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return number
