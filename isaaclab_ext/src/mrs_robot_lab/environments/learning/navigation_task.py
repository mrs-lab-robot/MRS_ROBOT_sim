"""Base-only point navigation using the contract-backed OpenFlex swerve drive."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from mrs_robot_lab.assets.mrs_robot_cfg import MRS_ROBOT_CFG
from mrs_robot_lab.assets.robot_interface import SWERVE_CONFIG
from mrs_robot_lab.controllers.swerve_tensor import (
    compute_swerve_targets_tensor,
    normalized_to_body_twist,
)
from mrs_robot_lab.environments.learning.configuration import load_task_configuration
from mrs_robot_lab.environments.learning.runtime_rates import apply_runtime_rates
from mrs_robot_lab.sensors.camera_cfg import add_capture_cameras


@configclass
class NavigationTaskEnvCfg(DirectRLEnvCfg):
    """Isaac Lab direct-RL interface: 8 observations and 3 normalized actions."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0)
    episode_length_s: float = 30.0
    decimation: int = 4
    task_spec_path: str | None = None
    scene_spec_path: str | None = None
    num_observations: int = 8
    num_actions: int = 3
    observation_space: int = 8
    action_space: int = 3


class NavigationTaskEnvironment(DirectRLEnv):
    """Navigate on a flat floor using odometry-like state and relative goal pose."""

    cfg: NavigationTaskEnvCfg

    def __init__(
        self,
        task_spec_path: str | Path,
        device: str = "cuda:0",
        num_envs: int = 1,
        scene_spec_path: str | Path | None = None,
        physics_hz: float = 120.0,
        control_hz: float = 30.0,
        render_hz: float = 30.0,
        camera_frequencies_hz: Mapping[str, float] | None = None,
        **kwargs,
    ):
        self.runtime_config = load_task_configuration(
            task_spec_path, scene_spec_path=scene_spec_path, backend="isaac_lab"
        )
        self.task_spec = self.runtime_config.task
        self._camera_frequencies_hz = dict(camera_frequencies_hz or {})
        criterion = _criterion(self.task_spec, {"base_pose_at_target", "base_at_target", "reach_target"})
        params = criterion.params
        target = params.get("target_position")
        if not isinstance(target, (tuple, list)) or len(target) < 2:
            raise ValueError("navigation criterion target_position must contain at least x and y")
        self._target_position_xy = tuple(float(component) for component in target[:2])
        self._target_yaw = float(params.get("target_yaw_rad", 0.0))
        self._position_tolerance = criterion.tolerance
        self._heading_tolerance = torch.deg2rad(
            torch.tensor(float(params.get("heading_tolerance_deg", 10.0)), dtype=torch.float32)
        ).item()
        self._hold_seconds = float(params.get("hold_seconds", 1.0))
        if self._heading_tolerance <= 0.0 or self._hold_seconds <= 0.0:
            raise ValueError("navigation heading tolerance and hold_seconds must be positive")

        robot_pose = self.task_spec.initial_state.get("robot_pose", {})
        scene_position, scene_rotation = self.runtime_config.scene.robot_spawn_pose
        position = robot_pose.get("position", scene_position)
        if len(position) != 3:
            raise ValueError("navigation initial robot position must contain x, y and z")
        self._initial_position = tuple(float(component) for component in position)
        self._initial_yaw = float(
            robot_pose.get("yaw_rad", _yaw_from_xyzw(scene_rotation))
        )

        cfg = NavigationTaskEnvCfg()
        cfg.sim.device = device
        cfg.scene.num_envs = num_envs
        apply_runtime_rates(
            cfg,
            physics_hz=physics_hz,
            control_hz=control_hz,
            render_hz=render_hz,
        )
        cfg.episode_length_s = self.task_spec.episode_length_s
        cfg.task_spec_path = str(self.runtime_config.task_spec_path)
        cfg.scene_spec_path = str(self.runtime_config.scene_spec_path)
        seed = kwargs.pop("seed", None)
        if seed is not None:
            cfg.seed = int(seed)
        super().__init__(cfg, **kwargs)

        self._steering_ids, steering_names = self._robot.find_joints(
            list(SWERVE_CONFIG.steering_joint_names), preserve_order=True
        )
        self._wheel_ids, wheel_names = self._robot.find_joints(
            list(SWERVE_CONFIG.wheel_joint_names), preserve_order=True
        )
        if tuple(steering_names) != SWERVE_CONFIG.steering_joint_names:
            raise RuntimeError(f"steering joint order does not match contract: {steering_names}")
        if tuple(wheel_names) != SWERVE_CONFIG.wheel_joint_names:
            raise RuntimeError(f"wheel joint order does not match contract: {wheel_names}")
        self._steering_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self._wheel_linear_targets = torch.zeros((self.num_envs, 4), device=self.device)
        self._previous_wheel_linear_speeds = torch.zeros_like(self._wheel_linear_targets)
        self._success_hold_seconds = torch.zeros(self.num_envs, device=self.device)
        self._success_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_heading_error = torch.zeros(self.num_envs, device=self.device)
        self._target_position = torch.tensor(self._target_position_xy, device=self.device)
        self._target_yaw_tensor = torch.full((self.num_envs,), self._target_yaw, device=self.device)
        self._heading_tolerance_tensor = torch.full((self.num_envs,), self._heading_tolerance, device=self.device)

    def _setup_scene(self) -> None:
        stage_cfg = sim_utils.UsdFileCfg(usd_path=str(self.runtime_config.base_stage_path))
        stage_cfg.func("/World/Environment", stage_cfg)

        robot_cfg = MRS_ROBOT_CFG.copy()
        robot_cfg.prim_path = f"{self.scene.env_regex_ns}/Robot"
        robot_cfg.init_state.pos = self._initial_position
        robot_cfg.init_state.rot = _yaw_to_xyzw(self._initial_yaw)
        self._robot = Articulation(robot_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot
        add_capture_cameras(self.scene, self._camera_frequencies_hz)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 3):
            raise ValueError(f"expected policy action shape ({self.num_envs}, 3), got {tuple(actions.shape)}")
        twist = normalized_to_body_twist(actions, SWERVE_CONFIG)
        steering, wheel_linear_speeds, _ = compute_swerve_targets_tensor(
            twist,
            SWERVE_CONFIG,
            self._previous_wheel_linear_speeds,
            dt=self.step_dt,
        )
        idle = twist.abs().amax(dim=1, keepdim=True) <= 1e-9
        steering = torch.where(idle, self._robot.data.joint_pos[:, self._steering_ids], steering)
        self._steering_targets.copy_(steering)
        self._wheel_linear_targets.copy_(wheel_linear_speeds)
        self._previous_wheel_linear_speeds.copy_(wheel_linear_speeds)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self._steering_targets,
            joint_ids=self._steering_ids,
        )
        self._robot.set_joint_velocity_target_index(
            target=self._wheel_linear_targets / SWERVE_CONFIG.wheel_radius,
            joint_ids=self._wheel_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_pos_local = _as_torch_tensor(self._robot.data.root_pos_w)[:, :2] - self.scene.env_origins[:, :2]
        yaw = _quat_xyzw_to_yaw(_as_torch_tensor(self._robot.data.root_quat_w))
        delta_world = self._target_position.unsqueeze(0) - root_pos_local
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        goal_body_x = cos_yaw * delta_world[:, 0] + sin_yaw * delta_world[:, 1]
        goal_body_y = -sin_yaw * delta_world[:, 0] + cos_yaw * delta_world[:, 1]
        goal_distance = torch.linalg.vector_norm(delta_world, dim=-1)
        heading_error = torch.atan2(
            torch.sin(self._target_yaw_tensor - yaw),
            torch.cos(self._target_yaw_tensor - yaw),
        )

        velocity_world = _as_torch_tensor(self._robot.data.root_lin_vel_w)[:, :2]
        velocity_body_x = cos_yaw * velocity_world[:, 0] + sin_yaw * velocity_world[:, 1]
        velocity_body_y = -sin_yaw * velocity_world[:, 0] + cos_yaw * velocity_world[:, 1]
        observation = torch.stack(
            (
                goal_body_x,
                goal_body_y,
                torch.sin(heading_error),
                torch.cos(heading_error),
                velocity_body_x,
                velocity_body_y,
                _as_torch_tensor(self._robot.data.root_ang_vel_w)[:, 2],
                goal_distance,
            ),
            dim=-1,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        root_position_local = _as_torch_tensor(self._robot.data.root_pos_w)[:, :2] - self.scene.env_origins[:, :2]
        distance = torch.linalg.vector_norm(root_position_local - self._target_position, dim=-1)
        yaw = _quat_xyzw_to_yaw(_as_torch_tensor(self._robot.data.root_quat_w))
        heading_error = torch.abs(
            torch.atan2(
                torch.sin(yaw - self._target_yaw_tensor),
                torch.cos(yaw - self._target_yaw_tensor),
            )
        )
        reward_cfg = self.task_spec.reward_config or {}
        distance_progress = (self._previous_distance - distance) / self.step_dt
        heading_progress = (self._previous_heading_error - heading_error) / self.step_dt
        reward = (
            float(reward_cfg.get("step_penalty", -0.01))
            - distance * float(reward_cfg.get("distance_reward_scale", 1.0))
            + distance_progress * float(reward_cfg.get("distance_progress_reward_scale", 0.0))
            - heading_error * float(reward_cfg.get("heading_error_reward_scale", 0.0))
            + heading_progress * float(reward_cfg.get("heading_progress_reward_scale", 0.0))
            + self._success_mask.float() * float(reward_cfg.get("success_reward", 100.0))
        )
        self._previous_distance.copy_(distance)
        self._previous_heading_error.copy_(heading_error)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        position = _as_torch_tensor(self._robot.data.root_pos_w)[:, :2] - self.scene.env_origins[:, :2]
        yaw = _quat_xyzw_to_yaw(_as_torch_tensor(self._robot.data.root_quat_w))
        distance = torch.linalg.vector_norm(position - self._target_position, dim=-1)
        heading_error = torch.abs(
            torch.atan2(
                torch.sin(yaw - self._target_yaw_tensor),
                torch.cos(yaw - self._target_yaw_tensor),
            )
        )
        inside_goal = (distance <= self._position_tolerance) & (heading_error <= self._heading_tolerance_tensor)
        self._success_hold_seconds = torch.where(
            inside_goal,
            self._success_hold_seconds + self.step_dt,
            torch.zeros_like(self._success_hold_seconds),
        )
        self._success_mask = self._success_hold_seconds + 1e-6 >= self._hold_seconds
        truncated = self.episode_length_buf >= self.max_episode_length
        return self._success_mask, truncated

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self._robot.data.joint_vel[env_ids])
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]
        local_position = torch.tensor(self._initial_position, device=self.device).expand(len(env_ids), 3)
        root_position = origins + local_position
        root_quat = torch.tensor(_yaw_to_xyzw(self._initial_yaw), device=self.device).expand(len(env_ids), 4)
        root_velocity = torch.zeros((len(env_ids), 6), device=self.device)
        self._robot.write_root_pose_to_sim(torch.cat((root_position, root_quat), dim=-1), env_ids=env_ids)
        self._robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            local_position[:, :2] - self._target_position,
            dim=-1,
        )
        initial_yaw = torch.full_like(self._target_yaw_tensor[env_ids], self._initial_yaw)
        initial_heading_delta = initial_yaw - self._target_yaw_tensor[env_ids]
        initial_heading_error = torch.atan2(
            torch.sin(initial_heading_delta),
            torch.cos(initial_heading_delta),
        ).abs()
        self._previous_heading_error[env_ids] = initial_heading_error
        self._success_hold_seconds[env_ids] = 0.0
        self._success_mask[env_ids] = False
        self._previous_wheel_linear_speeds[env_ids] = 0.0
        self._wheel_linear_targets[env_ids] = 0.0
        super()._reset_idx(env_ids)


def make_navigation_task_env(
    task_spec_path: str | Path,
    device: str = "cuda:0",
    num_envs: int = 1,
    scene_spec_path: str | Path | None = None,
    seed: int | None = None,
    physics_hz: float = 120.0,
    control_hz: float = 30.0,
    render_hz: float = 30.0,
    camera_frequencies_hz: Mapping[str, float] | None = None,
) -> NavigationTaskEnvironment:
    return NavigationTaskEnvironment(
        task_spec_path=task_spec_path,
        device=device,
        num_envs=num_envs,
        scene_spec_path=scene_spec_path,
        seed=seed,
        physics_hz=physics_hz,
        control_hz=control_hz,
        render_hz=render_hz,
        camera_frequencies_hz=camera_frequencies_hz,
    )


def _criterion(task_spec, supported_types: set[str]):
    for criterion in task_spec.success_criteria:
        if criterion.criterion_type in supported_types:
            return criterion
    raise ValueError(
        f"task {task_spec.task_id!r} must define one of these success criteria: "
        f"{', '.join(sorted(supported_types))}"
    )


def _yaw_from_xyzw(rotation: tuple[float, float, float, float]) -> float:
    x, y, z, w = rotation
    return torch.atan2(
        torch.tensor(2.0 * (w * z + x * y)),
        torch.tensor(1.0 - 2.0 * (y * y + z * z)),
    ).item()


def _yaw_to_xyzw(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = torch.tensor(yaw / 2.0)
    return (0.0, 0.0, torch.sin(half_yaw).item(), torch.cos(half_yaw).item())


def _quat_xyzw_to_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _as_torch_tensor(quaternion)
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _as_torch_tensor(value) -> torch.Tensor:
    """Use explicit Torch views for Isaac Lab 3.x ProxyArray root-state values."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    raise TypeError("Isaac Lab state value must be a Torch tensor or expose a .torch view")
