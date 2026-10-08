"""Configuration-driven fixed-base bilateral box transport task for Isaac Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.configclass import configclass

from mrs_robot_lab.actions.joint_target_adapter import normalized_to_joint_targets
from mrs_robot_lab.assets.mrs_robot_cfg import MRS_ROBOT_CFG
from mrs_robot_lab.assets.robot_interface import (
    JOINT_POSITION_LIMITS,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
)
from mrs_robot_lab.environments.learning.configuration import load_task_configuration
from mrs_robot_lab.environments.learning.runtime_rates import apply_runtime_rates
from mrs_robot_lab.environments.learning.task_logic_torch import (
    BatchedBoxTransportEvaluator,
    contact_mask_from_force_matrix,
)
from mrs_robot_lab.sensors.camera_cfg import add_capture_cameras


TASK_JOINT_NAMES = (
    *LEFT_ARM_JOINTS,
    *RIGHT_ARM_JOINTS,
    *LEFT_GRIPPER_JOINTS,
    *RIGHT_GRIPPER_JOINTS,
)
TASK_JOINT_LIMITS = tuple(JOINT_POSITION_LIMITS[name] for name in TASK_JOINT_NAMES)
GRIPPER_ACTION_INDICES = (len(LEFT_ARM_JOINTS) + len(RIGHT_ARM_JOINTS), len(TASK_JOINT_NAMES) - 1)
BOX_RIGID_BODY_CHILD = "PickCube"
GRIPPER_CONTACT_FORCE_THRESHOLD_N = 0.1
_CONTACT_LINKS = {
    "left_outer": "openarmx_left_left_finger",
    "left_inner": "openarmx_left_right_finger",
    "right_outer": "openarmx_right_left_finger",
    "right_inner": "openarmx_right_right_finger",
}


def _migrate_newton_articulation_root_api_to_parent(robot_root_prim) -> int:
    """Move Newton articulation-root APIs and authored attributes to fixed roots.

    Isaac Lab's fixed-root transform moves the Physics articulation-root API to
    the parent of the robot's root link. NewtonArticulationRootAPI inherits that
    API, so leaving it on the child would make both prims appear to be roots.
    """
    from pxr import UsdPhysics

    if not robot_root_prim or not robot_root_prim.IsValid():
        raise ValueError("robot_root_prim must be a valid USD prim")

    newton_roots = []
    pending = [robot_root_prim]
    while pending:
        prim = pending.pop()
        if "NewtonArticulationRootAPI" in prim.GetAppliedSchemas():
            newton_roots.append(prim)
        pending.extend(prim.GetChildren())

    migrated_count = 0
    for child_root in newton_roots:
        parent_root = child_root.GetParent()
        if not parent_root or not parent_root.IsValid():
            raise RuntimeError(f"Newton articulation root {child_root.GetPath()} has no valid parent")
        if not UsdPhysics.ArticulationRootAPI(parent_root):
            raise RuntimeError(
                f"Expected fixed Physics articulation root on parent {parent_root.GetPath()} "
                f"of Newton root {child_root.GetPath()}"
            )

        authored_newton_attributes = []
        for attribute in child_root.GetAttributes():
            if attribute.GetName().startswith("newton:") and attribute.HasAuthoredValueOpinion():
                authored_newton_attributes.append(
                    (
                        attribute.GetName(),
                        attribute.GetTypeName(),
                        attribute.IsCustom(),
                        attribute.Get(),
                        tuple(attribute.GetTimeSamples()),
                    )
                )

        if "NewtonArticulationRootAPI" not in parent_root.GetAppliedSchemas():
            if not parent_root.AddAppliedSchema("NewtonArticulationRootAPI"):
                raise RuntimeError(f"Failed to apply NewtonArticulationRootAPI to {parent_root.GetPath()}")

        for name, type_name, is_custom, default_value, time_samples in authored_newton_attributes:
            parent_attribute = parent_root.GetAttribute(name)
            if not parent_attribute:
                parent_attribute = parent_root.CreateAttribute(name, type_name, is_custom)
            if default_value is not None and not parent_attribute.Set(default_value):
                raise RuntimeError(f"Failed to copy authored attribute {name} to {parent_root.GetPath()}")
            for time_sample in time_samples:
                if not parent_attribute.Set(child_root.GetAttribute(name).Get(time_sample), time_sample):
                    raise RuntimeError(
                        f"Failed to copy time sample for {name} at {time_sample} to {parent_root.GetPath()}"
                    )

        if not child_root.RemoveAppliedSchema("NewtonArticulationRootAPI"):
            raise RuntimeError(f"Failed to remove NewtonArticulationRootAPI from {child_root.GetPath()}")
        if UsdPhysics.ArticulationRootAPI(child_root):
            raise RuntimeError(f"Child {child_root.GetPath()} still resolves as an articulation root")
        migrated_count += 1

    return migrated_count


def _activate_gripper_contact_reports(robot_root_prim, stage) -> int:
    """Enable PhysX contact reporting on the four task gripper links.

    Isaac Lab's recursive asset helper stops at the first rigid body in a
    hierarchy. The canonical robot nests additional rigid bodies beneath its
    base link, so task fingertip reporters must be activated by their own prims.
    """
    if not robot_root_prim or not robot_root_prim.IsValid():
        raise ValueError("robot_root_prim must be a valid USD prim")

    expected_names = set(_CONTACT_LINKS.values())
    matches: dict[str, list[str]] = {name: [] for name in expected_names}
    pending = [robot_root_prim]
    while pending:
        prim = pending.pop()
        if prim.GetName() in matches:
            matches[prim.GetName()].append(prim.GetPath().pathString)
        pending.extend(prim.GetChildren())

    invalid_matches = {
        name: paths for name, paths in matches.items() if len(paths) != 1
    }
    if invalid_matches:
        raise RuntimeError(f"Expected one prim for each gripper contact link, got {invalid_matches}")

    for paths in matches.values():
        finger_path = paths[0]
        sim_utils.activate_contact_sensors(finger_path, threshold=0.0, stage=stage)
        finger_prim = stage.GetPrimAtPath(finger_path)
        applied_schemas = finger_prim.GetAppliedSchemas()
        if "PhysxRigidBodyAPI" not in applied_schemas or "PhysxContactReportAPI" not in applied_schemas:
            raise RuntimeError(f"Contact reporter APIs were not applied to gripper link {finger_path}")

    return len(matches)


def _gripper_link_prim_path_expression(robot_root_path_expression: str, side: str, link_name: str) -> str:
    """Build a segment-aware path expression for a canonical fingertip link."""
    if side not in {"left", "right"}:
        raise ValueError(f"unsupported gripper side: {side!r}")
    arm_link_path = "/".join(f"openarmx_{side}_link{index}" for index in range(1, 8))
    return (
        f"{robot_root_path_expression}/Geometry/base_link/lift_carriage_link/"
        f"{arm_link_path}/{link_name}"
    )


@configclass
class DualArmBoxTaskEnvCfg(DirectRLEnvCfg):
    """Policy interface: 47 state features and 16 normalized joint commands."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=3.0)
    episode_length_s: float = 60.0
    decimation: int = 4
    task_spec_path: str | None = None
    scene_spec_path: str | None = None
    num_observations: int = 47
    num_actions: int = 16
    observation_space: int = 47
    action_space: int = 16


class DualArmBoxTaskEnvironment(DirectRLEnv):
    """Move the configured box through grasp, lift, carry, place and release."""

    cfg: DualArmBoxTaskEnvCfg

    def __init__(
        self,
        task_spec_path: str | Path,
        device: str = "cuda:0",
        num_envs: int = 1,
        enable_cameras: bool = False,
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
        self._enable_cameras = enable_cameras
        self._camera_frequencies_hz = dict(camera_frequencies_hz or {})
        criterion = _criterion(self.task_spec)
        params = criterion.params
        self._target_position_local = _vector3(params.get("target_position"), "target_position")
        self._position_tolerance = criterion.tolerance
        self._min_lift_height = _positive_param(params, "min_lift_height")
        self._min_carry_distance = _positive_param(params, "min_carry_distance")
        self._stable_seconds = _positive_param(params, "stable_seconds")
        self._max_linear_speed = float(params.get("max_linear_speed_mps", 0.02))
        if self._max_linear_speed <= 0.0:
            raise ValueError("max_linear_speed_mps must be positive")
        if params.get("require_both_grippers", True) is not True:
            raise ValueError("the baseline transport task requires bilateral gripper contact")
        self._contact_force_threshold = float(
            params.get("gripper_contact_force_threshold_n", GRIPPER_CONTACT_FORCE_THRESHOLD_N)
        )
        if self._contact_force_threshold <= 0.0:
            raise ValueError("gripper_contact_force_threshold_n must be positive")

        self._box_placement = _find_box_placement(self.runtime_config.scene, params.get("object_id"))
        initial_state = self.task_spec.initial_state
        robot_pose = initial_state.get("robot_pose", {})
        scene_position, scene_rotation = self.runtime_config.scene.robot_spawn_pose
        self._robot_initial_position = _vector3(robot_pose.get("position", scene_position), "robot initial position")
        self._robot_initial_yaw = float(robot_pose.get("yaw_rad", _yaw_from_xyzw(scene_rotation)))
        box_pose = initial_state.get("box_pose", {})
        self._box_initial_position = _vector3(
            box_pose.get("position", self._box_placement.position), "box initial position"
        )
        self._box_initial_rotation_xyzw = tuple(
            float(value) for value in box_pose.get("rotation_xyzw", self._box_placement.rotation)
        )
        if len(self._box_initial_rotation_xyzw) != 4:
            raise ValueError("box initial rotation must be an xyzw quaternion")

        self._randomization = dict(self.runtime_config.scene.randomization_config or {})
        self._randomization.update(self.task_spec.randomization_config)
        self._box_xy_randomization = float(
            self._randomization.get("box_xy_m", self._randomization.get("object_xy_m", 0.0))
        )
        self._box_yaw_randomization = float(
            self._randomization.get("box_yaw_rad", self._randomization.get("object_yaw_rad", 0.0))
        )
        if self._box_xy_randomization < 0.0 or self._box_yaw_randomization < 0.0:
            raise ValueError("box randomization ranges cannot be negative")

        cfg = DualArmBoxTaskEnvCfg()
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
            cfg.seed = seed
        super().__init__(cfg, **kwargs)

        self._action_joint_ids, joint_names = self._robot.find_joints(
            list(TASK_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != TASK_JOINT_NAMES:
            raise RuntimeError(f"arm/gripper joint order does not match embodiment contract: {joint_names}")
        self._joint_targets = self._robot.data.default_joint_pos[:, self._action_joint_ids].clone()
        self._task_logic = BatchedBoxTransportEvaluator(
            initial_position=self._box_initial_position,
            target_position=self._target_position_local,
            num_envs=self.num_envs,
            position_tolerance_m=self._position_tolerance,
            min_lift_height_m=self._min_lift_height,
            min_carry_distance_m=self._min_carry_distance,
            stable_seconds=self._stable_seconds,
            max_linear_speed_mps=self._max_linear_speed,
            device=self.device,
        )
        self._target_position_tensor = torch.tensor(self._target_position_local, device=self.device)
        self._left_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._right_contact = torch.zeros_like(self._left_contact)
        self._success_mask = torch.zeros_like(self._left_contact)
        self._last_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_phase = torch.zeros_like(self._last_phase)

    def _setup_scene(self) -> None:
        stage_cfg = sim_utils.UsdFileCfg(usd_path=str(self.runtime_config.base_stage_path))
        stage_cfg.func("/World/Environment", stage_cfg)

        table_placement = next(
            (item for item in self.runtime_config.scene.objects if item.object_type == "table"), None
        )
        if table_placement is None:
            raise ValueError("dual-arm tabletop scene must declare a table object")
        table_path = self.runtime_config.object_asset_paths[table_placement.object_id]
        table_cfg = sim_utils.UsdFileCfg(usd_path=str(table_path), scale=table_placement.scale)
        table_cfg.func(
            f"/World/envs/env_0/{table_placement.object_id}",
            table_cfg,
            translation=table_placement.position,
            orientation=_as_xyzw(table_placement.rotation),
        )

        box_path = self.runtime_config.object_asset_paths[self._box_placement.object_id]
        box_spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(box_path),
            scale=self._box_placement.scale,
            activate_contact_sensors=True,
        )
        box_spawn_cfg.func(
            f"/World/envs/env_0/{self._box_placement.object_id}",
            box_spawn_cfg,
            translation=self._box_initial_position,
            orientation=_as_xyzw(self._box_initial_rotation_xyzw),
        )

        robot_cfg = MRS_ROBOT_CFG.copy()
        robot_cfg.prim_path = f"{self.scene.env_regex_ns}/Robot"
        robot_cfg.init_state.pos = self._robot_initial_position
        robot_cfg.init_state.rot = _yaw_to_xyzw(self._robot_initial_yaw)
        robot_cfg.spawn.articulation_props.fix_root_link = True

        original_spawn = robot_cfg.spawn.func
        if not callable(original_spawn):
            raise RuntimeError("MRS robot USD spawn function is not callable")

        def spawn_fixed_robot_with_newton_root_migration(prim_path, spawn_cfg, *args, **kwargs):
            source_prim = original_spawn(prim_path, spawn_cfg, *args, **kwargs)
            if getattr(spawn_cfg.articulation_props, "fix_root_link", None) is True:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
                robot_paths = sim_utils.find_matching_prim_paths(str(prim_path), stage)
                if not robot_paths and source_prim and source_prim.IsValid():
                    robot_paths = [source_prim.GetPath().pathString]
                if not robot_paths:
                    raise RuntimeError(f"Robot spawn produced no prims matching {prim_path!r}")
                for robot_path in robot_paths:
                    robot_prim = stage.GetPrimAtPath(robot_path)
                    migrated_count = _migrate_newton_articulation_root_api_to_parent(robot_prim)
                    if migrated_count != 1:
                        raise RuntimeError(
                            f"Expected to migrate one Newton articulation root under {robot_path}, "
                            f"migrated {migrated_count}"
                        )
                    _activate_gripper_contact_reports(robot_prim, stage)
            return source_prim

        robot_cfg.spawn.func = spawn_fixed_robot_with_newton_root_migration
        self._robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self._robot
        self.scene.clone_environments(copy_from_source=False)

        box_cfg = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{self._box_placement.object_id}/{BOX_RIGID_BODY_CHILD}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=self._box_initial_position,
                rot=_as_xyzw(self._box_initial_rotation_xyzw),
            ),
        )
        self._box = RigidObject(box_cfg)
        self.scene.rigid_objects["box"] = self._box

        box_filter = f"/World/envs/env_.*/{self._box_placement.object_id}/{BOX_RIGID_BODY_CHILD}"
        self._contact_sensors: dict[str, ContactSensor] = {}
        robot_root_path_expression = f"{self.scene.env_regex_ns}/Robot"
        for sensor_name, link_name in _CONTACT_LINKS.items():
            side = "left" if sensor_name.startswith("left") else "right"
            prim_path = _gripper_link_prim_path_expression(
                robot_root_path_expression, side, link_name
            )
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=prim_path,
                    update_period=0.0,
                    history_length=1,
                    filter_prim_paths_expr=[box_filter],
                    force_threshold=self._contact_force_threshold,
                )
            )
            self._contact_sensors[sensor_name] = sensor
            self.scene.sensors[f"{side}_{sensor_name}"] = sensor
        add_capture_cameras(self.scene, self._camera_frequencies_hz)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, len(TASK_JOINT_NAMES)):
            raise ValueError(
                f"expected normalized bilateral command shape ({self.num_envs}, {len(TASK_JOINT_NAMES)}), "
                f"got {tuple(actions.shape)}"
            )
        self._joint_targets = normalized_to_joint_targets(
            actions,
            TASK_JOINT_LIMITS,
            gripper_indices=GRIPPER_ACTION_INDICES,
        )

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target_index(
            target=self._joint_targets,
            joint_ids=self._action_joint_ids,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos[:, self._action_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._action_joint_ids]
        box_position = _as_torch_tensor(self._box.data.root_pos_w)[:, :3] - self.scene.env_origins
        box_quaternion = _as_torch_tensor(self._box.data.root_quat_w)
        box_velocity = _as_torch_tensor(self._box.data.root_lin_vel_w)[:, :3]
        target_relative = self._target_position_tensor.unsqueeze(0) - box_position
        contacts = torch.stack((self._left_contact.float(), self._right_contact.float()), dim=-1)
        observation = torch.cat(
            (joint_pos, joint_vel, box_position, box_quaternion, box_velocity, target_relative, contacts),
            dim=-1,
        )
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        distance = torch.linalg.vector_norm(
            _as_torch_tensor(self._box.data.root_pos_w)[:, :3] - self.scene.env_origins
            - self._target_position_tensor,
            dim=-1,
        )
        reward_cfg = self.task_spec.reward_config or {}
        phase_reward = (self._last_phase.float() - self._previous_phase.float()) * float(
            reward_cfg.get("phase_progress_reward", 1.0)
        )
        return (
            float(reward_cfg.get("step_penalty", -0.1))
            - distance * float(reward_cfg.get("distance_reward_scale", 0.1))
            + phase_reward
            + self._success_mask.float() * float(reward_cfg.get("success_reward", 100.0))
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._left_contact = self._side_contact_mask("left")
        self._right_contact = self._side_contact_mask("right")
        box_position = _as_torch_tensor(self._box.data.root_pos_w)[:, :3] - self.scene.env_origins
        box_velocity = _as_torch_tensor(self._box.data.root_lin_vel_w)[:, :3]
        phase, success = self._task_logic.update(
            box_position,
            box_velocity,
            self._left_contact,
            self._right_contact,
            dt=self.step_dt,
        )
        self._previous_phase = self._last_phase.clone()
        self._last_phase = phase
        self._success_mask = success
        if hasattr(self, "extras"):
            self.extras["task_phase"] = phase
        truncated = self.episode_length_buf >= self.max_episode_length
        return success, truncated

    def _side_contact_mask(self, side: str) -> torch.Tensor:
        names = (f"{side}_outer", f"{side}_inner")
        masks = []
        for name in names:
            force_matrix = self._contact_sensors[name].data.force_matrix_w
            if force_matrix is None:
                raise RuntimeError(f"filtered force matrix is unavailable for {name} contact sensor")
            masks.append(
                contact_mask_from_force_matrix(
                    force_matrix,
                    force_threshold=self._contact_force_threshold,
                )
            )
        return masks[0] | masks[1]

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(self._robot.data.joint_vel[env_ids])
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]
        robot_local = torch.tensor(self._robot_initial_position, device=self.device).expand(len(env_ids), 3)
        robot_position = origins + robot_local
        robot_quat = torch.tensor(_yaw_to_xyzw(self._robot_initial_yaw), device=self.device).expand(len(env_ids), 4)
        self._robot.write_root_pose_to_sim(
            torch.cat((robot_position, robot_quat), dim=-1),
            env_ids=env_ids,
        )
        self._robot.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=self.device),
            env_ids=env_ids,
        )

        box_position = torch.tensor(self._box_initial_position, device=self.device).expand(len(env_ids), 3).clone()
        if self._box_xy_randomization > 0.0:
            box_position[:, :2] += (
                torch.rand((len(env_ids), 2), device=self.device) * 2.0 - 1.0
            ) * self._box_xy_randomization
        box_yaw = torch.full((len(env_ids),), _yaw_from_xyzw(self._box_initial_rotation_xyzw), device=self.device)
        if self._box_yaw_randomization > 0.0:
            box_yaw += (torch.rand((len(env_ids),), device=self.device) * 2.0 - 1.0) * self._box_yaw_randomization
        box_quat = _yaw_tensor_to_xyzw(box_yaw)
        self._box.write_root_pose_to_sim(torch.cat((origins + box_position, box_quat), dim=-1), env_ids=env_ids)
        self._box.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=self.device), env_ids=env_ids)

        self._task_logic.reset(env_ids, initial_position=box_position)
        self._left_contact[env_ids] = False
        self._right_contact[env_ids] = False
        self._success_mask[env_ids] = False
        self._last_phase[env_ids] = 0
        self._previous_phase[env_ids] = 0
        self._joint_targets[env_ids] = self._robot.data.default_joint_pos[env_ids][:, self._action_joint_ids]
        super()._reset_idx(env_ids)


def make_dual_arm_box_task_env(
    task_spec_path: str | Path,
    device: str = "cuda:0",
    num_envs: int = 1,
    enable_cameras: bool = False,
    scene_spec_path: str | Path | None = None,
    seed: int | None = None,
    physics_hz: float = 120.0,
    control_hz: float = 30.0,
    render_hz: float = 30.0,
    camera_frequencies_hz: Mapping[str, float] | None = None,
) -> DualArmBoxTaskEnvironment:
    return DualArmBoxTaskEnvironment(
        task_spec_path=task_spec_path,
        device=device,
        num_envs=num_envs,
        enable_cameras=enable_cameras,
        scene_spec_path=scene_spec_path,
        seed=seed,
        physics_hz=physics_hz,
        control_hz=control_hz,
        render_hz=render_hz,
        camera_frequencies_hz=camera_frequencies_hz,
    )


def _criterion(task_spec):
    for criterion in task_spec.success_criteria:
        if criterion.criterion_type in {"bilateral_box_transport", "box_at_target"}:
            required = {
                "target_position",
                "min_lift_height",
                "min_carry_distance",
                "stable_seconds",
            }
            missing = required - criterion.params.keys()
            if missing:
                raise ValueError(f"box transport criterion is missing parameters: {', '.join(sorted(missing))}")
            return criterion
    raise ValueError("dual-arm task must define a bilateral_box_transport success criterion")


def _find_box_placement(scene_spec, object_id: str | None):
    candidates = [item for item in scene_spec.objects if item.object_type in {"rigid_box", "box"}]
    if object_id:
        candidates = [item for item in candidates if item.object_id == object_id]
    if len(candidates) != 1:
        raise ValueError(f"scene must define exactly one configured transport box, got {len(candidates)}")
    return candidates[0]


def _vector3(value, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain three numbers")
    result = tuple(float(component) for component in value)
    if not all(torch.isfinite(torch.tensor(component)).item() for component in result):
        raise ValueError(f"{name} must contain finite numbers")
    return result


def _positive_param(params: dict, name: str) -> float:
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _yaw_from_xyzw(rotation: tuple[float, float, float, float]) -> float:
    x, y, z, w = rotation
    return torch.atan2(
        torch.tensor(2.0 * (w * z + x * y)),
        torch.tensor(1.0 - 2.0 * (y * y + z * z)),
    ).item()


def _as_xyzw(rotation: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if len(rotation) != 4:
        raise ValueError("rotation must contain exactly four xyzw quaternion components")
    return tuple(float(component) for component in rotation)


def _as_torch_tensor(value) -> torch.Tensor:
    """Use explicit Torch views for Isaac Lab 3.x ProxyArray state values."""
    if isinstance(value, torch.Tensor):
        return value
    tensor = getattr(value, "torch", None)
    if isinstance(tensor, torch.Tensor):
        return tensor
    raise TypeError("Isaac Lab state value must be a Torch tensor or expose a .torch view")


def _yaw_to_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return _yaw_tensor_to_xyzw(torch.tensor([yaw], dtype=torch.float32))[0].tolist()


def _yaw_tensor_to_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw / 2.0
    zeros = torch.zeros_like(half)
    return torch.stack((zeros, zeros, torch.sin(half), torch.cos(half)), dim=-1)
