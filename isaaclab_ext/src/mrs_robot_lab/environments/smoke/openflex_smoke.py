"""Direct single-robot Isaac Lab smoke harness with no benchmark dependency."""

from __future__ import annotations

import math
from typing import Mapping


class OpenFlexSmokeEnvironment:
    """Spawn the canonical robot, expose joint state/targets, and optionally read cameras."""

    def __init__(self, *, device: str = "cuda:0", enable_cameras: bool = True) -> None:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationContext

        from mrs_robot_lab.assets.mrs_robot_cfg import JOINT_STATE_NAMES, MRS_ROBOT_CFG
        from mrs_robot_lab.assets.robot_interface import (
            JOINT_POSITION_LIMITS,
            SWERVE_CONFIG,
        )
        self._torch = torch
        self._dt = 1.0 / 120.0
        self._joint_state_names = JOINT_STATE_NAMES
        self._position_limits = JOINT_POSITION_LIMITS
        self._wheel_velocity_limits = {
            name: SWERVE_CONFIG.max_wheel_speed / SWERVE_CONFIG.wheel_radius
            for name in SWERVE_CONFIG.wheel_joint_names
        }
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=self._dt, device=device))
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

        robot_cfg = MRS_ROBOT_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        self.robot = Articulation(robot_cfg)

        self._cameras = {}
        if enable_cameras:
            from isaaclab.sensors import Camera

            from mrs_robot_lab.sensors.camera_cfg import build_camera_cfgs

            self._cameras = {
                name: Camera(cfg)
                for name, cfg in build_camera_cfgs(robot_prim_path="/World/Robot").items()
            }

        self.sim.reset()
        self._joint_ids, joint_names = self.robot.find_joints(list(JOINT_STATE_NAMES), preserve_order=True)
        if tuple(joint_names) != tuple(JOINT_STATE_NAMES):
            raise RuntimeError("spawned articulation joint order does not match the canonical contract")
        self.robot.reset()
        self.robot.update(self._dt)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self._joint_state_names

    @property
    def step_dt(self) -> float:
        """Return the fixed simulation step used by the no-task environment."""

        return self._dt

    def reset(self) -> dict:
        """Reset the simulation articulation and return joint/camera observations."""

        self.sim.reset()
        self.robot.reset()
        self.robot.update(self._dt)
        return self.observations()

    def set_joint_positions(self, positions: Mapping[str, float]) -> None:
        """Set position targets for named joints from the robot contract."""

        from mrs_robot_lab.runners.joint_teleop import validate_joint_targets

        targets = validate_joint_targets(positions, self._position_limits)
        if not targets:
            return
        names = tuple(targets)
        joint_ids, found_names = self.robot.find_joints(list(names), preserve_order=True)
        if tuple(found_names) != names:
            raise ValueError(f"articulation does not expose requested joints: {names}")
        targets = self._torch.tensor(
            [[targets[name] for name in names]],
            dtype=self._torch.float32,
            device=self.robot.device,
        )
        self.robot.set_joint_position_target(targets, joint_ids=joint_ids)

    def set_joint_velocities(self, velocities: Mapping[str, float]) -> None:
        """Set bounded velocity targets for the four canonical swerve wheels."""

        unknown = set(velocities).difference(self._wheel_velocity_limits)
        if unknown:
            raise ValueError(f"velocity targets are only supported for swerve wheels: {sorted(unknown)}")
        for name, value in velocities.items():
            limit = self._wheel_velocity_limits[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"wheel velocity target for {name!r} must be a finite number")
            if abs(float(value)) > limit:
                raise ValueError(f"wheel velocity target for {name!r} exceeds the configured limit {limit}")
        if not velocities:
            return
        names = tuple(velocities)
        joint_ids, found_names = self.robot.find_joints(list(names), preserve_order=True)
        if tuple(found_names) != names:
            raise ValueError(f"articulation does not expose requested wheel joints: {names}")
        targets = self._torch.tensor(
            [[float(velocities[name]) for name in names]],
            dtype=self.robot.data.joint_vel.dtype,
            device=self.robot.device,
        )
        self.robot.set_joint_velocity_target(targets, joint_ids=joint_ids)

    def step(self) -> dict:
        """Advance one simulation step and return the latest observations."""

        self.robot.write_data_to_sim()
        self.sim.step(render=True)
        self.robot.update(self._dt)
        return self.observations()

    def observations(self) -> dict:
        """Return contract-ordered joint state and available RGB/depth camera samples."""

        joint_position = self.robot.data.joint_pos[:, self._joint_ids]
        joint_velocity = self.robot.data.joint_vel[:, self._joint_ids]
        observation = {
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
        }
        if self._cameras:
            camera_output = {}
            for name, camera in self._cameras.items():
                camera.update(self._dt)
                camera_output[name] = camera.data.output
            observation["cameras"] = camera_output
        return observation

    def close(self) -> None:
        """Release Python-side sensor and articulation references."""

        self._cameras.clear()
        self.robot = None


def make_openflex_smoke_environment(
    *, device: str = "cuda:0", enable_cameras: bool = True
) -> OpenFlexSmokeEnvironment:
    """Construct a single-robot smoke environment using Isaac Lab APIs only."""

    return OpenFlexSmokeEnvironment(device=device, enable_cameras=enable_cameras)
