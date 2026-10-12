"""Headless KitVisualizer probe for PhysX -> render sync and synthetic VR actions.

Run via the project Python with ``TEST_DISABLE_FABRIC=1`` or ``0``. It keeps the
KitVisualizer/app-update path active without requiring an X display, and reports
PhysX joint motion, USD link transforms, and rendered camera deltas.
"""
from __future__ import annotations

import os
import time

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, device="cuda:0", enable_cameras=True).app
# Headless normally suppresses all visualizers. Force the same KitVisualizer
# pre-render forward/app-update path used by the interactive GUI.
AppLauncher.sync_visualizer_cli_settings_to_carb({
    "visualizer": ["kit"],
    "visualizer_explicit": True,
    "visualizer_disable_all": False,
})

import numpy as np
import torch
from pxr import Usd, UsdGeom
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_physx.physics.physx_manager import PhysxManager
from mrs_arena.environments.openflex_smoke import make_openflex_smoke_environment
from mrs_robot_lab.adapters.synthetic_vr import SyntheticVrSource
from mrs_robot_lab.adapters.teleop_adapter import JOINT_STATE_NAMES, action_from_command
from mrs_teleoperation.safety import CommandWatchdog


def _main() -> None:
    disable_fabric = os.environ.get("TEST_DISABLE_FABRIC", "1") == "1"
    env = None
    try:
        arena_env = make_openflex_smoke_environment(enable_cameras=True)
        env = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=disable_fabric,
                device="cuda:0",
            ),
        ).make_registered()
        env.reset()
        sim = env.unwrapped.sim
        robot = env.unwrapped.scene["robot"]
        print(
            "PROBE CONFIG"
            f" disable_fabric={disable_fabric} use_fabric={sim.cfg.use_fabric}"
            f" updateToUsd={sim.get_setting('/physics/updateToUsd')}"
            f" fabric_setting={sim.get_setting('/isaaclab/fabric_enabled')}"
            f" render={sim.is_rendering} gui={sim.has_gui}"
            f" visualizers={[type(v).__name__ for v in sim._visualizers]}"
            f" fabric_loaded={PhysxManager._fabric is not None}"
            f" forward_required={sim._should_forward_before_visualizer_update()}",
            flush=True,
        )
        stage = sim.stage
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")

        def link_transforms() -> dict[str, tuple[float, float, float]]:
            values = {}
            for prim in Usd.PrimRange(robot_prim):
                if "link" in prim.GetName().lower():
                    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    position = transform.ExtractTranslation()
                    values[prim.GetName()] = tuple(round(float(x), 5) for x in position)
            return values

        def joint_positions() -> dict[str, float]:
            names = tuple(robot.joint_names)
            values = robot.data.joint_pos[0].detach().to(device="cpu", dtype=torch.float32).tolist()
            by_name = dict(zip(names, values, strict=True))
            return {name: float(by_name[name]) for name in JOINT_STATE_NAMES}

        start_xforms = link_transforms()
        start_joints = joint_positions()
        source = SyntheticVrSource()
        source.start(start_joints)
        watchdog = CommandWatchdog()
        step_dt = float(env.unwrapped.step_dt)
        camera_start = {}
        camera_end = {}
        initial_generation = sim.render_generation
        step_count = int(os.environ.get("PROBE_STEPS", "120"))
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        for index in range(step_count):
            now = time.monotonic()
            frame = source.next_frame(
                seq=index + 1,
                source_time_ns=int(index * step_dt * 1e9),
            )
            watchdog.accept(frame, now=now)
            values = action_from_command(
                watchdog.latest,
                joint_positions(),
                step_dt=step_dt,
                motion_enabled=watchdog.motion_enabled(now),
            )
            actions.zero_()
            row = actions[0] if actions.ndim == 2 else actions
            row.copy_(torch.tensor(values, device=row.device, dtype=row.dtype))
            observations, *_ = env.step(actions)
            if index in (0, step_count - 1):
                camera_obs = observations.get("camera_obs", {})
                snapshot = {
                    name: image[0].detach().to(device="cpu").numpy().copy()
                    for name, image in camera_obs.items()
                    if name.endswith("_rgb")
                }
                if index == 0:
                    camera_start = snapshot
                else:
                    camera_end = snapshot

        end_joints = joint_positions()
        end_xforms = link_transforms()
        moved = [name for name in start_xforms if start_xforms[name] != end_xforms.get(name)]
        print(
            "PROBE RESULT"
            f" steps={sim.get_physics_step_count()}"
            f" render_generation={initial_generation}->{sim.render_generation}"
            f" head_yaw={start_joints['openarmx_head_yaw_joint']:.4f}->{end_joints['openarmx_head_yaw_joint']:.4f}"
            f" left_j1={start_joints['openarmx_left_joint1']:.4f}->{end_joints['openarmx_left_joint1']:.4f}"
            f" usd_links_moved={len(moved)}/{len(start_xforms)} names={moved[:8]}",
            flush=True,
        )
        for name in sorted(camera_start):
            if name in camera_end:
                delta = np.abs(camera_start[name].astype(np.int16) - camera_end[name].astype(np.int16))
                print(
                    f"PROBE CAMERA {name} mean_abs_diff={float(delta.mean()):.4f}"
                    f" max_abs_diff={int(delta.max())}",
                    flush=True,
                )
        print("PROBE DONE", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


_main()
