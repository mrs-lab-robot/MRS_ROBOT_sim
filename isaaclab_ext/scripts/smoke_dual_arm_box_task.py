#!/usr/bin/env python3
"""Isaac Lab Kit smoke test for the dual-arm box transport task."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _configure_project_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(repo_root))
    for source_root in (
        repo_root / "isaaclab_ext/src",
        repo_root / "sim_runtime/teleoperation/src",
        repo_root / "sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract",
    ):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    return repo_root


def _finite(value) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return True


def main(argv: list[str] | None = None) -> None:
    repo_root = _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=repo_root / "sim_runtime/config/tasks/dual_arm_box_transport.yaml",
        help="path to task specification YAML",
    )
    parser.add_argument(
        "--scene-spec",
        type=Path,
        default=repo_root / "sim_runtime/config/scenes/dual_arm_box_tabletop.yaml",
        help="path to scene specification YAML",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="number of zero-action simulation steps to run",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args(argv)

    if args.steps < 1:
        parser.error("--steps must be at least 1")

    task_path = args.task_spec.expanduser().resolve(strict=True)
    scene_path = args.scene_spec.expanduser().resolve(strict=True)

    app_launcher = AppLauncher(args)
    environment = None
    try:
        print("[SMOKE] Starting environment construction", flush=True)
        import torch
        from mrs_robot_lab.environments.learning.dual_arm_box_task import make_dual_arm_box_task_env

        print("[SMOKE] Imports complete, calling make_dual_arm_box_task_env", flush=True)
        environment = make_dual_arm_box_task_env(
            task_spec_path=task_path,
            scene_spec_path=scene_path,
            device=args.device,
            num_envs=1,
            enable_cameras=False,
        )
        print("[SMOKE] Environment constructed successfully", flush=True)

        print("[SMOKE] Calling environment.reset()", flush=True)
        observations, extras = environment.reset()
        print("[SMOKE] Environment reset complete", flush=True)

        if not isinstance(observations, dict) or "policy" not in observations:
            raise RuntimeError("reset must return a dict with a 'policy' key")

        policy_obs = observations["policy"]
        expected_shape = (1, 47)
        if policy_obs.shape != expected_shape:
            raise RuntimeError(
                f"policy observation shape must be {expected_shape}, got {tuple(policy_obs.shape)}"
            )
        if not _finite(policy_obs):
            raise RuntimeError("reset returned non-finite policy observations")
        print("[SMOKE] Reset observations validated", flush=True)

        zero_actions = torch.zeros((1, 16), device=args.device)
        print(f"[SMOKE] Starting {args.steps} simulation steps", flush=True)
        for step_index in range(args.steps):
            print(f"[SMOKE] Step {step_index + 1}/{args.steps} starting", flush=True)
            observations, rewards, *_ = environment.step(zero_actions)
            if not _finite(observations):
                raise RuntimeError(f"step {step_index} returned non-finite observations")
            if not _finite(rewards):
                raise RuntimeError(f"step {step_index} returned non-finite rewards")
            print(f"[SMOKE] Step {step_index + 1}/{args.steps} complete", flush=True)
        print("[SMOKE] All simulation steps complete", flush=True)

        print("[SMOKE] Starting contact sensor validation", flush=True)
        contact_sensor_names = ["left_outer", "left_inner", "right_outer", "right_inner"]
        for name in contact_sensor_names:
            print(f"[SMOKE] Validating contact sensor: {name}", flush=True)
            if name not in environment._contact_sensors:
                raise RuntimeError(f"contact sensor {name!r} is not available in the environment")
            sensor = environment._contact_sensors[name]
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                raise RuntimeError(f"contact sensor {name!r} has None force_matrix_w after stepping")
            if not _finite(force_matrix):
                raise RuntimeError(f"contact sensor {name!r} has non-finite force_matrix_w")
            print(f"[SMOKE] Contact sensor {name} validated", flush=True)
        print("[SMOKE] All contact sensors validated", flush=True)

        print(
            "MRS_DUAL_ARM_BOX_KIT_SMOKE_OK "
            f"steps={args.steps} obs_shape={tuple(policy_obs.shape)} "
            f"action_shape={tuple(zero_actions.shape)} contact_sensors={len(contact_sensor_names)}",
            flush=True,
        )
    except BaseException as exc:
        print(f"[SMOKE] EXCEPTION CAUGHT: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        print("[SMOKE] Entering cleanup (finally block)", flush=True)
        if environment is not None:
            environment.close()
        app_launcher.app.close()
        print("[SMOKE] Cleanup complete", flush=True)


if __name__ == "__main__":
    main()
