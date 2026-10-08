#!/usr/bin/env python3
"""Spawn the canonical MRS robot and verify joint, action, and camera interfaces."""

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


def _validate_action_layout() -> None:
    from mrs_robot_lab.actions.openflex_actions import OpenFlexActionsCfg
    from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, ACTION_JOINTS_BY_TERM

    config = OpenFlexActionsCfg()
    if ACTION_DIMENSION != 22:
        raise RuntimeError(f"canonical OpenFlex action contract must have dimension 22, got {ACTION_DIMENSION}")
    for term_name, expected_joints in ACTION_JOINTS_BY_TERM.items():
        actual_joints = tuple(getattr(config, term_name).joint_names)
        if actual_joints != expected_joints:
            raise RuntimeError(
                f"{term_name} differs from canonical contract: expected {expected_joints}, got {actual_joints}"
            )


def main(argv: list[str] | None = None) -> None:
    _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20, help="simulation steps after the joint target probe")
    parser.add_argument("--no-cameras", action="store_true", help="skip RGB/depth initialization")
    parser.add_argument("--validate-actions", action="store_true", help="validate the contract-derived Lab action groups")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    args.enable_cameras = not args.no_cameras

    app_launcher = AppLauncher(args)
    environment = None
    try:
        from mrs_robot_lab.assets.robot_interface import JOINT_POSITION_LIMITS, JOINT_STATE_NAMES
        from mrs_robot_lab.environments.smoke.openflex_smoke import make_openflex_smoke_environment

        if args.validate_actions:
            _validate_action_layout()
        environment = make_openflex_smoke_environment(
            device=args.device,
            enable_cameras=not args.no_cameras,
        )
        observations = environment.reset()
        if not _finite(observations):
            raise RuntimeError("reset returned non-finite robot or sensor observations")
        if tuple(environment.joint_names) != tuple(JOINT_STATE_NAMES):
            raise RuntimeError("smoke environment joint state does not match the canonical contract ordering")

        probe_joint = JOINT_STATE_NAMES[3]
        initial_position = float(observations["joint_position"][0, 3].item())
        low, high = JOINT_POSITION_LIMITS[probe_joint]
        probe_target = initial_position + 0.01
        if probe_target > high:
            probe_target = initial_position - 0.01
        probe_target = min(max(probe_target, low), high)
        if abs(probe_target - initial_position) < 1e-6:
            raise RuntimeError(f"no safe 0.01-rad joint probe is available for {probe_joint}")
        environment.set_joint_positions({probe_joint: probe_target})
        for _ in range(args.steps):
            observations = environment.step()
            if not _finite(observations):
                raise RuntimeError("simulation step returned non-finite robot or sensor observations")

        camera_names = tuple(observations.get("cameras", {}))
        if not args.no_cameras:
            if not camera_names:
                raise RuntimeError("camera smoke was requested but no camera interfaces were created")
            for name, outputs in observations["cameras"].items():
                if "rgb" not in outputs or "distance_to_image_plane" not in outputs:
                    raise RuntimeError(f"camera {name} did not provide RGB and depth outputs")
        print(
            "MRS_ROBOT_LAB_SMOKE_OK "
            f"joints={len(environment.joint_names)} probe={probe_joint}:{probe_target:.4f} "
            f"steps={args.steps} cameras={camera_names} actions_validated={args.validate_actions}",
            flush=True,
        )
    finally:
        if environment is not None:
            environment.close()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
