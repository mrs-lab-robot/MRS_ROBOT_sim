#!/usr/bin/env python3
"""Run an Arena OpenFlex scene and verify RGB/depth camera policy observations."""

from __future__ import annotations

import argparse


CAMERA_NAMES = ("base_d435", "head_d435", "left_wrist_d405", "right_wrist_d405")


def main() -> None:
    print("OPENFLEX_ARENA_CAMERA_SMOKE: entered main", flush=True)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2, help="number of rendered steps")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if not args.enable_cameras:
        parser.error("camera observations require --enable_cameras")

    print(
        "OPENFLEX_ARENA_CAMERA_SMOKE: parsed args "
        f"cameras={args.enable_cameras} device={args.device}",
        flush=True,
    )
    app_launcher = AppLauncher(args)
    print("OPENFLEX_ARENA_CAMERA_SMOKE: Kit initialized", flush=True)
    env = None
    failure = None
    try:
        import torch

        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

        from mrs_arena.environments.openflex_smoke import make_openflex_smoke_environment

        print("OPENFLEX_ARENA_CAMERA_SMOKE: imports complete", flush=True)
        print("OPENFLEX_ARENA_CAMERA_SMOKE: building environment descriptor", flush=True)
        arena_environment = make_openflex_smoke_environment(enable_cameras=True)
        print("OPENFLEX_ARENA_CAMERA_SMOKE: environment descriptor built", flush=True)
        builder = ArenaEnvBuilder(
            arena_environment,
            ArenaEnvBuilderCfg(
                num_envs=1,
                solve_relations=False,
                disable_fabric=True,
                device=args.device,
            ),
        )
        print("OPENFLEX_ARENA_CAMERA_SMOKE: builder configured", flush=True)
        env = builder.make_registered()
        print("OPENFLEX_ARENA_CAMERA_SMOKE: environment registered", flush=True)
        env.reset()
        print("OPENFLEX_ARENA_CAMERA_SMOKE: reset complete", flush=True)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        observations = None
        for _ in range(args.steps):
            observations, _, _, _, _ = env.step(actions)

        camera_observations = observations.get("camera_obs") if observations is not None else None
        if not isinstance(camera_observations, dict):
            raise RuntimeError("Arena policy observations do not contain a camera_obs group")

        expected_keys = {
            f"{camera}_{modality}"
            for camera in CAMERA_NAMES
            for modality in ("rgb", "distance_to_image_plane")
        }
        missing_keys = expected_keys.difference(camera_observations)
        if missing_keys:
            raise RuntimeError(f"camera policy observations are missing: {sorted(missing_keys)}")

        for camera in CAMERA_NAMES:
            rgb = camera_observations[f"{camera}_rgb"]
            depth = camera_observations[f"{camera}_distance_to_image_plane"]
            if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3:
                raise RuntimeError(f"{camera} RGB observation has unexpected shape {tuple(rgb.shape)}")
            if depth.ndim != 4 or depth.shape[0] != 1 or depth.shape[-1] != 1:
                raise RuntimeError(f"{camera} depth observation has unexpected shape {tuple(depth.shape)}")
            if not torch.isfinite(rgb).all() or not torch.count_nonzero(rgb):
                raise RuntimeError(f"{camera} RGB observation is non-finite or empty")
            if not (torch.isfinite(depth) & (depth > 0)).any():
                raise RuntimeError(f"{camera} depth observation contains no positive finite samples")

        print(
            f"OPENFLEX_ARENA_CAMERA_SMOKE_OK cameras={len(CAMERA_NAMES)} "
            f"modalities=rgb, distance_to_image_plane steps={args.steps}",
            flush=True,
        )
    except BaseException as error:
        failure = error
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                if failure is None:
                    failure = error
        try:
            app_launcher.app.close()
        except SystemExit as error:
            if failure is None and error.code not in (None, 0):
                failure = error
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
