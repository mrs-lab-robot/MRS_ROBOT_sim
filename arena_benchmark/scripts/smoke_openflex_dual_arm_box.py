#!/usr/bin/env python3
"""Build and idle-step the Arena bilateral transport task without declaring grasp success."""

from __future__ import annotations

import argparse
import sys
from tempfile import TemporaryDirectory


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-steps", type=int, default=5)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.idle_steps < 1:
        parser.error("--idle-steps must be at least 1")

    app_launcher = AppLauncher(args)
    env = None
    failure = None
    smoke_completed = False
    try:
        import torch
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

        from mrs_arena.environments.openflex_smoke import make_openflex_dual_arm_box_environment

        with TemporaryDirectory(prefix="mrs-openflex-box-transport-") as dataset_dir:
            arena_environment = make_openflex_dual_arm_box_environment(dataset_dir)
            task = arena_environment.task
            if task is None or task.__class__.__name__ != "DualArmBoxTransportTask":
                raise RuntimeError("the Arena environment did not provide DualArmBoxTransportTask")

            builder = ArenaEnvBuilder(
                arena_environment,
                ArenaEnvBuilderCfg(
                    num_envs=1,
                    solve_relations=False,
                    disable_fabric=True,
                    device=args.device,
                ),
            )
            env = builder.make_registered()
            unwrapped = env.unwrapped
            active_terms = unwrapped.termination_manager.active_terms
            if "success" not in active_terms or "box_dropped" not in active_terms:
                raise RuntimeError(f"unexpected task termination terms: {active_terms}")

            # 诊断：在 reset 之前验证接触传感器配置
            left_contact_sensor = unwrapped.scene["transport_box_left_contact"]
            right_contact_sensor = unwrapped.scene["transport_box_right_contact"]
            left_filters = left_contact_sensor.cfg.filter_prim_paths_expr
            right_filters = right_contact_sensor.cfg.filter_prim_paths_expr
            print(
                f"Contact sensor filter paths configured:\n"
                f"  left: {left_filters}\n"
                f"  right: {right_filters}",
                flush=True,
            )

            observations, _ = env.reset()

            box = unwrapped.scene["transport_box_rigid_body"]
            box_position = box.data.root_pos_w
            box_position = box_position.torch if hasattr(box_position, "torch") else box_position
            table_position = unwrapped.scene["box_transport_table"].get_world_poses()[0]
            box_bottom_z = float(box_position[0, 2].item()) - 0.05
            tabletop_z = float(table_position[0, 2].item()) + 0.75
            contact_gap_m = abs(box_bottom_z - tabletop_z)
            if contact_gap_m > 0.03:
                raise RuntimeError(
                    f"canonical box does not reset onto the canonical table: gap={contact_gap_m:.4f} m"
                )

            action = torch.zeros(env.action_space.shape, device=unwrapped.device)
            for _ in range(args.idle_steps):
                observations, _, terminated, truncated, _ = env.step(action)
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError("an idle robot and stationary box must not count as success")

            sensor_shapes = {}
            for side in ("left", "right"):
                sensor = unwrapped.scene[f"transport_box_{side}_contact"]
                force = sensor.data.force_matrix_w
                if force is None:
                    raise RuntimeError(f"{side} filtered finger-contact force matrix is unavailable")
                force = force.torch if hasattr(force, "torch") else force
                sensor_shapes[side] = tuple(force.shape)

                if (
                    force.ndim != 4
                    or force.shape[0] != 1
                    or force.shape[1] != 1
                    or force.shape[2] != 2
                    or force.shape[3] != 3
                ):
                    expected = "(1, 1, 2, 3)"
                    actual = tuple(force.shape)
                    raise RuntimeError(
                        f"{side} sensor must resolve one box body against both finger links; "
                        f"expected force matrix shape {expected}, got {actual}; "
                        "verify that filter_prim_paths_expr matches exactly 2 finger prims on Kit stage"
                    )

            policy_obs = observations["policy"]
            if isinstance(policy_obs, dict):
                policy_obs_str = f"dict({', '.join(f'{k}={tuple(v.shape)}' for k, v in policy_obs.items())})"
            else:
                policy_obs_str = str(tuple(policy_obs.shape))

            print(
                "OPENFLEX_DUAL_ARM_BOX_TASK_SMOKE_OK "
                f"task=DualArmBoxTransportTask observations={policy_obs_str} "
                f"contact_gap_m={contact_gap_m:.4f} "
                f"left_force_shape={sensor_shapes['left']} right_force_shape={sensor_shapes['right']} "
                f"idle_steps={args.idle_steps} "
                "physical_grasp_and_success_trajectory=not_tested",
                flush=True,
            )
            smoke_completed = True
    except BaseException as error:
        failure = error
        print(
            f"OPENFLEX_DUAL_ARM_BOX_TASK_SMOKE_FAILED {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
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
    if isinstance(failure, SystemExit) and failure.code in (None, 0):
        failure = RuntimeError("Kit exited before the dual-arm box smoke reported success")
    if failure is None and not smoke_completed:
        failure = RuntimeError("dual-arm box smoke ended without reporting success")
    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
