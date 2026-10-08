#!/usr/bin/env python3
"""Diagnostic script to inspect ArticulationRoot API behavior with fix_root_link.

Spawns the canonical MRS_ROBOT_CFG USD twice:
  - /World/envs/env_0/Mobile with fix_root_link=None (default floating base)
  - /World/envs/env_1/Fixed with fix_root_link=True (fixed base)

Then enumerates ArticulationRootAPI prims under each target and validates
expected root counts: exactly one for Mobile (Geometry/base_link) and exactly
one for Fixed (Geometry).
"""

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


def main(argv: list[str] | None = None) -> None:
    _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True

    app_launcher = AppLauncher(args)

    sim_context = None
    try:
        import omni.usd
        from pxr import UsdPhysics
        import isaaclab.sim as sim_utils
        from isaaclab.sim import SimulationCfg, SimulationContext

        print("[DEBUG] About to import MRS_ROBOT_CFG", flush=True)
        from mrs_robot_lab.assets.mrs_robot_cfg import MRS_ROBOT_CFG
        print("[DEBUG] MRS_ROBOT_CFG imported successfully", flush=True)

        # 初始化 SimulationContext 以确保 USD stage 可用
        print("[DEBUG] Initializing SimulationContext", flush=True)
        sim_cfg = SimulationCfg(dt=1/120, device=args.device)
        sim_context = SimulationContext(sim_cfg)
        print("[DEBUG] SimulationContext initialized successfully", flush=True)

        # 准备两个生成配置
        mobile_spawn_cfg = MRS_ROBOT_CFG.spawn.copy()
        mobile_spawn_cfg.articulation_props.fix_root_link = None

        fixed_spawn_cfg = MRS_ROBOT_CFG.spawn.copy()
        fixed_spawn_cfg.articulation_props.fix_root_link = True

        # 生成 Mobile (fix_root_link=None, 默认浮动基座)
        mobile_target = "/World/envs/env_0/Mobile"
        print(f"[DEBUG] About to spawn Mobile at {mobile_target}", flush=True)
        mobile_spawn_cfg.func(mobile_target, mobile_spawn_cfg)
        print(f"[DEBUG] Mobile spawned successfully", flush=True)

        # 生成 Fixed (fix_root_link=True, 固定基座)
        fixed_target = "/World/envs/env_1/Fixed"
        print(f"[DEBUG] About to spawn Fixed at {fixed_target}", flush=True)
        fixed_spawn_cfg.func(fixed_target, fixed_spawn_cfg)
        print(f"[DEBUG] Fixed spawned successfully", flush=True)

        # 获取当前 stage
        print("[DEBUG] About to get stage", flush=True)
        stage = omni.usd.get_context().get_stage()
        print("[DEBUG] Stage retrieved successfully", flush=True)
        from mrs_robot_lab.environments.learning.dual_arm_box_task import (
            _CONTACT_LINKS,
            _activate_gripper_contact_reports,
            _gripper_link_prim_path_expression,
            _migrate_newton_articulation_root_api_to_parent,
        )

        fixed_prim = stage.GetPrimAtPath(fixed_target)
        migrated_count = _migrate_newton_articulation_root_api_to_parent(fixed_prim)
        if migrated_count != 1:
            raise RuntimeError(f"Expected to migrate one Newton articulation root, migrated {migrated_count}")
        activated_count = _activate_gripper_contact_reports(fixed_prim, stage)
        if activated_count != 4:
            raise RuntimeError(f"Expected to activate four gripper contact reporters, activated {activated_count}")

        print("\n=== Inspecting finger contact reporters ===")
        finger_names = {
            "openarmx_left_left_finger",
            "openarmx_left_right_finger",
            "openarmx_right_left_finger",
            "openarmx_right_right_finger",
        }
        pending_prims = [fixed_prim]
        finger_reporters = {}
        while pending_prims:
            prim = pending_prims.pop()
            if prim.GetName() in finger_names:
                applied = prim.GetAppliedSchemas()
                has_reporter = "PhysxContactReportAPI" in applied
                finger_reporters[prim.GetName()] = has_reporter
                print(
                    f"  {prim.GetPath()}: rigid_body={bool(UsdPhysics.RigidBodyAPI(prim))}, "
                    f"contact_reporter={has_reporter}, schemas={applied}"
                )
            pending_prims.extend(prim.GetChildren())
        print(f"Finger reporter results: {finger_reporters}")
        print("\n=== Validating configured contact-sensor path expressions ===")
        sensor_match_results = {}
        for sensor_name, link_name in _CONTACT_LINKS.items():
            side = "left" if sensor_name.startswith("left") else "right"
            path_expression = _gripper_link_prim_path_expression(
                "/World/envs/env_.*/Fixed", side, link_name
            )
            matched_paths = sim_utils.find_matching_prim_paths(path_expression, stage)
            sensor_match_results[sensor_name] = matched_paths
            print(f"  {sensor_name}: {path_expression} -> {matched_paths}")
            if len(matched_paths) != 1:
                raise RuntimeError(
                    f"Expected one matched prim for {sensor_name}, found {matched_paths}"
                )
            matched_prim = stage.GetPrimAtPath(matched_paths[0])
            if "PhysxContactReportAPI" not in matched_prim.GetAppliedSchemas():
                raise RuntimeError(f"Matched contact prim {matched_paths[0]} lacks PhysxContactReportAPI")

        def inspect_prim_schemas(prim, indent="  "):
            """详细检查 prim 的 schema、层堆栈和 apiSchemas 元数据"""
            print(f"{indent}=== Prim: {prim.GetPath()} ===")

            # 1. GetAppliedSchemas() - 组合后的 schema 列表
            applied_schemas = prim.GetAppliedSchemas()
            print(f"{indent}GetAppliedSchemas(): {applied_schemas}")

            # 2. GetPrimStack() - 层堆栈中的所有规范
            prim_stack = prim.GetPrimStack()
            print(f"{indent}GetPrimStack() [{len(prim_stack)} specs]:")
            for i, spec in enumerate(prim_stack):
                layer_id = spec.layer.identifier
                spec_path = spec.path
                print(f"{indent}  [{i}] Layer: {layer_id}")
                print(f"{indent}      Spec path: {spec_path}")

                # 3. 检查每个 spec 中 authored 的 apiSchemas 列表操作
                if spec.HasInfo("apiSchemas"):
                    api_schemas_info = spec.GetInfo("apiSchemas")
                    print(f"{indent}      apiSchemas authored: {api_schemas_info}")
                else:
                    print(f"{indent}      apiSchemas authored: (none)")

            # 4. 检查是否有 ArticulationRootAPI
            has_articulation_root = UsdPhysics.ArticulationRootAPI(prim)
            print(f"{indent}Has ArticulationRootAPI: {bool(has_articulation_root)}")
            print()

        def collect_articulation_roots(prim):
            """递归收集所有应用了 ArticulationRootAPI 的后代节点"""
            roots = []
            if UsdPhysics.ArticulationRootAPI(prim):
                roots.append(prim.GetPath())
                inspect_prim_schemas(prim)
            for child in prim.GetAllChildren():
                roots.extend(collect_articulation_roots(child))
            return roots

        # 枚举 Mobile 下的 ArticulationRoot
        print(f"\n=== Inspecting {mobile_target} (fix_root_link=None) ===")
        print("[DEBUG] About to enumerate Mobile roots", flush=True)
        mobile_prim = stage.GetPrimAtPath(mobile_target)
        mobile_roots = collect_articulation_roots(mobile_prim)
        print(f"[DEBUG] Mobile roots enumeration complete: {len(mobile_roots)} found", flush=True)

        # 枚举 Fixed 下的 ArticulationRoot
        print(f"\n=== Inspecting {fixed_target} (fix_root_link=True) ===")
        print("[DEBUG] About to enumerate Fixed roots", flush=True)
        fixed_prim = stage.GetPrimAtPath(fixed_target)
        fixed_roots = collect_articulation_roots(fixed_prim)
        print(f"[DEBUG] Fixed roots enumeration complete: {len(fixed_roots)} found", flush=True)

        # 验证预期结果
        print(f"\n=== Validation ===")
        print(f"Mobile roots found: {len(mobile_roots)}")
        print(f"Fixed roots found: {len(fixed_roots)}")

        errors = []

        if len(mobile_roots) != 1:
            errors.append(
                f"Expected exactly 1 ArticulationRoot for Mobile (Geometry/base_link), "
                f"but found {len(mobile_roots)}: {mobile_roots}"
            )
        elif not str(mobile_roots[0]).endswith("Geometry/base_link"):
            errors.append(
                f"Expected Mobile root at Geometry/base_link, but found: {mobile_roots[0]}"
            )

        if len(fixed_roots) != 1:
            errors.append(
                f"Expected exactly 1 ArticulationRoot for Fixed (Geometry), "
                f"but found {len(fixed_roots)}: {fixed_roots}"
            )
        elif not str(fixed_roots[0]).endswith("Geometry"):
            errors.append(
                f"Expected Fixed root at Geometry, but found: {fixed_roots[0]}"
            )

        geometry_prim = stage.GetPrimAtPath(f"{fixed_target}/Geometry")
        base_link_prim = stage.GetPrimAtPath(f"{fixed_target}/Geometry/base_link")
        if not geometry_prim or "NewtonArticulationRootAPI" not in geometry_prim.GetAppliedSchemas():
            errors.append("Expected NewtonArticulationRootAPI on fixed Geometry parent")
        newton_self_collision = geometry_prim.GetAttribute("newton:selfCollisionEnabled")
        if not newton_self_collision or not newton_self_collision.HasAuthoredValueOpinion():
            errors.append("Expected authored newton:selfCollisionEnabled on fixed Geometry parent")
        elif bool(newton_self_collision.Get()):
            errors.append("Expected transferred newton:selfCollisionEnabled=false on fixed Geometry parent")
        if not base_link_prim or "NewtonArticulationRootAPI" in base_link_prim.GetAppliedSchemas():
            errors.append("Expected NewtonArticulationRootAPI to be removed from fixed base_link")
        elif UsdPhysics.ArticulationRootAPI(base_link_prim):
            errors.append("Expected fixed base_link not to resolve as an ArticulationRootAPI")

        if errors:
            error_message = "\n".join(errors)
            print(f"\n✗ ARTICULATION_ROOT_DIAGNOSTIC_FAILED")
            print(error_message)
            raise RuntimeError(f"ArticulationRoot validation failed:\n{error_message}")

        print("[DEBUG] About to print success message", flush=True)
        print("\n✓ ARTICULATION_ROOT_DIAGNOSTIC_PASSED")
        print(f"  Mobile: {mobile_roots[0]}")
        print(f"  Fixed: {fixed_roots[0]}")

    except BaseException as e:
        print(f"[ERROR] Exception caught: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    finally:
        if sim_context is not None:
            print("[DEBUG] Clearing SimulationContext", flush=True)
            SimulationContext.clear_instance()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
