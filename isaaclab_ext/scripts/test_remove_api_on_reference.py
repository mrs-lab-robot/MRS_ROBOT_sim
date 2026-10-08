#!/usr/bin/env python3
"""测试在 USD 引用的组合层级上调用 RemoveAPI 的行为。

该脚本创建一个简单的场景来重现问题：
1. 创建一个引用了 openflex_robot.usda 的场景
2. 在 session 层上对父节点应用 ArticulationRootAPI（模拟 modify_articulation_root_properties 的行为）
3. 尝试从子节点移除 ArticulationRootAPI
4. 检查移除是否成功，以及层堆栈中发生了什么
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
    ):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    return repo_root


def main(argv: list[str] | None = None) -> None:
    repo_root = _configure_project_paths()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    args.headless = True

    app_launcher = AppLauncher(args)

    try:
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
        from isaaclab.sim import SimulationCfg, SimulationContext

        # 初始化 SimulationContext
        print("[INFO] Initializing SimulationContext...")
        sim_cfg = SimulationCfg(dt=1/120, device=args.device)
        sim_context = SimulationContext(sim_cfg)
        stage = omni.usd.get_context().get_stage()

        # 创建一个简单的测试场景
        robot_usd_path = repo_root / "sim_runtime/assets/robots/openflex_robot.usda"
        test_prim_path = "/World/TestRobot"

        print(f"\n[TEST 1] 创建引用到: {robot_usd_path}")
        # 创建父 Xform
        parent_prim = stage.DefinePrim(f"{test_prim_path}/Geometry", "Xform")

        # 添加引用（引用 base_link，它已经有 ArticulationRootAPI）
        child_prim_path = f"{test_prim_path}/Geometry/base_link"
        references = parent_prim.GetReferences()
        references.AddReference(
            assetPath=str(robot_usd_path),
            primPath="/openarmx_integrated/Geometry/base_link"
        )

        print(f"\n[TEST 2] 检查引用后的初始状态")
        child_prim = stage.GetPrimAtPath(child_prim_path)

        def inspect_prim(prim, label):
            """检查 prim 的 schema 状态"""
            print(f"\n  === {label}: {prim.GetPath()} ===")
            applied_schemas = prim.GetAppliedSchemas()
            print(f"  GetAppliedSchemas(): {applied_schemas}")

            has_api = bool(UsdPhysics.ArticulationRootAPI(prim))
            print(f"  Has ArticulationRootAPI: {has_api}")

            # 检查层堆栈
            prim_stack = prim.GetPrimStack()
            print(f"  PrimStack ({len(prim_stack)} specs):")
            for i, spec in enumerate(prim_stack):
                layer_id = Path(spec.layer.identifier).name if spec.layer.identifier else "(anonymous)"
                has_api_schemas = spec.HasInfo("apiSchemas")
                if has_api_schemas:
                    api_info = spec.GetInfo("apiSchemas")
                    print(f"    [{i}] {layer_id}: apiSchemas = {api_info}")
                else:
                    print(f"    [{i}] {layer_id}: (no apiSchemas)")

        inspect_prim(child_prim, "Child (引用后)")

        print(f"\n[TEST 3] 在父节点上应用 ArticulationRootAPI（模拟 fix_root_link=True）")
        UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
        parent_prim.AddAppliedSchema("PhysxArticulationAPI")

        inspect_prim(parent_prim, "Parent (应用 API 后)")
        inspect_prim(child_prim, "Child (父节点应用 API 后)")

        print(f"\n[TEST 4] 尝试从子节点移除 ArticulationRootAPI")
        print(f"  调用: child_prim.RemoveAppliedSchema('PhysxArticulationAPI')")
        child_prim.RemoveAppliedSchema("PhysxArticulationAPI")

        print(f"  调用: child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)")
        child_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

        inspect_prim(child_prim, "Child (RemoveAPI 后)")

        # 最终验证
        still_has_api = bool(UsdPhysics.ArticulationRootAPI(child_prim))
        print(f"\n[RESULT] 子节点在 RemoveAPI 后仍然有 ArticulationRootAPI: {still_has_api}")

        if still_has_api:
            print("\n✗ BUG 确认: RemoveAPI 无法移除来自引用层的 API")
            print("  原因: RemoveAPI 只在 session 层添加删除意见，但无法覆盖引用层的强意见")
        else:
            print("\n✓ RemoveAPI 成功移除 API")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
    finally:
        if 'sim_context' in locals():
            SimulationContext.clear_instance()
        app_launcher.app.close()


if __name__ == "__main__":
    main()
