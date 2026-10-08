"""Resolve an Arena YAML with its own relation solver for the classic Sim GUI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys


def _parse_args() -> argparse.Namespace:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-yaml", required=True)
    parser.add_argument("--task-yaml", default="")
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--source-scene-yaml", required=True)
    parser.add_argument("--source-task-yaml", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mrs-robot-sim-root", default=os.environ.get("MRS_ROBOT_SIM_ROOT", ""))
    parser.add_argument("--x-offset-m", type=float, default=1.05)
    parser.add_argument("--placement-seed", type=int, default=42)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = False
    return args


def _task_scene_path(task_path: Path) -> Path | None:
    import yaml

    source = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError(f"Arena task YAML must be a mapping: {task_path}")
    external_yaml = source.get("external_yaml")
    if external_yaml is None:
        return task_path
    if not isinstance(external_yaml, str) or not external_yaml.strip():
        raise ValueError(f"Arena task external_yaml is invalid: {task_path}")
    return (task_path.parent / external_yaml).resolve(strict=True)


def main() -> int:
    args = _parse_args()
    sim_root = Path(args.mrs_robot_sim_root).expanduser().resolve(strict=True) if args.mrs_robot_sim_root else Path(__file__).resolve().parents[4]
    if not (sim_root / "sim_runtime").is_dir():
        raise ValueError(f"MRS_ROBOT_SIM_ROOT is not a MRS_ROBOT_sim checkout: {sim_root}")
    os.environ["MRS_ROBOT_SIM_ROOT"] = str(sim_root)
    from isaaclab.app import AppLauncher

    app = AppLauncher(args).app
    try:
        import yaml

        from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
        from mrs_arena.compat.scene_manifest import build_openflex_scene_task_manifest
        from mrs_arena.environments.openflex_graph_loader import (
            adapt_graph_spec_for_openflex,
            scene_only_graph_data,
        )
        from mrs_arena.embodiments.openflex import OpenFlexEmbodiment as _OpenFlexEmbodiment
        from mrs_arena.tasks.openflex_no_task import OpenFlexNoTask as _OpenFlexNoTask

        scene_path = Path(args.scene_yaml).expanduser().resolve(strict=True)
        task_path = Path(args.task_yaml).expanduser().resolve(strict=True) if args.task_yaml else None
        if task_path is not None:
            linked_scene = _task_scene_path(task_path)
            if linked_scene is not None and linked_scene != scene_path:
                raise ValueError(
                    f"所选 Arena 任务引用的场景与界面选择不一致：{linked_scene} != {scene_path}"
                )
            graph_spec = ArenaEnvGraphSpec.from_yaml(task_path)
        else:
            scene_data = yaml.safe_load(scene_path.read_text(encoding="utf-8"))
            scene_stem = re.sub(r"[^a-zA-Z0-9_]+", "_", scene_path.stem).strip("_").lower()
            graph_data = scene_only_graph_data(scene_data, scene_stem or args.scene_id)
            graph_spec = ArenaEnvGraphSpec.from_dict(graph_data)
        graph_spec = adapt_graph_spec_for_openflex(graph_spec)
        arena_env = graph_spec.to_arena_env(enable_cameras=False)
        builder = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(
                num_envs=1,
                seed=args.placement_seed,
                placement_seed=args.placement_seed,
                resolve_on_reset=False,
                device="cuda:0",
            ),
        )
        builder._solve_relations()
        manifest = build_openflex_scene_task_manifest(
            arena_env,
            graph_spec,
            scene_id=args.scene_id,
            task_id=args.task_id,
            source_scene_yaml=args.source_scene_yaml,
            source_task_yaml=args.source_task_yaml or None,
            x_offset_m=args.x_offset_m,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"OPENFLEX_ARENA_MANIFEST_READY {output}", flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Arena scene manifest export failed: {error}", file=sys.stderr, flush=True)
        raise
