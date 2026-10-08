#!/usr/bin/env python3
"""Convert a saved OpenFlex Arena teleoperation HDF5 episode to LeRobot."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _configure_project_imports() -> None:
    sim_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(sim_root))
    for source in (
        sim_root / "isaaclab_ext" / "src",
        sim_root / "sim_runtime" / "teleoperation" / "src",
    ):
        if source.is_dir():
            sys.path.insert(0, str(source))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="saved Arena HDF5 Episode")
    parser.add_argument("--prepared-output", help="write an HDF5-free staging directory")
    parser.add_argument("--prepared-dir", help="read a prepared HDF5-free staging directory")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-root", help="LeRobot dataset root")
    parser.add_argument("--repo-id", help="LeRobot dataset ID")
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--task", required=True, help="LeRobot task description")
    args = parser.parse_args()

    _configure_project_imports()
    if args.prepare_only:
        if not args.source or not args.prepared_output or args.prepared_dir:
            parser.error("--prepare-only requires --source and --prepared-output")
        from mrs_robot_lab.recorders.lerobot_export import prepare_arena_hdf5_for_lerobot

        destination = prepare_arena_hdf5_for_lerobot(
            args.source,
            args.prepared_output,
            fps=args.fps,
            task=args.task,
        )
        print(f"ARENA_LEROBOT_PREPARE_OK path={destination}", flush=True)
        return 0

    if not args.output_root or not args.repo_id:
        parser.error("dataset export requires --output-root and --repo-id")
    if args.prepared_dir:
        if args.source or args.prepared_output:
            parser.error("--prepared-dir cannot be combined with HDF5 input options")
        from mrs_robot_lab.recorders.lerobot_export import export_prepared_episode_to_lerobot

        destination = export_prepared_episode_to_lerobot(
            args.prepared_dir,
            args.output_root,
            args.repo_id,
            fps=args.fps,
            task=args.task,
        )
    else:
        if not args.source or args.prepared_output:
            parser.error("direct export requires --source and cannot use --prepared-output")
        from mrs_robot_lab.recorders.lerobot_export import export_arena_hdf5_to_lerobot

        destination = export_arena_hdf5_to_lerobot(
            args.source,
            args.output_root,
            args.repo_id,
            fps=args.fps,
            task=args.task,
        )
    print(f"ARENA_LEROBOT_EXPORT_OK path={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
