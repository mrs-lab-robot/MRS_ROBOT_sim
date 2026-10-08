"""Append one Isaac Lab-prepared episode using the selected LeRobot runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mrs_robot_lab.recorders.lerobot_export import append_prepared_episode_to_lerobot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-directory", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fps", required=True, type=int)
    parser.add_argument("--task", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset_path = append_prepared_episode_to_lerobot(
            args.prepared_directory,
            args.dataset_root,
            args.repo_id,
            fps=args.fps,
            task=args.task,
        )
    except Exception as error:
        print(
            json.dumps({"passed": False, "error": str(error)}, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        json.dumps(
            {"passed": True, "dataset_path": str(Path(dataset_path))},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
