#!/usr/bin/env python3
"""Stop an existing OpenFlex VR control stack before starting a replacement."""

import os
from pathlib import Path
import signal
import sys
import time


VR_EXECUTABLES = frozenset(
    {
        "pico_pose_bridge_node",
        "openarmx_teleop_vr_node",
        "head_teleop_node",
        "vr_teleop_node",
        "vr_lift_control_node",
    }
)


def _read_cmdline(process_dir: Path):
    try:
        raw = (process_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _is_vr_stack_process(argv) -> bool:
    basenames = {Path(argument).name for argument in argv}
    if basenames & VR_EXECUTABLES:
        return True
    return (
        "ros2" in basenames
        and "launch" in argv
        and "openflex_isaac_bringup" in argv
        and "vr_teleop.launch.py" in argv
    )


def find_stale_vr_processes(proc_root=Path("/proc"), exclude_pids=None):
    excluded = set(exclude_pids or ())
    matches = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid in excluded:
            continue
        if _is_vr_stack_process(_read_cmdline(process_dir)):
            matches.append(pid)
    return sorted(matches)


def _parent_pid(pid: int, proc_root=Path("/proc")) -> int:
    try:
        for line in (proc_root / str(pid) / "status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return 0


def current_process_ancestors(proc_root=Path("/proc")):
    ancestors = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        pid = _parent_pid(pid, proc_root)
    return ancestors


def _is_alive(pid: int, proc_root=Path("/proc")) -> bool:
    return (proc_root / str(pid)).exists()


def terminate_processes(pids, timeout_sec=2.0):
    targets = list(pids)
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            continue

    deadline = time.monotonic() + float(timeout_sec)
    while time.monotonic() < deadline:
        if not any(_is_alive(pid) for pid in targets):
            return
        time.sleep(0.05)

    for pid in targets:
        if not _is_alive(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    kill_deadline = time.monotonic() + 1.0
    while time.monotonic() < kill_deadline and any(
        _is_alive(pid) for pid in targets
    ):
        time.sleep(0.05)


def main() -> int:
    targets = find_stale_vr_processes(exclude_pids=current_process_ancestors())
    if not targets:
        print("No existing OpenFlex VR processes found")
        return 0
    print("Stopping existing OpenFlex VR processes: " + ", ".join(map(str, targets)))
    terminate_processes(targets)
    survivors = [pid for pid in targets if _is_alive(pid)]
    if survivors:
        print("Failed to stop VR processes: " + ", ".join(map(str, survivors)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
