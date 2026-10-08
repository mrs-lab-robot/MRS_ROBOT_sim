"""Command-line entry point for the single Isaac Lab capture worker."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

from openflex_isaac_contract.session_config import SessionConfig, WorkerState
from mrs_robot_lab.recorders.capture_api import (
    CaptureControlServer,
    DEFAULT_CAPTURE_API_PORT,
)
from mrs_robot_lab.workers import IsaacLabWorker, WorkerStatus


PREFLIGHT_PREFIX = "ISAACLAB_PREFLIGHT_RESULT "
logger = logging.getLogger("mrs_robot_lab.worker_cli")


def _load_config(args: argparse.Namespace) -> SessionConfig:
    if args.config_json:
        payload = json.loads(args.config_json)
    else:
        path = Path(args.config_file).expanduser()
        if path.stat().st_size > 1_000_000:
            raise ValueError("SessionConfig 文件超过 1 MB")
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("SessionConfig 必须是 JSON 对象")
    return SessionConfig.from_dict(payload)


def create_lerobot_export_callback(
    *,
    subprocess_runner: Callable[..., Any] | None = None,
    prepare_episode: Callable[..., Path] | None = None,
) -> Callable[[Path, SessionConfig, dict[str, Any]], Path]:
    """Bridge Isaac Lab's HDF5 runtime to the configured LeRobot Python.

    HDF5 decoding and staging happen in the Isaac Lab process, where h5py is
    available. The selected LeRobot interpreter receives only NumPy/JPEG
    staging files, so the two runtime environments need not share packages.
    """
    if subprocess_runner is None:
        subprocess_runner = subprocess.run
    if prepare_episode is None:
        from mrs_robot_lab.recorders.lerobot_export import prepare_arena_hdf5_for_lerobot

        prepare_episode = prepare_arena_hdf5_for_lerobot

    def export_episode(
        hdf5_path: Path,
        config: SessionConfig,
        _metadata: dict[str, Any],
    ) -> Path:
        required = {
            "simulation_repo_root": config.simulation_repo_root,
            "lerobot_python": config.lerobot_python,
            "dataset_root": config.dataset_root,
            "dataset_id": config.dataset_id,
        }
        missing = sorted(name for name, value in required.items() if not str(value).strip())
        if missing:
            raise ValueError("LeRobot 导出配置缺少：" + ", ".join(missing))

        source = Path(hdf5_path).expanduser().resolve(strict=True)
        source_reference = source.name
        if config.episode_root.strip():
            episode_root = Path(config.episode_root).expanduser().resolve()
            try:
                source_reference = source.relative_to(episode_root).as_posix()
            except ValueError:
                # The source may be imported from an external location. Keep
                # only its basename rather than persisting a host-specific path.
                pass
        simulation_root = Path(config.simulation_repo_root).expanduser()
        dataset_root = Path(config.dataset_root).expanduser()
        repo_id = config.dataset_id.strip()
        task_id = config.task_id.strip()
        fps = int(round(config.frequency.output_fps))

        package_paths = (
            str(simulation_root / "isaaclab_ext" / "src"),
            str(
                simulation_root
                / "sim_runtime"
                / "ros2"
                / "openflex_isaac_sim"
                / "openflex_isaac_contract"
            ),
        )
        environment = os.environ.copy()
        environment["MRS_ROBOT_SIM_ROOT"] = str(simulation_root)
        # The selected interpreter must not import Isaac Lab/Kit's Python
        # packages from the parent process. Only expose the framework-neutral
        # MRS sources; the LeRobot interpreter owns all third-party packages.
        environment["PYTHONPATH"] = os.pathsep.join(package_paths)

        with tempfile.TemporaryDirectory(
            prefix=".lerobot-staging-",
            dir=source.parent,
        ) as staging_root:
            prepared_directory = Path(staging_root) / "episode"
            prepared_episode = prepare_episode(
                source,
                prepared_directory,
                fps=fps,
                task=task_id,
                source_reference=source_reference,
            )
            command = [
                str(Path(config.lerobot_python).expanduser()),
                "-m",
                "mrs_robot_lab.recorders.lerobot_export_cli",
                "--prepared-directory",
                str(prepared_episode),
                "--dataset-root",
                str(dataset_root),
                "--repo-id",
                repo_id,
                "--fps",
                str(fps),
                "--task",
                task_id,
            ]
            try:
                completed = subprocess_runner(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=1200,
                    check=False,
                    env=environment,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("LeRobot 导出超过 20 分钟，HDF5 已保留，可重试导出") from error
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "无进程错误输出").strip()
                raise RuntimeError(
                    f"LeRobot Python 导出失败 (exit={completed.returncode}): {detail[-4000:]}"
                )

        return dataset_root.joinpath(*repo_id.split("/"))

    return export_episode


def preflight_result(
    config: SessionConfig,
    *,
    worker_factory: Callable[[SessionConfig], Any] = IsaacLabWorker,
) -> tuple[dict[str, Any], int]:
    """Validate static configuration without importing/starting Kit."""
    try:
        worker = worker_factory(config)
        worker.configure()
        status = worker.get_status()
        return (
            {
                "passed": True,
                "state": status.state.value,
                "message": "静态检查通过；尚未启动 Isaac Sim/Kit。",
                "metadata": status.metadata,
            },
            0,
        )
    except Exception as error:
        return (
            {
                "passed": False,
                "state": "preflight_failed",
                "message": str(error),
            },
            2,
        )


def _update_api_status(server: Any, status: WorkerStatus) -> None:
    state = status.state.value
    metadata = status.metadata
    server.update_status(
        state=state,
        ready=state in {WorkerState.ENV_READY.value, WorkerState.RECORDING.value},
        recording=state == WorkerState.RECORDING.value,
        steps=int(metadata.get("steps", 0)),
        last_saved_path=str(metadata.get("last_saved_path", "")),
        last_export_error=str(metadata.get("last_export_error", "")),
        last_quarantined_path=str(metadata.get("last_quarantined_path", "")),
        last_qc_failure_reasons=list(metadata.get("last_qc_failure_reasons", ())),
        teleop_mode=str(metadata.get("teleop_mode", "")),
        teleop_ready=bool(metadata.get("teleop", {}).get("ready", False)),
        teleop_message=str(metadata.get("teleop", {}).get("message", "")),
        message=status.message,
        session_id=str(metadata.get("session_id", "")),
        config_hash=str(metadata.get("config_hash", "")),
        episode_id=str(metadata.get("episode_id", "")),
        task_id=str(metadata.get("environment_id", "")),
        failure_reason=str(metadata.get("failure_reason", "")),
    )


def serve_worker(
    worker: Any,
    server: Any,
    *,
    sleep_fn: Callable[[float], Any] = time.sleep,
    max_iterations: int | None = None,
    stop_event: threading.Event | None = None,
) -> int:
    """Run Kit on this thread and dispatch loopback API commands on that thread."""
    worker.add_status_listener(lambda status: _update_api_status(server, status))
    server.start()
    attach_transport = getattr(worker, "attach_teleop_transport", None)
    if callable(attach_transport):
        attach_transport(server)
    iteration = 0
    try:
        worker.launch()
        while worker.state != WorkerState.STOPPING:
            if stop_event is not None and stop_event.is_set():
                try:
                    worker.shutdown()
                except Exception as error:
                    server.update_status(message=str(error), last_command_error=str(error))
                    logger.error("Shutdown deferred until active Episode is saved/discarded: %s", error)
                    stop_event.clear()
                    continue
                break
            for request in server.take_requests():
                try:
                    worker.handle_command(request.command, request.payload)
                    server.update_status(last_command_error="")
                except Exception as error:
                    logger.exception("Worker API command failed: %s", request.command)
                    server.update_status(
                        message=str(error),
                        last_command_error=str(error),
                    )
                _update_api_status(server, worker.get_status())
                if worker.state == WorkerState.STOPPING:
                    break
            if worker.state == WorkerState.STOPPING:
                break
            worker.pump_once()
            _update_api_status(server, worker.get_status())
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            sleep_fn(0.01)
        return 0
    except Exception as error:
        logger.exception("Isaac Lab worker failed")
        server.update_status(
            state=WorkerState.FAILED.value,
            ready=False,
            recording=False,
            message=str(error),
            failure_reason=str(error),
        )
        print(f"ISAACLAB_WORKER_ERROR {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        if worker.state not in {WorkerState.STOPPING, WorkerState.IDLE}:
            try:
                worker.shutdown()
            except Exception as error:
                logger.exception("Worker cleanup failed")
                print(f"ISAACLAB_WORKER_CLEANUP_ERROR {error}", file=sys.stderr, flush=True)
        server.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config-json", help="frozen SessionConfig JSON")
    config_group.add_argument("--config-file", help="path to a frozen SessionConfig JSON file")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--api-port", type=int, default=DEFAULT_CAPTURE_API_PORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = _load_config(args)
    except Exception as error:
        print(f"ISAACLAB_WORKER_ERROR {error}", file=sys.stderr, flush=True)
        return 2

    if args.preflight_only:
        result, code = preflight_result(config)
        print(PREFLIGHT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
        return code

    if not 1 <= args.api_port <= 65535:
        print("ISAACLAB_WORKER_ERROR API 端口必须在 1 到 65535 之间", file=sys.stderr, flush=True)
        return 2

    worker = IsaacLabWorker(
        config,
        export_callback=create_lerobot_export_callback(),
    )
    try:
        worker.configure()
    except Exception as error:
        print(f"ISAACLAB_WORKER_ERROR {error}", file=sys.stderr, flush=True)
        return 2
    server = CaptureControlServer(host="127.0.0.1", port=args.api_port)

    previous_handlers = {}
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    try:
        return serve_worker(worker, server, stop_event=stop_event)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
