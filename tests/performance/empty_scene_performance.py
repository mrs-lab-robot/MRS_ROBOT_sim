#!/usr/bin/env python3
"""测试空 Isaac 场景的实时仿真基线，不生成机器人或启动控制器。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "performance"
REPORT_PATH = REPORT_DIR / "empty_scene_performance.md"
RAW_ROOT = REPORT_DIR / "raw" / "empty_scene"


@dataclass
class ClockSamples:
    receive_wall: list[float] = field(default_factory=list)
    sim: list[float] = field(default_factory=list)
    backward_count: int = 0
    duplicate_count: int = 0

    def add(self, wall: float, sim: float) -> None:
        self.receive_wall.append(wall)
        if not self.sim:
            self.sim.append(sim)
        elif sim < self.sim[-1]:
            self.backward_count += 1
        elif sim == self.sim[-1]:
            self.duplicate_count += 1
        else:
            self.sim.append(sim)


class ClockMonitor(Node):
    def __init__(self) -> None:
        super().__init__("measure_empty_scene_performance")
        self.samples = ClockSamples()
        self.create_subscription(Clock, "/clock", self._clock_cb, 10)

    def _clock_cb(self, message: Clock) -> None:
        sim = float(message.clock.sec) + float(message.clock.nanosec) / 1_000_000_000.0
        self.samples.add(time.monotonic(), sim)


def _interval_hz(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    gaps = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not gaps:
        return None
    return 1.0 / (sum(gaps) / len(gaps))


def _default_isaac_path() -> Path:
    configured = os.environ.get("ISAACSIM_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    for candidate in (Path.home() / "isaacsim-6.0", Path.home() / "isaacsim", Path("/isaac-sim")):
        if (candidate / "python.sh").is_file():
            return candidate
    return Path("/isaac-sim")


def _resolve_runtime_paths() -> tuple[Path, Path, Path]:
    from ament_index_python.packages import get_package_prefix, get_package_share_directory

    package_prefix = Path(get_package_prefix("openflex_isaac_bringup"))
    adapter = package_prefix / "lib" / "openflex_isaac_bringup" / "start_robot_control_sim.py"
    scripts_share = Path(get_package_share_directory("isaac_ros2_scripts"))
    stage = Path(get_package_share_directory("openflex_isaac_bringup")) / "config" / "empty_stage.usd"
    return adapter, scripts_share, stage


def _launch_command(args: argparse.Namespace) -> list[str]:
    adapter, scripts_share, stage = _resolve_runtime_paths()
    return [
        str(_default_isaac_path() / "python.sh"),
        str(adapter),
        str(scripts_share),
        str(stage),
        str(args.render_hz),
        str(args.physics_hz),
        str(args.real_hz),
        "true" if args.headless else "false",
        str(args.api_port),
    ]


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _run(args: argparse.Namespace, batch_dir: Path) -> dict[str, Any]:
    log_path = batch_dir / "launch.log"
    command = _launch_command(args)
    environment = os.environ.copy()
    environment.setdefault("RCUTILS_COLORIZED_OUTPUT", "0")
    environment["OPENFLEX_EMPTY_SCENE"] = "1"
    process = subprocess.Popen(
        command,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
        text=True,
    )
    node = ClockMonitor()
    result: dict[str, Any] = {
        "status": "NOT_RUN",
        "command": command,
        "launch_log": str(log_path),
        "configured_physics_hz": args.physics_hz,
        "configured_render_hz": args.render_hz,
        "configured_real_hz": args.real_hz,
        "headless": args.headless,
    }
    try:
        deadline = time.monotonic() + args.startup_timeout
        while rclpy.ok() and time.monotonic() < deadline and process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.1)
            if len(node.samples.sim) >= 2:
                break
        if len(node.samples.sim) < 2:
            result["failure"] = "empty scene did not publish enough /clock samples"
            return result

        node.samples = ClockSamples()
        start_wall = time.monotonic()
        end_wall = start_wall + args.duration
        while rclpy.ok() and time.monotonic() < end_wall and process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
        end_wall = time.monotonic()
        samples = node.samples
        sim_seconds = samples.sim[-1] - samples.sim[0] if len(samples.sim) >= 2 else None
        wall_seconds = samples.receive_wall[-1] - samples.receive_wall[0] if len(samples.receive_wall) >= 2 else None
        rtf = sim_seconds / wall_seconds if sim_seconds is not None and wall_seconds and wall_seconds > 0.0 else None
        physics_step_rate = (
            sim_seconds / (1.0 / args.physics_hz) / wall_seconds
            if sim_seconds is not None and wall_seconds and wall_seconds > 0.0
            else None
        )
        result.update(
            {
                "status": "PASS" if rtf is not None and samples.backward_count == 0 else "NOT_VALID_TIME_RESET",
                "duration_seconds": end_wall - start_wall,
                "clock_receive_hz": _interval_hz(samples.receive_wall),
                "clock_sim_timestamp_hz": _interval_hz(samples.sim),
                "sim_time_seconds": sim_seconds,
                "wall_time_seconds": wall_seconds,
                "rtf": rtf,
                "estimated_physics_step_rate_hz": physics_step_rate,
                "clock_samples": len(samples.receive_wall),
                "clock_unique_sim_samples": len(samples.sim),
                "sim_backward_count": samples.backward_count,
                "sim_duplicate_count": samples.duplicate_count,
            }
        )
    except Exception as error:  # noqa: BLE001
        result["status"] = "TEST_ERROR"
        result["failure"] = str(error)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _stop_process(process)
        result["launch_returncode"] = process.returncode
    return result


def _batch_id(executor: str) -> str:
    safe_executor = "".join(
        character if character.isalnum() or character in "_-" else "-" for character in executor
    ).strip("-") or "manual"
    return f"{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-空场景-{safe_executor}"


def _markdown(result: dict[str, Any], batch_id: str, executed_at: str, executor: str) -> str:
    def fmt(value: Any) -> str:
        return "未采集" if value is None else f"{float(value):.3f}"

    realtime_conclusion = (
        "通过"
        if result.get("rtf") is not None
        and float(result["rtf"]) >= 1.0
        and result.get("sim_backward_count") == 0
        and result.get("sim_duplicate_count") == 0
        else ("测试异常" if result.get("status") == "TEST_ERROR" else "未通过")
    )

    return "\n".join(
        [
            f"### 批次：{batch_id}",
            "",
            "| 字段 | 内容 |",
            "| --- | --- |",
            f"| 执行时间 | {executed_at} |",
            f"| 执行者 | {executor} |",
            "| 版本 | 未记录 |",
            "| 测试范围 | 空场景；不生成机器人；不启动 `ros2_control` 或传感器 |",
            f"| 配置 | {'headless' if result.get('headless') else 'GUI'}；physics `{result.get('configured_physics_hz')} Hz`；render `{result.get('configured_render_hz')} Hz`；real loop `{result.get('configured_real_hz')} Hz` |",
            "| 功能结论 | 不适用 |",
            f"| 实时结论 | {realtime_conclusion} |",
            "",
            "| 指标 | 通过线 | 实测值 | 判定 |",
            "| --- | --- | ---: | --- |",
            f"| 仿真时间回退 | 0 | {result.get('sim_backward_count', '未采集')} | {'通过' if result.get('sim_backward_count') == 0 else '未通过'} |",
            f"| 重复时间戳 | 0 | {result.get('sim_duplicate_count', '未采集')} | {'通过' if result.get('sim_duplicate_count') == 0 else '未通过'} |",
            f"| `/clock` 接收频率 | 仅记录 | {fmt(result.get('clock_receive_hz'))} Hz | 仅记录 |",
            f"| physics step rate | 仅记录 | {fmt(result.get('estimated_physics_step_rate_hz'))} Hz | 仅记录 |",
            f"| RTF | 不低于 1.0 | {fmt(result.get('rtf'))} | {'通过' if result.get('rtf') is not None and float(result['rtf']) >= 1.0 else '未通过'} |",
            "",
            "结论：空场景只用于负载基线，不能替代机器人控制或传感器验收。",
            "",
            f"原始证据：`raw/empty_scene/{batch_id}/result.json`、`raw/empty_scene/{batch_id}/launch.log`。",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--physics-hz", type=float, default=120.0)
    parser.add_argument("--render-hz", type=float, default=60.0)
    parser.add_argument("--real-hz", type=float, default=120.0)
    parser.add_argument("--api-port", type=int, default=8141)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--executor", default=os.environ.get("ISAACSIM_TEST_EXECUTOR", "manual"))
    args = parser.parse_args()
    if args.duration <= 0.0 or args.physics_hz <= 0.0 or args.render_hz <= 0.0 or args.real_hz <= 0.0:
        parser.error("duration and frequencies must be positive")
    os.environ.setdefault("ROS_DOMAIN_ID", str(240 + os.getpid() % 10))
    batch_id = _batch_id(args.executor)
    batch_dir = RAW_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    executed_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    rclpy.init()
    result = _run(args, batch_dir)
    raw_path = batch_dir / "result.json"
    raw_path.write_text(
        json.dumps(
            {"batch_id": batch_id, "executed_at": executed_at, "executor": args.executor, "result": result},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with REPORT_PATH.open("a", encoding="utf-8") as report:
        report.write("\n" + _markdown(result, batch_id, executed_at, args.executor))
    print(REPORT_PATH)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
