#!/usr/bin/env python3
"""复测独立 RealSense 场景的 Isaac 内部时序和 ROS 2 实收图像频率。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


ROOT = Path(__file__).resolve().parents[2]
SENSOR_SCRIPT = (
    ROOT
    / "sim_runtime"
    / "ros2"
    / "openflex_isaac_sim"
    / "openflex_isaac_sensors"
    / "scripts"
    / "realsense_standalone.py"
)
SENSOR_CONFIG_ROOT = ROOT / "sim_runtime" / "config" / "sensors" / "realsense"
RAW_ROOT = ROOT / "reports" / "sensors" / "raw"
REPORT_PATH = ROOT / "reports" / "sensors" / "sensor_validation.md"
IMAGE_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
)

CAMERA_NAMES = {
    "both": ("d435_standalone", "d405_standalone"),
    "quad": (
        "d435_front_standalone",
        "d435_rear_standalone",
        "d405_left_standalone",
        "d405_right_standalone",
    ),
}


@dataclass
class StreamSamples:
    wall: list[float] = field(default_factory=list)
    sim: list[float] = field(default_factory=list)
    backward_count: int = 0
    duplicate_count: int = 0

    def add(self, wall: float, sim: float) -> None:
        if self.sim and sim < self.sim[-1]:
            self.backward_count += 1
            return
        if self.sim and sim == self.sim[-1]:
            self.duplicate_count += 1
            return
        self.wall.append(wall)
        self.sim.append(sim)

    def reset(self) -> None:
        self.wall.clear()
        self.sim.clear()
        self.backward_count = 0
        self.duplicate_count = 0

    def metrics(self) -> dict[str, float | int | bool | None]:
        if len(self.wall) < 2 or len(self.sim) < 2:
            return {
                "messages": len(self.wall),
                "wall_receive_hz": None,
                "sim_timestamp_hz": None,
                "timestamp_monotonic": self.backward_count == 0,
                "backward_count": self.backward_count,
                "duplicate_count": self.duplicate_count,
            }
        wall_duration = self.wall[-1] - self.wall[0]
        sim_duration = self.sim[-1] - self.sim[0]
        return {
            "messages": len(self.wall),
            "wall_receive_hz": (len(self.wall) - 1) / wall_duration if wall_duration > 0.0 else None,
            "sim_timestamp_hz": (len(self.sim) - 1) / sim_duration if sim_duration > 0.0 else None,
            "timestamp_monotonic": self.backward_count == 0,
            "backward_count": self.backward_count,
            "duplicate_count": self.duplicate_count,
        }


class ImageAudit(Node):
    def __init__(self, topics: tuple[str, ...]) -> None:
        super().__init__("audit_realsense_standalone")
        self.samples = {topic: StreamSamples() for topic in topics}
        self._subscriptions = [
            self.create_subscription(Image, topic, lambda message, name=topic: self._callback(name, message), IMAGE_QOS)
            for topic in topics
        ]

    def _callback(self, topic: str, message: Image) -> None:
        stamp = message.header.stamp
        sim = float(stamp.sec) + float(stamp.nanosec) / 1_000_000_000.0
        self.samples[topic].add(time.monotonic(), sim)

    def ready(self) -> bool:
        return all(sample.wall for sample in self.samples.values())

    def reset(self) -> None:
        for sample in self.samples.values():
            sample.reset()

    def metrics(self) -> dict[str, dict[str, float | int | bool | None]]:
        return {topic: sample.metrics() for topic, sample in self.samples.items()}


def _default_isaac_path() -> Path:
    configured = os.environ.get("ISAACSIM_PATH", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend((Path.home() / "isaacsim-6.0", Path.home() / "isaacsim", Path("/isaac-sim")))
    for candidate in candidates:
        if (candidate / "python.sh").is_file():
            return candidate
    raise FileNotFoundError("Isaac Sim python.sh was not found; set ISAACSIM_PATH")


def _topics(camera_group: str) -> tuple[str, ...]:
    return tuple(
        f"/{name}/{stream}"
        for name in CAMERA_NAMES[camera_group]
        for stream in ("color/image_raw", "depth/image_rect_raw")
    )


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=30.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _batch_id(camera_group: str, executor: str) -> str:
    safe_executor = "".join(char if char.isalnum() or char in "_-" else "-" for char in executor).strip("-")
    return f"{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-独立相机-{camera_group}-{safe_executor or 'manual'}"


def _command(args: argparse.Namespace, output: Path, steps: int) -> list[str]:
    config_name = "realsense_standalone.yaml" if args.camera == "both" else "realsense_quad.yaml"
    return [
        str(_default_isaac_path() / "python.sh"),
        str(SENSOR_SCRIPT),
        "--config", str(SENSOR_CONFIG_ROOT / config_name),
        "--camera", args.camera,
        "--physics-hz", str(args.physics_hz),
        "--render-hz", str(args.render_hz),
        "--pacing", "paced_online",
        "--warmup-steps", str(args.warmup_steps),
        "--steps", str(steps),
        "--pace-spin-us", str(args.pace_spin_us),
        "--pace-sleep-guard-us", str(args.pace_sleep_guard_us),
        "--kit-threads", str(args.kit_threads),
        "--ros2",
        "--ros2-qos-profile", "reliable",
        "--ros2-publish-queue-thread", "off",
        "--output", str(output),
    ]


def _run(args: argparse.Namespace, batch_dir: Path) -> dict[str, Any]:
    isaac_output = batch_dir / "isaac"
    isaac_output.mkdir()
    log_path = batch_dir / "launch.log"
    # Keep the Isaac loop alive beyond the ROS sampling window, then let it
    # exit naturally. Sending SIGINT immediately after an audit can prevent
    # the standalone entry point from flushing its internal metrics.json.
    steps = args.steps or math.ceil((args.duration + 5.0) * args.physics_hz)
    command = _command(args, isaac_output, steps)
    environment = os.environ.copy()
    environment.setdefault("ROS_DOMAIN_ID", "42")
    environment["ROS_LOCALHOST_ONLY"] = "1"
    environment["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
    environment.pop("ROS_DISCOVERY_SERVER", None)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=True,
        text=True,
    )
    audit = ImageAudit(_topics(args.camera))
    result: dict[str, Any] = {
        "status": "TEST_ERROR",
        "command": command,
        "camera_group": args.camera,
        "expected_render_products": len(CAMERA_NAMES[args.camera]),
        "expected_ros_image_topics": len(_topics(args.camera)),
        "configured_steps": steps,
        "launch_log": str(log_path.relative_to(ROOT)),
        "isaac_metrics": str((isaac_output / "metrics.json").relative_to(ROOT)),
    }
    try:
        startup_deadline = time.monotonic() + args.startup_timeout
        while rclpy.ok() and time.monotonic() < startup_deadline and process.poll() is None:
            rclpy.spin_once(audit, timeout_sec=0.1)
            if audit.ready():
                break
        if not audit.ready():
            result["failure"] = "未在启动等待窗口内收到全部 ROS 图像话题"
            return result
        audit.reset()
        deadline = time.monotonic() + args.duration
        while rclpy.ok() and time.monotonic() < deadline and process.poll() is None:
            rclpy.spin_once(audit, timeout_sec=0.05)
        result["ros2_streams"] = audit.metrics()
        result["measurement_wall_seconds"] = args.duration
        if process.poll() is not None:
            result["failure"] = "Isaac Sim 在 ROS 2 测量窗口结束前退出"
            return result
        natural_exit_deadline = time.monotonic() + max(10.0, args.duration)
        while rclpy.ok() and time.monotonic() < natural_exit_deadline and process.poll() is None:
            rclpy.spin_once(audit, timeout_sec=0.05)
        if process.poll() is None:
            result["failure"] = "Isaac Sim 未在预定步数完成后自然退出"
            return result
        if process.returncode != 0:
            result["failure"] = f"Isaac Sim 正常完成窗口后异常退出：{process.returncode}"
            return result
        result["status"] = "COMPLETE"
        return result
    except Exception as error:  # noqa: BLE001
        result["failure"] = str(error)
        return result
    finally:
        audit.destroy_node()
        _stop_process(process)
        result["launch_returncode"] = process.returncode
        internal = _read_json(isaac_output / "metrics.json")
        result["isaac_internal"] = internal
        if result.get("status") == "COMPLETE" and internal is None:
            result["status"] = "TEST_ERROR"
            result["failure"] = "Isaac Sim 未写出内部 metrics.json"
        if rclpy.ok():
            rclpy.shutdown()


def _result_conclusions(result: dict[str, Any]) -> tuple[str, str]:
    internal = result.get("isaac_internal") or {}
    streams = result.get("ros2_streams") or {}
    rtf = internal.get("physics_rtf")
    stream_valid = bool(streams) and all(
        metrics.get("wall_receive_hz") is not None
        and float(metrics["wall_receive_hz"]) >= 29.0
        and bool(metrics.get("timestamp_monotonic"))
        for metrics in streams.values()
    )
    functional = "通过" if result.get("status") == "COMPLETE" and stream_valid else "未通过"
    realtime = "通过" if functional == "通过" and rtf is not None and float(rtf) >= 1.0 else "未通过"
    return functional, realtime


def _fmt(value: Any, digits: int = 3) -> str:
    return "未采集" if value is None else f"{float(value):.{digits}f}"


def _markdown(batch_id: str, executed_at: str, executor: str, result: dict[str, Any]) -> str:
    functional, realtime = _result_conclusions(result)
    internal = result.get("isaac_internal") or {}
    streams = result.get("ros2_streams") or {}
    image_summary = "；".join(
        f"`{topic}`={_fmt(metrics.get('wall_receive_hz'))} Hz"
        for topic, metrics in streams.items()
    ) or "未采集"
    return "\n".join((
        f"### 批次：{batch_id}",
        "",
        "| 字段 | 内容 |",
        "| --- | --- |",
        f"| 执行时间 | {executed_at} |",
        f"| 执行者 | {executor} |",
        "| 版本 | Isaac Sim 6.0；当前迁移仓库同源独立相机入口 |",
        "| 测试范围 | 独立静态场景；不加载机器人、不启动机器人控制或阶段 4 集成 |",
        f"| 配置 | `{result.get('camera_group')}`；physics/render `90 Hz`；640×480；每路 RGB+Depth `30 Hz`；headless；Reliable QoS |",
        f"| 功能结论 | {functional} |",
        f"| 实时结论 | {realtime} |",
        "",
        "| 指标 | 通过线 | 实测值 | 判定 |",
        "| --- | --- | ---: | --- |",
        f"| RenderProduct 数 | `{result.get('expected_render_products')}` 且唯一 | {len(internal.get('render_products') or [])} | {'通过' if len(internal.get('render_products') or []) == result.get('expected_render_products') else '未通过'} |",
        f"| physics wall Hz | `90 Hz` | {_fmt(internal.get('physics_wall_hz'))} Hz | {'通过' if internal.get('physics_wall_hz') is not None and float(internal['physics_wall_hz']) >= 90.0 else '未通过'} |",
        f"| RTF | 不低于 `1.0` | {_fmt(internal.get('physics_rtf'))} | {'通过' if internal.get('physics_rtf') is not None and float(internal['physics_rtf']) >= 1.0 else '未通过'} |",
        f"| physics gap P99 / max | P99≤`13.333 ms`；max≤`22.222 ms` | {_fmt((internal.get('physics_wall_gap_p99_s') or 0.0) * 1000)} / {_fmt((internal.get('physics_wall_gap_max_s') or 0.0) * 1000)} ms | {'通过' if internal.get('physics_wall_gap_p99_s') is not None and internal.get('physics_wall_gap_max_s') is not None and float(internal['physics_wall_gap_p99_s']) <= 0.013333 and float(internal['physics_wall_gap_max_s']) <= 0.022222 else '未通过'} |",
        f"| ROS 图像实收频率 | 每路不低于 `29 Hz` | {image_summary} | {'通过' if functional == '通过' else '未通过'} |",
        "",
        f"结论：本批次只验证独立相机方法。{'未达到旧记录的全部指标。' if realtime != '通过' else '达到本次短测实时测评线；仍不能替代 30 分钟稳定性或机器人集成验收。'}",
        "",
        f"原始证据：`raw/{batch_id}/result.json`、`raw/{batch_id}/launch.log`、`raw/{batch_id}/isaac/metrics.json`。",
        "",
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=tuple(CAMERA_NAMES), required=True)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--physics-hz", type=float, default=90.0)
    parser.add_argument("--render-hz", type=float, default=90.0)
    parser.add_argument("--warmup-steps", type=int, default=90)
    parser.add_argument("--steps", type=int, default=0, help="0 表示按测量时长自动计算并留 5 秒收尾")
    parser.add_argument("--pace-spin-us", type=float, default=750.0)
    parser.add_argument("--pace-sleep-guard-us", type=float, default=0.0)
    parser.add_argument("--kit-threads", type=int, default=16)
    parser.add_argument("--executor", default=os.environ.get("ISAACSIM_TEST_EXECUTOR", "manual"))
    parser.add_argument("--append-report", action="store_true")
    args = parser.parse_args()
    if args.duration <= 0.0 or args.startup_timeout <= 0.0 or args.steps < 0:
        parser.error("duration and startup timeout must be positive; steps must be non-negative")
    if args.physics_hz != 90.0 or args.render_hz != 90.0:
        parser.error("此旧记录复测固定使用 physics/render 90 Hz")
    batch_id = _batch_id(args.camera, args.executor)
    batch_dir = RAW_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    executed_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    rclpy.init()
    result = _run(args, batch_dir)
    result["configured"] = {
        "physics_hz": args.physics_hz,
        "render_hz": args.render_hz,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "pace_spin_us": args.pace_spin_us,
        "pace_sleep_guard_us": args.pace_sleep_guard_us,
        "kit_threads": args.kit_threads,
    }
    result_path = batch_dir / "result.json"
    result_path.write_text(
        json.dumps({"batch_id": batch_id, "executed_at": executed_at, "executor": args.executor, "result": result}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    if args.append_report:
        with REPORT_PATH.open("a", encoding="utf-8") as report:
            report.write("\n" + _markdown(batch_id, executed_at, args.executor, result))
    print(result_path)
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
