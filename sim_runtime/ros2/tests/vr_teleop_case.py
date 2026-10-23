#!/usr/bin/env python3
"""VR 遥操作单体测试（一次只测一个执行器，空场景）。

用法（需要 source ROS 与 openflex_ws 的 install/setup.bash）：
    python3 vr_teleop_case.py --case arms
    python3 vr_teleop_case.py --case head
    python3 vr_teleop_case.py --case base
    python3 vr_teleop_case.py --case lift
    python3 vr_teleop_case.py --case hold

前提：仿真（空场景 openflex_empty_scene）+ ROS 桥都在跑，5100/24102/24103 已就绪。
脚本只发 UDP 到 5100（等价头显），订阅 ROS 话题做判定；不依赖真机硬件。
每个 case 只驱动对应的那一路，其余输入保持静默，便于定位。
"""
from __future__ import annotations

import argparse
import math
import socket
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray

UDP_TARGET = ("127.0.0.1", 5100)
ARM_JOINTS = [f"openarmx_{side}_joint{i}" for side in ("left", "right") for i in range(1, 8)]
HEAD_JOINTS = ["openarmx_head_yaw_joint", "openarmx_head_pitch_joint"]
LIFT_JOINT = "lift_joint"


class Probe(Node):
    def __init__(self) -> None:
        super().__init__("vr_case_probe")
        self.joints: dict[str, float] = {}
        self.joint_hist: list[dict[str, float]] = []
        self.cmd: list[tuple[float, float, float]] = []
        self.jog: list[float] = []
        self.head_cmd: list[tuple[float, ...]] = []
        self.create_subscription(JointState, "/joint_states", self._on_joints, 20)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
        self.create_subscription(
            Float64, "/lift_manual_position_controller/jog_command", self._on_jog, 20
        )
        self.create_subscription(
            Float64MultiArray, "/head_forward_position_controller/commands", self._on_head, 20
        )

    def _on_joints(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self.joints[name] = float(pos)
        if self.joints:
            self.joint_hist.append(dict(self.joints))

    def _on_cmd(self, msg: Twist) -> None:
        self.cmd.append((msg.linear.x, msg.linear.y, msg.angular.z))

    def _on_jog(self, msg: Float64) -> None:
        self.jog.append(float(msg.data))

    def _on_head(self, msg: Float64MultiArray) -> None:
        self.head_cmd.append(tuple(float(v) for v in msg.data))


def send(sock: socket.socket, text: str) -> None:
    sock.sendto(text.encode(), UDP_TARGET)


def hand_packets(sock: socket.socket, t: float) -> None:
    ang = 2.0 * math.pi * (t / 4.0)
    lx, ly = 0.15 + 0.10 * math.cos(ang), 0.10 * math.sin(ang)
    rx, ry = -0.15 - 0.10 * math.cos(ang), 0.10 * math.sin(ang)
    ts = time.time_ns()
    send(sock, f"HAND L {lx:.4f} {ly:.4f} 0.0 0 0 0 1 1.0 1.0 0.0 0.0 0 {ts}")
    send(sock, f"HAND R {rx:.4f} {ry:.4f} 0.0 0 0 0 1 1.0 1.0 0.0 0.0 0 {ts}")


def lift_packets(sock: socket.socket, t: float, pressed: bool) -> None:
    """升降依赖左手 pose 续命，所以同时用静态 HAND L 心跳。"""
    ts = time.time_ns()
    send(sock, f"HAND L 0.15 0.0 0.0 0 0 0 1 1.0 1.0 0.0 0.0 0 {ts}")
    send(sock, f"BTN LEFT Y {1 if pressed else 0} {ts}")


def head_packets(sock: socket.socket, t: float, press_b: bool) -> None:
    ts = time.time_ns()
    if press_b:
        send(sock, f"BTN RIGHT B 1 {ts}")
    yaw = math.radians(20.0) * math.sin(2.0 * math.pi * t / 4.0)
    send(sock, f"HEAD 0 0 0 0 {math.sin(yaw/2):.5f} 0 {math.cos(yaw/2):.5f} {ts}")


def range_of(hist: list[dict[str, float]], names: list[str]) -> float:
    vals = {n: [h[n] for h in hist if n in h] for n in names}
    best = 0.0
    for samples in vals.values():
        if len(samples) >= 2:
            best = max(best, max(samples) - min(samples))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=("arms", "head", "base", "lift", "hold"))
    args = parser.parse_args()

    rclpy.init()
    probe = Probe()
    spin = threading.Thread(target=rclpy.spin, args=(probe,), daemon=True)
    spin.start()
    time.sleep(1.5)  # 等首批 /joint_states

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.monotonic()
    verdict = "FAIL"
    detail = ""

    if args.case == "arms":
        while time.monotonic() - start < 12.0:
            hand_packets(sock, time.monotonic() - start)
            time.sleep(0.05)
        span = range_of(probe.joint_hist, ARM_JOINTS)
        verdict = "PASS" if span > 0.05 else "FAIL"
        detail = f"双臂 12s 最大摆幅 {span:.3f} rad（阈值 >0.05）"

    elif args.case == "head":
        pressed = False
        while time.monotonic() - start < 12.0:
            t = time.monotonic() - start
            head_packets(sock, t, press_b=(not pressed and t >= 1.0))
            if not pressed and t >= 1.0:
                pressed = True
            time.sleep(0.05)
        span = range_of(probe.joint_hist, HEAD_JOINTS)
        verdict = "PASS" if span > 0.05 else "FAIL"
        detail = f"头部 12s 最大摆幅 {span:.3f} rad（阈值 >0.05，含一次 B 开启）"

    elif args.case == "base":
        while time.monotonic() - start < 8.0:
            send(sock, f"JOY L 0.6 0.0 {time.time_ns()}")
            time.sleep(0.05)
        send(sock, f"JOY L 0.0 0.0 {time.time_ns()}")
        time.sleep(1.0)
        peak = max((max(abs(x), abs(y)) for x, y, _ in probe.cmd), default=0.0)
        verdict = "PASS" if peak > 0.05 else "FAIL"
        detail = f"/cmd_vel 峰值 {peak:.3f} m/s（阈值 >0.05）"

    elif args.case == "lift":
        while time.monotonic() - start < 14.0:
            t = time.monotonic() - start
            lift_packets(sock, t, pressed=(3.0 <= t < 11.0))
            time.sleep(0.05)
        hist = [h[LIFT_JOINT] for h in probe.joint_hist if LIFT_JOINT in h]
        if len(hist) >= 2:
            delta = hist[-1] - hist[0]
            verdict = "PASS" if delta > 0.02 else "FAIL"
            detail = (
                f"升降 8s 上升 delta={delta:+.3f} rad（阈值 >+0.02）；"
                f"首 {hist[0]:+.3f} 末 {hist[-1]:+.3f}；jog 样本 {len(probe.jog)}"
            )
        else:
            detail = "没有拿到升降关节数据"

    elif args.case == "hold":
        # 静默 10 秒，验证不会因重力下沉
        time.sleep(10.0)
        hist = [h[LIFT_JOINT] for h in probe.joint_hist if LIFT_JOINT in h]
        if len(hist) >= 2:
            drift = hist[-1] - hist[0]
            verdict = "PASS" if abs(drift) < 0.005 else "FAIL"
            detail = f"静默 10s 升降漂移 {drift:+.4f} rad（阈值 |drift|<0.005）"
        else:
            detail = "没有拿到升降关节数据"

    sock.close()
    rclpy.shutdown()
    print(f"CASE {args.case}: {verdict} | {detail}")


if __name__ == "__main__":
    main()
