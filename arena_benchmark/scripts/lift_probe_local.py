"""孤立探针：IsaacLab 直接加载 openflex_robot.usda，零指令步进，观察立柱/头部漂移。

用法（4090 桌面终端）:
  cd /media/y/workdisk/project/MRS_ROBOT/MRS_ROBOT_sim/arena_benchmark
  bash scripts/run_native_isaac.sh /tmp/lift_probe.py

看什么:
  1. 启动日志里 actuator 表的 lift_joint 行 —— k 应为 1.0e+04 / c=600（USD 值是否被读入）
  2. 窗口里按播放后立柱是否下坠（零指令，机器人应静止）
  3. 终端每 0.5 秒仿真时打印一次关键关节位置；lift 相对初始漂移越大=掉得越狠
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)  # 想要纯数字无窗口: 改成 True
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext

USD = "/media/y/workdisk/project/MRS_ROBOT/MRS_ROBOT_sim/sim_runtime/assets/robots/openflex_robot.usda"
WATCH = ("lift_joint", "openarmx_head_yaw_joint", "openarmx_head_pitch_joint")

# --- 场景：地面 + 灯光 + 与 Arena 同款物理频率（120Hz），GPU 用空闲的 1 号卡 ---
sim = SimulationContext(SimulationCfg(device="cuda:0", dt=1.0 / 120.0))
sim.set_camera_view(eye=(3.0, 3.0, 2.0), target=(0.0, 0.0, 0.8))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
light = sim_utils.DomeLightCfg(intensity=2000.0)
light.func("/World/Light", light)

# --- 机器人：原样加载 USD，驱动参数完全用 USD 里 authored 的值（stiffness=None = 不覆盖）---
robot_cfg = ArticulationCfg(
    prim_path="/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=USD),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.175)),
    actuators={
        "usd_authored_drives": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
            effort_limit=None,
            velocity_limit=None,
        )
    },
)
robot = Articulation(robot_cfg)

sim.reset()
print("\n===== 关节顺序（IsaacLab 实际顺序）=====")
for i, name in enumerate(robot.joint_names):
    print(f"  [{i:2d}] {name}")

idx = {n: i for i, n in enumerate(robot.joint_names)}
watch_idx = [idx[n] for n in WATCH if n in idx]

p0 = robot.data.joint_pos[0].detach().cpu()
print("\n===== 零指令步进 600 步（5 秒仿真时间）=====")
for step in range(601):
    robot.set_joint_position_target(torch.zeros(robot.num_joints, device=sim.device))
    robot.write_data_to_sim()
    sim.step()
    if step % 60 == 0:
        p = robot.data.joint_pos[0].detach().cpu()
        msg = "  ".join(
            f"{n}={p[i].item():+.4f}(漂移{p[i].item()-p0[i].item():+.4f})"
            for n, i in zip(WATCH, watch_idx)
        )
        print(f"[step {step:4d}] {msg}")

print("\n完成。若 lift 漂移持续增大 = IsaacLab 链路下立柱在掉；≈0 = 资产没问题，问题在链路别处。")
simulation_app.close()
