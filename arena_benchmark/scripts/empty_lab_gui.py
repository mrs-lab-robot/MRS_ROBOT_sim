"""空 IsaacLab GUI：只起环境和地面，不加载任何机器人资产。

跑法:
  cd /media/y/workdisk/project/MRS_ROBOT/MRS_ROBOT_sim/arena_benchmark
  bash scripts/run_native_isaac.sh scripts/empty_lab_gui.py

窗口出来后自己拖 USD 进来（或 File -> Open），物理一直在步进，
拖入的机器人立即参与物理，直接看立柱掉不掉。
"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

sim = SimulationContext(SimulationCfg(device="cuda:0", dt=1.0 / 120.0))

# 地面 + 灯光：判断"掉不掉"必须有地面参照；要纯空场景删掉下面三行
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
light = sim_utils.DomeLightCfg(intensity=2000.0)
light.func("/World/Light", light)

sim.reset()

while simulation_app.is_running():
    sim.step()  # 物理持续步进：拖入的 USD 立即受重力作用

simulation_app.close()
