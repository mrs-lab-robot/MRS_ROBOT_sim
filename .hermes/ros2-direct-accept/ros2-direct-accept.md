# ros2_direct LeRobot 直采完整验收（修复项 C）

日期：2026-10-13。执行环境：本机（代码核对/dry-run）+ A6000（实测目标，本轮被占用）。

## 凌晨实测结果（2026-10-14 01:53 收束，已落盘 ✅）

**A6000 实测完成**：`run_ros2_direct_accept.sh` 一键流程全程跑通，LeRobot ros2_direct
直采 1 集落盘（parquet + 4 路 mp4 + meta/provenance 齐全），**record 退出码 0**。

### 实际执行命令（A6000 = ssh 6000PRO-WYH）

同步本目录后一条命令（脚本内部展开 P0 清理→kit→rosbridge→driver→自检→record→列产物）：

```
bash ~/mrs_robot/MRS_ROBOT/MRS_ROBOT_sim/.hermes/ros2-direct-accept/run_ros2_direct_accept.sh
```

其中 P4 等价 GUI「开始采集」命令：

```
bash ~/mrs_robot/MRS_ROBOT/scripts/start_capture.sh \
  --config ~/mrs_robot/MRS_ROBOT/MRS_ROBOT_sim/.hermes/ros2-direct-accept/capture.yaml
```

### 实测暴露并修复的根因（dry-run 发现不了）

**start_capture.sh 的 ROS_LOCALHOST_ONLY 被 setup 链污染**——连续 2 次 record rc=1 的唯一根因：

- 脚本第 154 行 `eval` 出 `ROS_LOCALHOST_ONLY=true` 后，191-203 行 `source` humble 与
  openflex_ws 的 setup.bash 链把该值改写，212 行再用**被污染的值**归一化 → record 进程
  `ROS_LOCALHOST_ONLY=0`（/proc/<pid>/environ 铁证），而 relay/driver 均为 1 →
  **DDS 发现隔离**：record 自己的 /joint_states、/cam_* 订阅收不到任何消息，
  `robot.connect()` 等 joint_states 5s 超时（Missing joints 列出全部 19 个）。
- 同窗口对照实测：独立订阅者（=1 环境）在 record 运行中 12s/18s 两次探针均秒收
  joint_states 与 jpeg；kit/relay 全程存活无异常（ros.log 无任何 relay 报错）——
  **链路本身完好，只有 record 进程被隔离**。
- **修复**：`eval` 后快照解析值（`PARSED_ROS_DOMAIN_ID`/`PARSED_ROS_LOCALHOST_ONLY`），
  212 行改用快照导出（=1）。此即诊断 3"ros2_direct 未存 episode"的原始病根，
  GUI 本地采集流程同样受益。已同步 A6000，未提交 git。

### 脚本加固（本次实测新增）

1. P0 预清理：清上轮孤儿 kit/relay/vr 节点/driver（stop_all 只杀 wrapper 时子进程会孤儿化，
   VALIDATE_ONLY 模式特意保留进程，残留 kit 会顶掉下一轮 GPU 检查）。
2. 数据流闸门：record 前 `ros2 topic echo --once /joint_states`（3×30s），失败自动抓
   kit capture API status 后中止——把 DDS/链路问题拦在 record 之前，不再误报成 record 错误。
3. stop_all 补杀 arena_relay/vr 节点/driver 兜底。

### episode 产物（实测帧数 / 文件清单）

数据集根：`~/mrs_robot/MRS_ROBOT/openflex_ws/datasets/lerobot/ros2_accept_demo`

| 文件 | 大小 (B) |
|---|---|
| data/chunk-000/file-000.parquet | 33,673 |
| videos/observation.images.base/chunk-000/file-000.mp4 | 99,258 |
| videos/observation.images.head/chunk-000/file-000.mp4 | 405,734 |
| videos/observation.images.left_wrist/chunk-000/file-000.mp4 | 44,341 |
| videos/observation.images.right_wrist/chunk-000/file-000.mp4 | 71,146 |
| meta/info.json | 5,915 |
| meta/stats.json | 27,978 |
| meta/tasks.parquet | 2,151 |
| meta/episodes/chunk-000/file-000.parquet | 71,748 |
| meta/openflex_camera_config.yaml / effective_schema.yaml / capture_manifest.json | 1,367/1,762/319 |
| openflex_episode_provenance/episode_000000.jsonl | 453,151 |

- info.json：codebase v3.0 / fps=30 / total_episodes=1 / **total_frames=136** /
  observation.state shape [19]，features 含 action + state + 4 路相机
- parquet 136 行（episode_index=0），timestamp 按 30fps 打（间隔 33.3ms）
- provenance：task "pick up the red block"，started_at 2026-10-13T17:52:53Z

### 与 dry-run 的差异

1. dry-run 只核对命令文本、不执行 source 分支——**ROS_LOCALHOST_ONLY 污染只有实测暴露**。
2. 实测 136 帧：15s episode 窗口内采集循环跟不上 30fps 墙钟（ROS2 obs 拉取 + 4 路 jpeg
   解码），lerobot 按帧序号打时间戳，故 136 帧时间跨度 4.5s。**遗留**：采集帧率优化
   （MVP 未处理，不影响"落盘齐全"验收口径）。
3. 实测用 A6000 本地变体 hosts（全部 127.0.0.1），dry-run 报告已注明，无新增偏差。

### 过程记录

- run1 01:01 record rc=1（joint_states 超时）→ /proc 抓到环境隔离
- run2 01:28 加闸门后仍 rc=1 → 3 分钟稳定性复测 + record 运行中独立探针 → 锁定 record 进程自身
- run3 01:53 修复后 **rc=0**，episode 落盘；GPU 峰值 ~6 GB，结束回 84 MiB

## 结论（dry-run 轮，已被上节实测取代）

**本轮未完成 A6000 实测落盘**——按任务书协作约束跳过：启动时 nvidia-smi 检查发现修复项 A 的
agent 已在 A6000 起仿真（`teleop_vr_openflex.py --headless --enable_cameras --enable-lidar
--task-id openflex_pick_cube`，21:17 起，GPU 84 MiB → 5940 MiB，另有 livox relay）。

已完成的离线部分（不依赖 GPU）：
1. **全链路代码核对**：GUI→capture.yaml→start_capture.sh→lerobot_record_openflex.py 的
   ros2_direct 数据流逐文件核对（含 A6000 SHA 一致性校验，12 个关键文件 11 个 OK）。
2. **GUI 等价命令 dry-run 双双通过**（命令文本见下）；可疑点已排除。
3. **缺环定位 + 补齐**：无 VR 头显时整条链路唯一的数据源缺环（模拟 VR 命令流）已用
   `leader_driver.py` 补齐（见"链路与缺环"）。
4. **一键补测脚本** `run_ros2_direct_accept.sh` 已写好并通过 bash -n，修复项 A 仿真停止后
   一条命令即可完成实测验收。

## 链路与缺环（修复项 C 的实际解释）

诊断 3 结论"链路全通、只差跑流程"经核对属实，但补上"跑流程"需要一次**外部命令流注入**，
这是 GUI 流程中由 VR 硬件（pico_pose_bridge→openarmx_teleop_vr）隐式提供的一环：

```
[启动时的 GUI 等价]                        [采集时的 GUI 等价动作源]
GUI「启动仿真」                              VR 手柄 → pico_pose_bridge
  = scripts/start_sim.sh --config run.yaml  → /pico_*_controller/{pose,grip,…}
    = run_native_isaac.sh teleop_vr_openflex  → openarmx_teleop_vr (IK)
      --data-mode ros2_direct --enable_cameras→ /left|right_forward_position_controller/commands
                                              /head_…/commands, /lift_position_controller/commands
GUI「启动 ROS 桥」                           → /cmd_vel (底盘)
  = colcon build mrs_robot_arena_bridge
      + ros2 launch arena_vr_teleop.launch.py   ┌─ ARENA ROS bridge(relay 节点)
        (内含集成 VR launch + arena_relay)       │  ← /pico_* pose + 命令话题
      [缺环：无头环境无 VR 硬件]                  │  ← leader_driver.py(模拟 VR 命令流)
        ↓ relay 收 sim UDP(24103 状态/24104 相机)  →发 /joint_states + /cam_*/color/image/compressed
        ↓ relay 发 UDP 命令(24102) → sim 机器人执行动作
GUI「开始采集」(数据采集页)
  = bash scripts/start_capture.sh --config capture.yaml
    = lerobot_record_openflex.py
        robot=openarmx_follower_ros2  订 /joint_states + /cam_* + schema-driven obs
        teleop=openarmx_leader_ros2   订同批命令话题(require_live/both/non_arm=True)
        → 录满 episode_time_s 自动 dataset.save_episode() → parquet/mp4/meta 落盘
```

要点（代码依据）：
- `teleop_bridge.py:83-110`：schema 的 `actions.chassis.velocity.enabled=true` →
  leader `enable_base=True` 且 `base_action_mode=twist`（用 /cmd_vel，无 odom 依赖）——
  schema 选 `no_chassis_lio_act_vel_base.yaml`，观测侧 lio_pose/lio_velocity 均 disabled，
  不触发 /fastlio2/lio_odom 依赖（ros2_direct 的 sim 不发 odom；lidar/IMU 属修复项 A 领域）。
- `aggregator.py command_frame`：control_mode=vr 时 deadman 要求 /pico_*_controller/pose
  在 0.25s 内有消息（头显在线判定）；leader command_timeout_s=1.0 同样要求命令话题在线。
  → leader_driver.py 以 20Hz 同时发 pose（仅保活）与左右臂(7关节+手指1)、头(2)、升降(1)、
  cmd_vel(0) 命令，全部落在 relay 已声明的订阅上。
- `teleop_adapter.py action_from_command`：sim 端将 arm_position 当**绝对关节目标**用，
  限位/限速 clamp 在 sim 侧做，小幅度正弦安全。
- relay 输出相机话题 `/cam_{right,left,head,base}/color/image/compressed` 与
  `cameras_config_arena_30fps.yaml` 的 rgb_topic 完全一致；相机 fps=30 与 dataset.fps=30 一致。

## 产物清单（本目录）

| 文件 | 说明 |
|---|---|
| run.yaml | GUI「启动仿真」等价配置（start_sim.sh --config 用） |
| capture.yaml | GUI「开始采集」等价配置（start_capture.sh --config 用，A6000 路径） |
| leader_driver.py | 模拟 VR 命令流发布节点（补缺环，A6000 系统 python3 + humble，py_compile 过） |
| run_ros2_direct_accept.sh | 一键补测：GPU 检查→kit→rosbridge→driver→topic 自检→record→stop→列产物 |
| ros2-direct-accept.md | 本文件 |

## 命令文本（dry-run 实测输出）

start_sim.sh --dry-run 生成（remote 等价；实测用 A6000 本地变体，state/camera/command 全部
127.0.0.1，规避跨机 UDP——唯一偏差，已固定在脚本里）：

```
bash scripts/run_native_isaac.sh scripts/teleop_vr_openflex.py --headless --device cuda:0 \
  --record-dir /tmp/mrs-accept/arena_datasets --render-hz 30 --physics-hz 120 \
  --command-host 127.0.0.1 --state-host 127.0.0.1 --camera-stream-host 127.0.0.1 \
  --control-host 127.0.0.1 --control-port 24105 --record-image-hz 15.0 --episodes 1 \
  --enable_cameras --camera-stream-port 24104 --data-mode ros2_direct --seed 7
```

start_capture.sh --dry-run 生成（与 GUI SimulationTaskCommandBuilder.build_record 生成一致）：

```
/home/1024201092WYH/.conda/envs/lerobot/bin/python \
  …/openflex_ws/src/openflex_vla/lerobot_scripts/lerobot_record_openflex.py \
  --schema_config=…/no_chassis_lio_act_vel_base.yaml \
  --camera_config=…/cameras_config_arena_30fps.yaml \
  --robot.type=openarmx_follower_ros2 --robot.skip_send_action=true \
  --teleop.type=openarmx_leader_ros2 --teleop.schema_config=…schema.yaml \
  --teleop.ros2.command_timeout_s=1.0 --teleop.ros2.connect_timeout_s=60.0 \
  --teleop.ros2.require_live_commands=true --teleop.ros2.require_non_arm_commands=true \
  --teleop.ros2.require_both_arm_commands=true \
  --dataset.repo_id=direct/ros2_accept --dataset.root=…/datasets/lerobot/ros2_accept_demo \
  --dataset.num_episodes=1 --dataset.fps=30 --dataset.episode_time_s=15 --dataset.reset_time_s=5 \
  --dataset.single_task='pick up the red block' --resume=false
```

代码一致性（SHA256 前 12 位，本地 vs A6000 ~/mrs_robot/MRS_ROBOT）：11/12 OK，唯一差异
`scripts/start_sim.sh`——本地多出 enable_lidar 时补传 --enable_cameras 的 4 行（修复项 A
agent 本地改动未同步 A6000；A6000 该文件为 untracked 旧版）。本任务不依赖该差异。

## 潜在风险点（补测时先看这几处）

1. rclpy 与 conda lerobot env：`.conda/envs/lerobot/bin/python` 无 rclpy；start_capture.sh
   先 source /opt/ros/humble/setup.bash（PYTHONPATH 注入 humble python3.10 site-packages，
   conda 亦 3.10，ABI 兼容）。这是 GUI 既有行为；若 import 失败看 record.log 首屏。
2. openarmx_integrated_bringup 集成 VR launch 在无头 + 无 Pico 下应空转不退出；若
   arena_vr_teleop launch 提前退出看 ros.log，可临时仅起 relay node 验证。
3. mp4 编码走 LeRobot save_episode；缺 ffmpeg 之类依赖要到落盘时才暴露，看 record.log 尾部。

## 补测执行（待修复项 A 仿真停止后）

A6000:
```
bash ~/mrs_robot/MRS_ROBOT/MRS_ROBOT_sim/.hermes/ros2-direct-accept/run_ros2_direct_accept.sh
```
验收标准：`datasets/lerobot/ros2_accept_demo` 下 1 集 parquet + mp4 + meta 齐全，脚本末尾
自动列文件清单；`VALIDATE_ONLY=1` 只拉链路自检不录。
