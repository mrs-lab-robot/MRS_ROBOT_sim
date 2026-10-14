> 落位说明：任务书指定 `MRS_ROBOT_sim/.hermes/gt-fullchain-accept.md`，该路径受 harness 保护钩子拦截（无人值守会话无法批准写入，项目根 `.hermes/` 同样被拦）。按最接近的可写位置落在本目录（test/），交互会话下一条命令归位：
> `mv /media/y/workdisk/project/MRS_ROBOT/MRS_ROBOT_sim/test/gt-fullchain-accept.md /media/y/workdisk/project/MRS_ROBOT/MRS_ROBOT_sim/.hermes/gt-fullchain-accept.md`

# ground_truth 全链验收（无雷达降级版）交付

状态：**已完成(无雷达降级)·全链 PASS（RESULT OK / RC=0）**
验收时间：2026-10-14 01:13–01:21（A6000），01:24 终验收口
执行机：A6000 = ssh 6000PRO-WYH（RTX 6000 Ada 48G；开工前 84 MiB / 0%，收工后 84 MiB / 0%，现场干净）
代码基线：/home/1024201092WYH/mrs_robot/MRS_ROBOT/（MRS_ROBOT_sim + openflex_ws），**未提交 git**
参照：项目根 `.hermes/hdf5-primary-capture-delivery.md`（2026-10-13 phase3 成功全链）
范围：ground_truth 数据模式、--enable_cameras、**不开雷达**（雷达首帧三轮修复无果，凌晨搁置，不阻塞主线）

## 一、结论（先看这里）

无雷达 ground_truth 全链**一次修复后全绿**：起仿真 → 2 集 HDF5（含 LIO 真值字段）→ 数据管理页同款两阶段转换 → LeRobot 25 维（含 LIO 6 维）→ 结构校验 RESULT OK。验收过程中揪出并修复 **data-mode 重构引入的 LIO 真值回归**（详见第四节），该修复是本轮最重要的副产物：不修则"真值+噪声域随机化"主线路线在 ground_truth 模式下采不出 LIO 字段，25 维 schema 直接不可用。

## 二、T1 起仿真

```bash
# 编排：/tmp/mrs-gt-nolidar-r2/run_all.sh（复用昨晚 21:22 版，without-lidar 分支）
cd $M/arena_benchmark   # M=/home/1024201092WYH/mrs_robot/MRS_ROBOT/MRS_ROBOT_sim
env -u LD_LIBRARY_PATH -u CUDA_HOME -u ROS_DISTRO -u PYTHONPATH OMNI_KIT_ACCEPT_EULA=YES \
  bash scripts/run_native_isaac.sh scripts/teleop_vr_openflex.py \
    --headless --device cuda:0 --render-hz 30 --physics-hz 120 --record-image-hz 10 \
    --record-dir $RUN/episodes --control-host 127.0.0.1 --control-port 24105 \
    --enable_cameras --camera-stream-port 24104 \
    --data-mode ground_truth --synthetic-vr        # 无 --enable-lidar
```

- ARENA_READY：01:13:30（RUN_START 后 17s；昨晚 with-lidar 同链路死于雷达传感器初始化，无雷达无此环节）
- GPU 峰值：8,578 MiB / 39%（r1 首跑实测；r2 同量级）
- 遗留说明：A6000 有两个昨日遗留 ROS 节点（openarmx_teleop_vr_node / head_teleop_node，PID 229462/229464，无 GPU 占用）；本链路 synthetic-vr 不经 ROS，判定惰性未清理，收工仍在（见第六节遗留 4）

## 三、T2 采集 2 集（episode 级 API）

```
GET  http://127.0.0.1:24105/api/v1/status        # ready:true
POST http://127.0.0.1:24105/api/v1/episode/start # 01:15:00 ep1
POST http://127.0.0.1:24105/api/v1/episode/save  # 01:17:00（120s 窗）
POST .../episode/start                           # 01:17:15 ep2
POST .../episode/save                            # 01:19:15（120s 窗）
POST http://127.0.0.1:24105/api/v1/session/stop  # 01:19:30
```

产物（/tmp/mrs-gt-nolidar-r2/episodes/）：

| 文件 | 步数 | 大小 |
|---|---|---|
| openflex_vr_demo_20261014_011700_174889_0001.hdf5 | 878 | 21,258,952 B |
| openflex_vr_demo_20261014_011915_159694_0002.hdf5 | 894 | 21,575,289 B |

session/stop 后 SIM_EXITED code=1（phase3 同款清理期噪音：RenderProduct/Hydra/Event.count，不阻塞，进程本体退出，GPU 回 84 MiB）。

## 四、本轮关键修复：LIO 真值回归（data-mode 重构引入）

**现象（r1 首跑，01:03，失败现场保留在 /tmp/mrs-gt-nolidar/）**：2 集 HDF5 一切正常唯独缺 `lio_pose/lio_velocity`，verify FAIL；转换链 prepare 阶段被拒：`ValueError: Arena HDF5 缺少 LIO 字段…无法按 25 维 schema 转换`。

**根因**（arena_benchmark/scripts/teleop_vr_openflex.py）：

```python
# 686 行（录制闸门）——重构前 phase3 时代 LIO 无条件记录，data-mode 重构后：
if args.data_mode == "ground_truth" and args.enable_lidar:   # ← 错误耦合
    lio_pose_before, lio_velocity_before = _lio_state(robot, env)
else:
    lio_pose_before, lio_velocity_before = None, None
# 432-436 行（session_metadata.lio_source）同病：arena_ground_truth 要求 enable_lidar
```

`_lio_state()` 只读 `robot.data.root_pos_w / root_quat_w / root_lin_vel_b / root_ang_vel_b`（仿真根状态真值），**与雷达完全无关**。修复＝解开 enable_lidar 耦合（两处 `and args.enable_lidar` 删除）。

**部署**：本地 dev copy 与 A6000 同步修复（远端用精确替换+断言 patch，不动远端其他脏改；远端改前快照备份 `.pre-gtlio-fix`，双侧 `py_compile` 通过，本地/远端 grep 语义比对 SEMANTIC_MATCH）。**未提交 git**。

**回归验证（r2）**：lio 字段恢复且逐步对齐、有限、有位移（ep1 first=[-0.0778, 0.0393, 0.2324] → last=[-0.1465, 0.0680, 0.2609]）。

## 五、T3 LeRobot 25 维转换（数据管理页 build_append_batch 两阶段）

```bash
# 阶段1 prepare（Arena venv Python 3.12）
$M/arena_benchmark/.venv/bin/python $M/arena_benchmark/scripts/export_arena_hdf5_to_lerobot.py \
  --source <ep.hdf5> --prepare-only --prepared-output $RUN/staging/episode_000N \
  --fps 10 --state-dim 25 --task "ground_truth 验收集"
# 阶段2 append（LeRobot conda Python）
/home/1024201092WYH/.conda/envs/lerobot/bin/python $CONVERT \
  --prepared-dir $RUN/staging/episode_000N --append-prepared \
  --output-root $RUN/lerobot25 --repo-id openflex/gt_acceptance_25d \
  --fps 10 --state-dim 25 --task "ground_truth 验收集"
```

- 实测：PREPARE_OK×2（各 ~4s）、APPEND_OK×2（各 ~25s）、`ARENA_LEROBOT_EXPORT_OK path=.../gt_acceptance_25d`
- 与 phase3 坑位对照：冗余 mkdir 冲突（R1）与 `_feature_signature` 过严（R2）已于 10-13 17:13 append-fix 修复并在 A6000 验证；本轮零适配直通，二集追加一次成功——**R1/R2 修复在 ground_truth 新数据上复验通过**
- 产物（/tmp/mrs-gt-nolidar-r2/lerobot25/openflex/gt_acceptance_25d/，共 10M）：
  - `data/chunk-000/file-000.parquet`、`file-001.parquet`（每集一个）
  - `videos/observation.images.{base,head,left_wrist,right_wrist}/chunk-000/file-00{0,1}.mp4`（4 路 × 2 集 = 8）
  - `meta/`：info.json、stats.json、tasks.parquet、mrs_episode_manifest.json、episodes/chunk-000/file-00{0,1}.parquet
  - repo 级：`.gt_acceptance_25d.mrs-episode-ledger.json`、`.mrs-export.lock`

## 六、T4 schema 与 phase3 差异

**HDF5 schema（format=mrs_robot_capture_v1，两集一致）**：

| 字段 | shape | dtype |
|---|---|---|
| sim_time_ns / step_index / command_seq | (878/894,) | uint64/int64/uint64 |
| action / applied_target / operator_command | (n,22) | float32 |
| joint_position / joint_velocity / next_joint_position | (n,19) | float32 |
| **lio_pose / lio_velocity** | **(n,3)** | float32 |
| raw_command_json | (n,) | object |
| camera_frames/{base_d435,head_d435,left_wrist_d405,right_wrist_d405}/jpeg | 439/447 帧 | jpeg bytes |
| attrs: metadata_json 内 data_mode=**ground_truth**、lio_source=**arena_ground_truth** | | |

**LeRobot 25 维 features**（meta/info.json：fps=10、codebase v3.0、total_episodes=2）：observation.state shape [25]，names = 左臂7+j指+右臂7+j指+头2+升降1（19 维）+ **chassis.lio_odom_x/y/theta + chassis.lio_vx/vy/wz（LIO 6 维）**；observation.images.{base,head,left_wrist,right_wrist}；action 维度同源 schema。

**与 phase3（20261013_165038/165055）差异表**：

| 维度 | phase3 | 本轮 r2 | 判读 |
|---|---|---|---|
| data_mode | 无此参数（重构前） | ground_truth | 新链路正确落 metadata |
| lio 字段 | 有 (121/122,3) | r1 **缺失→修复后** (878/894,3) | **本轮发现并修复的回归**，其余结构零差异 |
| 步数/集 | 121/122（~17s 短采，~7.3Hz） | 878/894（120s 满窗，~7.3Hz） | 步频一致，时长策略不同 |
| 相机帧/集/路 | 3 | 439/447（约 3.7Hz，每 2 步 1 帧） | r2 落帧密度显著更健康 |
| 文件大小 | 2.7MB/集 | 21.3/21.6MB/集 | 随步数/帧数同比例放大 |
| LeRobot 布局 | chunk 嵌套（v3.0） | 逐文件同构 | 转换链行为一致 |
| 25 维 schema | 同（含 LIO 6 维） | 同 | 无漂移 |

**verify 判据勘误**：verify_fullchain.py 原 glob `data/*.parquet`、`videos/*/*.mp4` 层级写错（LeRobot v3.0 实际落 `data/chunk-000/`、`videos/<cam>/chunk-000/` 嵌套，phase3 文件树为证）——数据无错，判据已修（`chunk-*` 层级），修后 **RESULT OK / RC=0，全判据 PASS**。

## 七、遗留（白天接手清单）

1. **LIO 真值修复待评审入库**：teleop_vr_openflex.py 两处（686 行录制闸门、432 行 lio_source），本地+A6000 已生效、未提交 git；远端备份 `teleop_vr_openflex.py.pre-gtlio-fix`。建议日间评审后提交（这是主线数据有效性的前置）。
2. verify_fullchain.py glob 勘误已同步本地 `/tmp/mrs-gt-accept-src/` 与 A6000 两个 run 目录；该验收脚本建议从 /tmp 收进仓库 `test/`。
3. stop 后清理期日志噪音（RenderProduct/Hydra error 6、IsaacLab Event.count、exit code=1）沿用 phase3 遗留 #3，不阻塞，未处理。
4. A6000 两个昨日遗留 ROS 节点（PID 229462/229464）惰性存活，需要时 `kill 229462 229464`。
5. 交付文档位置偏差：任务书目标 `.hermes/` 被保护钩子拦，本文档落 `test/`，待交互会话 mv 归位。
6. 雷达（MID360 首帧）保持搁置，不影响本轮全链；enable_lidar 打开时 lio_source/录制闸门行为与修复前一致（lidar_lio 模式未动）。

## 附：过程日志

- 本机会话：`MRS_ROBOT_sim/.hermes/gt-finite.log`
- A6000 成功现场：`/tmp/mrs-gt-nolidar-r2/`（status.txt / arena.log / convert.log / verify_h5_and_lerobot.log / verify_final.log=RESULT OK）
- A6000 失败现场（r1，LIO 回归证据）：`/tmp/mrs-gt-nolidar/`
- 昨晚 with-lidar 失败现场（对照）：`/tmp/mrs-gt-accept/`
