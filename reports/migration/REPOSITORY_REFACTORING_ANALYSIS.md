# MRS_ROBOT_sim 分层重构分析

> 阶段：迁移前基线（2026-10-01）。此报告只记录当前实现和迁移边界；运行行为以 ROS/Isaac Sim 集成验收为准。

## A. 当前真实目录与职责

| 当前路径 | 当前内容 | 目标层 |
| --- | --- | --- |
| `isaac_sim_core/assets/` | canonical 机器人 USD、环境 stage、传感器与纹理资源 | `sim_runtime/assets/` |
| `isaac_sim_core/components/` | MID360 语义处理、控制器和工具代码 | `sim_runtime/sensors/`、`sim_runtime/components/` |
| `isaac_sim_core/config/` | 物理、渲染和传感器配置 | `sim_runtime/config/` |
| `isaac_sim_core/scenarios/` | 相机、激光雷达复制场景脚本 | `sim_runtime/scenarios/` |
| `ros2_pkgs/openflex_isaac_sim/` | description、contract、controllers、sensors、bridge、bringup 六个 ROS 2 包 | `sim_runtime/ros2/openflex_isaac_sim/` |
| `arena/src/mrs_robot_arena/` | resolver/contract、Arena embodiment、smoke env、teleop、录制和 runner 混合包 | 拆入 `isaaclab_ext/` 与 `arena_benchmark/` |
| `arena/integrations/ros2/` | VR 命令 ROS relay ROS 包 | `sim_runtime/ros2/` |
| `arena/third_party/isaaclab-arena/` | 固定到 `8737b4ceb25f99f81a81786b7fde73139b52f324` 的上游子模块 | `third_party/IsaacLab-Arena/` |
| `arena/configs/runtime/versions.yaml` | 全局 Isaac/ROS/Python 版本锁 | `configs/versions.yaml` |
| `test/`、`docs/`、`reports/`、`tools/` | 测试、文档、运行证据和资产工具 | 根级对应目录，测试统一为 `tests/` |

当前尚无独立 `isaaclab_ext`。Arena 的 `openflex.py` 同时声明 Isaac Lab articulation/action 配置并继承 Arena `EmbodimentBase`、注册 Arena asset；`openflex_smoke.py` 直接构造 `IsaacLabArenaEnvironment`，因此目前不能独立于 Arena 做 Lab smoke。

## B. 当前依赖关系

```text
ROS 2 Runtime (Humble / Python 3.10)
  ├── openflex_isaac_bringup → bridge, description, sensors, controllers, contract
  ├── openflex_isaac_sensors / controllers / bridge → rclpy 与 ROS 消息/控制包
  └── arena ROS relay → rclpy + UDP/VR 命令聚合

Arena Runtime (Isaac Sim 6 / Isaac Lab 3 / Python 3.12)
  ├── mrs_robot_arena.embodiments.openflex → Isaac Lab + IsaacLab-Arena + resolver + contract + swerve config
  ├── environments.openflex_smoke → IsaacLab-Arena（当前 smoke 的耦合点）
  ├── teleoperation / recording / runners → 部分纯 Python，部分使用 Torch/Kit/Isaac Lab
  └── AssetResolver → 仓库内 USD、ROS contract YAML、sensor YAML、base-controller YAML
```

机器人接口唯一可复用的数据源是 `openflex_isaac_contract/config/embodiment.yaml`；传感器 topic/安装和底盘控制配置分别在运行时传感器 YAML、bringup controller YAML。Arena 的 Python contract loader 是读取/校验代码，不应再成为一份独立关节/传感器定义。

主要路径依赖包括 `AssetResolver`、ROS launch 的 `ISAACSIM_ROBOT_ROOT` 根目录探测、传感器 integration、description/contract 校验脚本、性能脚本、迁移/集成测试、启动 README，以及外部 GUI 当前约定的 `arena` 与 `ros2_pkgs` 路径。迁移时保留旧路径兼容入口，再逐步将仓库内调用改为新路径。

ROS 包内部依赖以 `package.xml` 为准：bringup 依赖本仓库五个 ROS 包及 `isaac_ros2_scripts`、`topic_based_ros2_control`、`swerve_controller`、Livox 等 underlay；bridge 依赖 rclcpp/rclpy 和标准消息、TF、message_filters；controllers 依赖 ros2_control/controller_interface 等。Arena policy 的逐步控制不应进入 ROS relay。

## C. 文件职责分类

| 分类 | 当前文件/目录 |
| --- | --- |
| `SIM_RUNTIME` | `isaac_sim_core/`、六个 `openflex_isaac_*` ROS 包、机器人/传感器/控制器/bringup 配置、Isaac Sim 场景脚本 |
| `ISAACLAB` | 尚未独立；当前 `arena/src/mrs_robot_arena/assets/resolver.py`、Isaac Lab articulation/action 配置、smoke env、joint teleop、episode recorder、VR→action adapter 中有一部分应下沉 |
| `ARENA` | Arena `EmbodimentBase`/asset 注册与组合、benchmark/task/scene/predicate/variation/policy/evaluation 扩展；当前只具备 embodiment/smoke 起步代码，未见完整工业任务套件 |
| `SHARED` | ROS contract YAML、sensor YAML、controller YAML、版本锁、协议边界与资源解析 API；数据只保留一份，解析器可以分层封装 |
| `THIRD_PARTY` | `arena/third_party/isaaclab-arena/` 子模块；禁止改上游文件 |
| `UNKNOWN / 本地生成` | `build/`、`install/`、`log/`、`.deps/`、`arena/outputs/`、缓存和本地数据；不迁入源码层、不纳入 Git |

## D. 路径迁移映射

| 旧路径 | 新路径 |
| --- | --- |
| `isaac_sim_core/assets/**` | `sim_runtime/assets/**` |
| `isaac_sim_core/components/sensors/**` | `sim_runtime/sensors/**` |
| `isaac_sim_core/components/{controllers,utils}/**` | `sim_runtime/components/{controllers,utils}/**` |
| `isaac_sim_core/config/**` | `sim_runtime/config/**`（`sensor_params` 归一为 `sensors`） |
| `isaac_sim_core/scenarios/**` | `sim_runtime/scenarios/**` |
| `ros2_pkgs/openflex_isaac_sim/**` | `sim_runtime/ros2/openflex_isaac_sim/**` |
| `arena/src/mrs_robot_arena/assets/resolver.py` | `isaaclab_ext/src/mrs_robot_lab/assets/asset_resolver.py` |
| `arena/src/mrs_robot_arena/embodiments/{actions,swerve}.py` 与 articulation 配置 | `isaaclab_ext/src/mrs_robot_lab/`；Arena embodiment 只组合/注册 Lab 接口 |
| `arena/src/mrs_robot_arena/environments/openflex_smoke.py` | `isaaclab_ext` 中不导入 Arena 的 smoke 入口 |
| `arena/src/mrs_robot_arena/teleoperation/` | 协议/VR transport 下沉 `sim_runtime/teleoperation/`；命令到 Lab action 的 mapper 留在 Lab adapter |
| `arena/src/mrs_robot_arena/recording/`、Lab joint runner | `isaaclab_ext/src/mrs_robot_lab/recorders/`、`isaaclab_ext/scripts/` |
| `arena/src/mrs_robot_arena/embodiments/` 中 Arena 注册/包装 | `arena_benchmark/src/mrs_arena/embodiments/` |
| `arena/integrations/ros2/mrs_robot_arena_bridge/` | `sim_runtime/ros2/mrs_robot_arena_bridge/` |
| `arena/configs/runtime/versions.yaml` | `configs/versions.yaml`；任务/场景实验配置留在 `arena_benchmark/configs/` |
| `arena/third_party/isaaclab-arena/` | `third_party/IsaacLab-Arena/` |
| `test/**`、`arena/tests/**` | 根 `tests/sim_runtime/`、`tests/isaaclab_ext/`、`tests/arena_benchmark/`；迁移期保持旧命令可用 |

## E. 风险及处理

1. **路径硬编码多**：场景、ROS launch、sensor integration、安装布局测试、性能脚本和 README 使用旧根目录；更新内部路径解析并为旧目录提供兼容符号链接/包装入口。
2. **双 Python/ROS 环境**：ROS Humble 使用系统 Python 3.10；Isaac Sim 6/Arena 使用 Python 3.12。不可在一个环境中合并依赖或让逐步 Arena action 通过 ROS。
3. **USD 引用**：机器人资产只有 `openflex_robot.usda` 一份。迁移目录时须验证 USDA 内部相对 mesh/material 引用未断，并让 Lab Resolver 指向该 canonical 文件。
4. **contract/接口重复**：保留 ROS contract YAML 为关节/动作/观测唯一数据源；Arena/Lab 只解析，不复制列表或关节限位。
5. **子模块元数据已迁移**：Arena 子模块从 `arena/third_party/isaaclab-arena` 移至根级 `third_party/IsaacLab-Arena`，保留 commit `8737b4ceb25f99f81a81786b7fde73139b52f324`；上游工作树保持未修改。
6. **外部 launch/GUI 兼容**：`ISAACSIM_ROBOT_ROOT`、`MRS_ROBOT_SIM_ROOT` 和 GUI 对 `arena`、`ros2_pkgs` 的路径合同需兼容，避免要求同步修改另一个仓库。
7. **生成物与用户改动**：build/install/log/.deps/output 不属于源码迁移；仓库现有未提交变更均按用户内容保留，不做清理或重置。
8. **验证环境**：系统 Python 3.10 与 ROS Humble 保持隔离；Arena 在本机 `arena_benchmark/.venv` 使用 Python 3.12 和上游锁文件。全局 `PYTHONPATH` 指向另一工作区的 Lab/Arena 版本，native launcher 必须使用项目 venv 和本仓库包路径。

## 分阶段执行顺序

1. 先添加不依赖仿真器的架构/资源路径合同测试。
2. 建立根级三层目录及统一版本配置；保持资产与接口数据唯一。
3. 迁移 `sim_runtime`，更新 ROS 构建和 launch 路径，并维持兼容入口。
4. 提取独立 `mrs_robot_lab`：canonical asset resolver、robot config/action、sensor/observation 接口、无 Arena smoke、teleop adapter/recorder。
5. 把 Arena 侧保留为场景/任务/embodiment/评测组合，迁出 ROS bridge 和通用 VR 协议；更新子模块路径。
6. 更新文档、验证脚本及 GUI 所需兼容合同；分别执行纯 Python、ROS/colcon 和锁定 Isaac Sim/Arena 验收。仿真运行验收若本机缺少目标 runtime，将明确保留为未运行项，不以静态测试冒充通过。

## F. 本轮实施结果与验收边界（2026-10-01）

| 层 | 当前职责/实现 | 已执行的验证 |
| --- | --- | --- |
| `sim_runtime/` | 现有 USD、场景/传感器配置、ROS 2 包、VR 协议与 relay 的 canonical 位置；旧 GUI/launch 路径由符号链接兼容 | 6 个 runtime ROS 包 + VR relay 共 7 个包通过隔离目录中的 `colcon build` |
| `isaaclab_ext/` | 独立 `mrs_robot_lab` 包：读取 runtime contract/USD，提供 Articulation、action/observation、传感器 smoke、关节控制、teleop adapter 与 recorder | 源码级测试通过；真机 Kit 完成 19 关节控制、22 维动作契约及 4 路 RGB/depth 相机 smoke。回归覆盖仿真 reset 后再查询 articulation joints，以及将 Warp dtype 转成 PyTorch `float32` |
| `arena_benchmark/` | `mrs_arena` 只组合 Lab embodiment 与 Arena Scene/Task；旧 `mrs_robot_arena` import 保留为 wrapper | 真机 CUDA no-task/action smoke 通过；`OpenFlexSmokeTask` 注册、成功终止和指标/episode 记录 smoke 通过（1 episode，success_rate=1.0） |
| 资产/版本 | USD 唯一放在 `sim_runtime/assets/robots/`；机器人关节、动作和观测映射由 runtime embodiment contract 解析；依赖锁在根 `configs/versions.yaml` | 迁移结构测试验证无重复 USD、旧入口解析到 canonical 路径、下层不导入上层 |

本机宿主机验证记录：检查确认已有 Isaac Sim 6.0 runtime（`/media/y/workdisk/sim/isaacsim-6.0/runtime`），并依上游 `uv.lock` 在仓库工作盘创建隔离的 `arena_benchmark/.venv`，安装 277 个锁定包及四个本仓库 editable 包；Python 3.12.13、Isaac Sim 6.0.0.1、Isaac Lab 3.0.0b2、Arena 0.3.0、Torch 2.11.0+cu128，CUDA 可用且识别 RTX 4060 Ti。venv 占用约 28 GB、uv 缓存约 1.1 GB，均位于 `/media/y/workdisk`（当次检查可用空间约 180 GB），未将大型环境放到系统盘。操作者已明确接受 NVIDIA Omniverse EULA；真机 Kit smoke 使用 `OMNI_KIT_ACCEPT_EULA=YES` 启动，不在仓库中默认写入接受状态。

真机验收结果（2026-10-01，RTX 4060 Ti / CUDA）：`test_joint_control.py --steps 2` 输出 `MRS_ROBOT_LAB_SMOKE_OK`；`spawn_robot.py --validate-actions --steps 2` 验证 19 个关节、22 维动作契约及 `base_d435`、`head_d435`、`left_wrist_d405`、`right_wrist_d405` 四路相机 RGB/depth；Arena no-task smoke 输出 `OPENFLEX_ARENA_SMOKE_OK`，完成底盘/左臂安全探测及 2 步零动作；任务 smoke 输出 `OPENFLEX_ARENA_TASK_SMOKE_OK`，记录一个成功 episode 且 success_rate 为 1.0。合并运行 `tests/`、`isaaclab_ext/tests/` 与 `arena_benchmark/tests/unit/` 的仿真器无关测试：**59 passed, 197 subtests passed**。7 个 ROS 包的 `colcon build` 结果见本轮前序验证记录。Kit 有 inotify watch、Kit 用户目录和 PhysX 等非阻断告警；以上场景均正常退出并打印成功标记。

Arena 上游 override 将 NumPy 固定为 2.3.1、websockets 固定为 16.1.1，因此通用 `pip check` 会报告 `openpi-client` 与 Isaac Sim kernel 的两条元数据版本冲突；这是上游 `pyproject.toml` 明确记录的组合，不应擅自降级。

本轮恢复后的真机补充验收（2026-10-01，Isaac Sim 6.0 / RTX 5060 Laptop GPU）：本机 shell 曾加载 `/home/y/Robot/OpenFlex_ws/setup_remote_90.bash`，其中 FastDDS 默认 profile 只配置远端初始 peer 且关闭内置传输；本地仿真验收需清除 `FASTRTPS_DEFAULT_PROFILES_FILE`，并让 Isaac Sim 与 ROS 2 CLI 使用同一 domain。按本机闭环方式重启后，`/clock`、`/joint_states` 和底盘控制话题可见。真实导入 stage 的头部相机挂载路径已修正为包含 `lift_base_link`；底盘和头部相机可同时进入 `active`，对应 RGB/depth/CameraInfo 话题出现，实测 RGB 约 20–24 Hz。卸载底盘相机时发现直接删除 RenderProduct USD prim 会令 Hydra 下一渲染周期崩溃；现改为先停发布和采样、等待 Kit 更新，再通过 Replicator RenderProduct 所有者 API 释放 Hydra/SyntheticData 资源。修复后真机 create→并发发布→销毁底盘→保留头部→销毁头部流程完成，无 Hydra 崩溃；两路相机停用后 `ros2 topic hz` 均未收到新图像，仿真由 launch 正常停止。传感器生命周期单测 **15 passed**；隔离 ROS 环境变量并允许 loopback socket 后，`tests/`、`isaaclab_ext/tests/` 与 `arena_benchmark/tests/unit/` **61 passed, 197 subtests passed**。`sim.launch.py` 本机运行说明也补充了清除远端 FastDDS profile 的提示。

Arena 相机观测补充（2026-10-01，本机 Isaac Sim 6.0 / RTX 5060 Laptop GPU）：修复 `make_openflex_smoke_environment()` 未暴露相机开关、OpenFlex Arena Embodiment 未挂载 Arena camera config 的缺口；配置直接从 `sim_runtime/config/sensors/realsense/realsense_robot_mounts.yaml` 构建，不复制 mount/interface 定义。新增 `smoke_openflex_cameras.py`，在 Arena observation manager 中实际收到四路 640×480 RGB 与 `distance_to_image_plane` 张量，输出 `OPENFLEX_ARENA_CAMERA_SMOKE_OK cameras=4 modalities=rgb, distance_to_image_plane steps=1`。相机默认关闭。修复后本机无相机 Arena smoke 输出 `OPENFLEX_ARENA_SMOKE_OK`，任务/指标 smoke 输出 `OPENFLEX_ARENA_TASK_SMOKE_OK`（1 episode，success_rate=1.0）；仿真器无关快速回归 **59 passed, 197 subtests passed**。该验证不代表 ROS 图像发布、VR HDF5 图像录制、工业任务套件或完整 Benchmark 已完成。

桌面方块任务与无 VR 验收补充（2026-10-01，本机 Isaac Sim 6.0 / CUDA）：新增注册任务 `OpenFlexPickCubeTask`，使用 Arena 程序化刚体桌面/200 g 方块，并以相对初始位置抬升 25 cm 作为成功条件；场景含独立可视桌面外观与任务物理碰撞体。`smoke_openflex_pick_cube.py` 不连接 VR、ROS relay 或真机：先检查方块与桌面接触，再验证多个零动作步骤不会成功，最后仅将方块仿真状态注入目标位置，以验收成功谓词和 success_rate。真机 smoke 输出 `OPENFLEX_PICK_CUBE_MDP_SMOKE_OK ... table_cube_contact_gap_m=0.0000 success_rate=1.0 goal_check=state_injection physical_grasp=not_tested`。因此通过的是物理场景装配和 MDP 谓词，不是机器人接近、夹持或抬升能力；后续仍需无 VR 的机器人动作/抓取策略验收。测试脚本使用 Isaac Lab 当前的 ProxyArray `.torch` 访问及 index 写状态 API，避免弃用接口。Kit 输出 inotify 和不可写用户目录告警，但仿真仍正常结束。
