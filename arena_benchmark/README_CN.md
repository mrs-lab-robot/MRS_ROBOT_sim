# Arena 扩展

这是 `MRS_ROBOT_sim/arena_benchmark`，负责通过 Isaac Lab-Arena 组合 OpenFleX Embodiment、Scene、Task 和 Benchmark。机器人运行时与 Isaac Lab 适配分别由同仓库的 `sim_runtime/`、`isaaclab_ext/` 负责；真机安全控制仍由真机仓库负责。

English documentation: [README.md](README.md)

## 仓库边界

| 仓库 | 负责内容 |
| --- | --- |
| MRS_ROBOT | 真实机器人应用、ROS 2 GUI/业务流程、部署和系统集成 |
| MRS_ROBOT_sim | Isaac Sim + ROS 2 仿真运行时、Arena 扩展、OpenFleX USD、传感器参数与机器人接口契约 |

本目录通过 `MRS_ROBOT_SIM_ROOT` 读取本仓库中的权威 USD 和接口契约；不复制、不修改这些资源。Arena policy step 直接控制仿真 articulation，不经 ROS 转发每步动作。

## 版本基线

当前目标/锁定组合如下；是否可运行以本机对应版本的集成测试结果为准。完整锁定信息见 [全局版本锁](../configs/versions.yaml)。

| 组件 | 版本/提交 |
| --- | --- |
| Python | 3.12 |
| Isaac Sim | 6.0.0.1（6.0 系列） |
| Isaac Lab | 3.0.0b2（源码子模块提交见版本锁） |
| IsaacLab-Arena | release/0.3.0，提交 8737b4ceb25f99f81a81786b7fde73139b52f324 |

Arena 上游源码以 Git 子模块固定在仓库根目录 `third_party/IsaacLab-Arena`，该目录只读；项目代码位于 `src/mrs_arena`。`mrs_robot_arena` 仅作为旧导入路径兼容 wrapper。

## 当前实现

- AssetResolver：从统一仓库根目录定位 openflex_robot.usda、embodiment.yaml 和传感器参数文件。
- contracts.py：在不启动 Isaac Sim 的情况下解析并校验机器人接口契约，包括动作维度、观测维度和有效关节动作集合。
- OpenFlexEmbodiment：Arena 原生移动操作机器人 Embodiment，配置 OpenFleX USD articulation、底盘/双臂/升降柱/头部/二值夹爪动作，以及关节位置/速度观测。
- 机器人 USD 的驱动参数继续以 USD 中写入的 PhysX drive 属性为准，避免在 Arena 仓库维护第二套动力学参数。

当前 Embodiment 提供机器人接入与基础仿真控制，不等同于完整训练任务。22 维动作按顺序为底盘 Twist(3)、左臂(7)、右臂(7)、升降柱(1)、头部(2)、左右夹爪(各 1)。自定义 Arena ActionTerm 从同一仓库根目录的底盘控制器 YAML 读取四轮几何、轮半径、限速和加速度限制，将 Twist 原生映射到仿真转向/轮速关节，不发布 ROS 指令。No-task smoke 环境可以加载/reset/step 并检查观测；键盘入口支持底盘和全身关节的基础遥操作。Arena 已可选接入四路相机 RGB 与 `distance_to_image_plane` 深度 policy observation（`camera_obs` 组）；这不表示图像已写入 VR episode HDF5。现有首个注册任务是桌面方块抬升 25 cm，具备物理桌面/方块和成功谓词，但验收 smoke 用仿真状态注入检查谓词，不代表机械臂已能完成真实抓取。工业任务套件、训练配置和完整 Benchmark 仍未实现。

## 目录结构

    MRS_ROBOT_sim/arena_benchmark/
    ├── ../third_party/IsaacLab-Arena/    # 根级上游 Arena 子模块，只读
    ├── configs/
    │   ├── runtime/                      # Isaac Sim/Lab/Arena 版本锁
    │   ├── environments/                 # 环境组合配置
    │   ├── experiments/                  # 训练/实验配置
    │   └── benchmarks/                   # 评测套件配置
    ├── src/mrs_arena/
    │   ├── embodiments/openflex.py       # 复用 mrs_robot_lab 的机器人配置并注册 Arena
    │   ├── scenes/                       # Arena 场景扩展
    │   ├── tasks/                        # TaskBase/MDP 任务实现
    │   ├── policies/                     # 策略与算法适配
    │   ├── learning/                     # 训练配置/入口
    │   ├── runners/                      # 训练、评估、遥操作运行器
    │   └── benchmarks/                   # 指标、协议和注册
    ├── tests/
    │   ├── unit/                         # 不启动 Isaac Sim 的快速测试
    │   ├── integration/                  # 目标 Arena 容器中的集成测试
    │   └── acceptance/                   # 端到端验收
    └── docs/

各层职责应保持清晰：资产描述“机器人/物体是什么”，Embodiment 描述“机器人如何进入 Arena、有哪些动作和观测”，Scene 描述“仿真世界是什么”，Task/MDP 描述“目标和奖励是什么”，Benchmark 描述“如何可复现地比较结果”。

## 检出与安装

Docker 不是必须的。Arena 0.3.0 支持原生 Linux `uv` 安装与 Docker 两种方式；原生运行仍要求锁定的 Python 3.12/Isaac Sim 6.0/Isaac Lab 3.0 依赖，系统 Python 3.10/3.11 不适用。以下示例把 uv 缓存和虚拟环境放在仓库所在磁盘：

    git clone git@github.com:mrs-lab-robot/MRS_ROBOT_sim.git
    cd MRS_ROBOT_sim
    git submodule update --init third_party/IsaacLab-Arena
    cd arena_benchmark
    uv python install 3.12
    UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
      uv sync --project ../third_party/IsaacLab-Arena --no-default-groups --group isaaclab-from-wheel
    uv pip install --python .venv/bin/python --no-deps --editable ../third_party/IsaacLab-Arena
    uv pip install --python .venv/bin/python --editable ../sim_runtime/teleoperation
    uv pip install --python .venv/bin/python --editable ../isaaclab_ext
    uv pip install --python .venv/bin/python -e .
    export MRS_ROBOT_SIM_ROOT="$(realpath ..)"

如果仓库已检出但上游 Arena 子模块未初始化（在仓库根目录运行）：

    git submodule update --init third_party/IsaacLab-Arena

此方式使用上游锁定的 Isaac Lab 3.0.0b2 wheel。若需 Isaac Lab 源码开发环境，可选用上游 `isaaclab-from-source` 依赖组，但会额外检出和安装 Isaac Lab 子模块。Docker 运行时请把 MRS_ROBOT_sim 挂载进容器，并设置容器内的 MRS_ROBOT_SIM_ROOT；不要把宿主机路径写进仓库配置。

首次启动 Isaac Sim/Kit 时，NVIDIA 可能会在终端显示 Omniverse EULA 并要求交互确认。请先阅读 [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)，由操作者本人决定是否接受；不要在无人值守命令中自动代为确认。尚未完成首次确认时，headless smoke 会因没有可读取的终端输入而退出。

如果操作者已经阅读并明确接受 EULA，而运行方式没有交互终端，可在本次命令前显式设置 `OMNI_KIT_ACCEPT_EULA=YES`（例如 `OMNI_KIT_ACCEPT_EULA=YES bash scripts/run_native_isaac.sh ...`）。该变量代表操作者已作出的接受决定；不要把它设为仓库或系统的默认值。

先跑快速测试，再启动 Arena no-task 冒烟环境：

    PYTHONPATH=src:../isaaclab_ext/src:../sim_runtime/teleoperation/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit -q
    bash scripts/run_native_isaac.sh scripts/smoke_openflex.py --viz none --device cuda:0

看到 `OPENFLEX_ARENA_SMOKE_OK` 表示 Arena 已加载机器人 USD、reset 成功，处理了低速底盘/左臂探测动作，并完成多步零动作仿真且观测有限；这不是任务奖励或评测结果。

四路 RGB/depth policy-observation 验收（启用相机，运行单环境一步）：

    bash scripts/run_native_isaac.sh scripts/smoke_openflex_cameras.py --enable_cameras --viz none --device cuda:0 --steps 1

看到 `OPENFLEX_ARENA_CAMERA_SMOKE_OK` 且 `camera_obs` 含四路 RGB 与深度项，表示 Arena observation manager 已实际返回图像张量；相机默认关闭。该 smoke 本身不验证 ROS 图像话题，也不把图像写入 Kit 内 VR HDF5。桌面 GUI 启用 Arena 相机流时，会将 RGB 编为 JPEG，经本机 UDP relay 发布到 `/cam_{base,head,left,right}/color/image/compressed`，供统一 LeRobot 采集页读取。

Arena 任务/指标验收（在临时目录记录一个成功 episode）：

    bash scripts/run_native_isaac.sh scripts/smoke_openflex_task.py --viz none --device cuda:0

无 VR 的桌面方块任务 smoke（检查真实刚体桌面接触、静止不误报，再用状态注入验证任务谓词）：

    bash scripts/run_native_isaac.sh scripts/smoke_openflex_pick_cube.py --viz none --device cuda:0

看到 `OPENFLEX_PICK_CUBE_MDP_SMOKE_OK` 表示物理场景、静止不成功和成功谓词/指标通过；状态注入只验证谓词，不是机器人完成了抓取或抬升。

桌面键盘遥操作：

    bash scripts/run_native_isaac.sh scripts/teleop_openflex.py --viz kit --device cuda:0

本机启动包装脚本会从 Isaac Sim 的 Python 3.12 进程环境中滤除 ROS Humble 的 Python 3.10 与动态库路径，同时保留 CUDA 等其他路径。若当前终端 source 过 ROS Humble，请始终通过该脚本启动；Arena 每步控制本身不依赖 ROS。

### VR 遥操作与 GUI 数据采集

MRS_ROBOT 桌面程序的仿真工作流由以下九个页面组成：

| 页面 | 用途与当前能力 |
| --- | --- |
| 仿真配置 | 配置经典 Isaac Sim 与 Arena 运行路径、Python 环境 |
| 传感配置 | 配置经典 Isaac Sim 传感器；Arena 四路 RGB/depth 由数据采集页的 Arena 相机流开关控制 |
| 资产管理 | 查看机器人 USD、接口契约、传感器参数和底盘控制器资源 |
| 场景任务 | 查看已登记的环境配置和任务实现（含首个桌面方块抬升 MDP）；图形化任务编辑器尚未接入 |
| 数据采集 | 可选经典 Isaac Sim 或 IsaacLab-Arena 后端；两者共用同一 LeRobot 表单与录制脚本 |
| 数据管理 | 使用现有 LeRobot 数据工具；Arena HDF5 转换和增广尚未接入 |
| 模型训练 | 使用现有单任务 LeRobot 离线训练；Arena 仿真训练、RL 和并行调度尚未接入 |
| 模型评测 | 使用现有模型推理；Arena Benchmark 和并行评估尚未接入 |
| 任务中心 | 汇总仿真、采集、训练和评估进程状态及运行日志 |

在“数据采集”页可选择经典 Isaac Sim 或 IsaacLab-Arena。选择 Arena 后，先在同页 Arena 区域选择上游场景，再选择任务：任务留空时会将所选场景作为独立环境载入，并附加 Arena `NoTask`（只加载场景/机器人，不绑定操作任务）；选择兼容任务时则载入该任务 YAML 中定义的完整场景与任务。启用相机流后再启动 Kit 与 VR ROS 桥；数据表单任务名会跟随所选 YAML 更新。通过同页“检查仿真采集/开始采集”调用和真机相同的 LeRobot 录制器、Schema 与数据集格式。Arena 不走经典仿真传感器状态 API，也不传入经典环境复位配置；回合复位由 Kit/操作者处理。Kit 内 `OpenFlex VR Teleoperation` HDF5 面板仍可独立用于调试，但其 HDF5 不含相机图像。两种仿真后端互斥，且均不启动实体机器人 bringup。Arena 数据增广和仿真训练尚未接入 GUI；现有 LeRobot 数据管理与离线训练能力仍可使用。终端手动启动命令仍作为调试备用方式。

Arena Kit 仿真窗口另有 `OpenFlex VR Teleoperation` 面板，用于开始采集、保存/丢弃 Episode、复位机器人和触发本地急停。第一阶段写入 HDF5 的逐步字段包括 22 维 Arena action、19 个关节的位置/速度、仿真时间、ROS 命令序号和原始 VR/控制器命令 JSON；虽然 Arena 环境现已支持可选 RGB/depth policy observations，VR HDF5 recorder 尚未记录相机图像。

桌面程序默认使用 ROS domain `49`（与仿真页设置一致），调用的 launch 只启动 VR 遥操作节点和 Arena relay；**不会启动实体机器人 bringup 或 ros2_control**。未收到有效 VR 命令时，仿真保持安全停止。

#### 手动启动方式

终端 A 编译并启动 ROS VR relay（ROS Humble 已安装且外层 `openflex_ws` 已编译）：

```bash
source /opt/ros/humble/setup.bash
source ../../openflex_ws/install/setup.bash
colcon build --base-paths integrations/ros2/mrs_robot_arena_bridge \
  --build-base integrations/ros2/build --install-base integrations/ros2/install
source integrations/ros2/install/setup.bash
ROS_DOMAIN_ID=49 ros2 launch mrs_robot_arena_bridge arena_vr_teleop.launch.py
```

这个 launch 会启动现有 VR UDP 桥、底盘/双臂 IK/升降/头部 ROS 节点，以及 Arena relay；**不会启动真机或 ros2_control 机器人 bringup**。ROS relay 用 `/joint_states` 接收 Arena 仿真关节反馈，并在 `127.0.0.1:24102/24103` 和 Arena 交换命令/状态。

终端 B 启动 Arena GUI：

```bash
bash scripts/run_native_isaac.sh scripts/teleop_vr_openflex.py \
  --viz kit --device cuda:0 --record-dir outputs/datasets
```

Arena 进程不导入 `rclpy`；ROS Humble 与 Isaac Sim Python 3.12 保持进程隔离。ROS domain 使用仿真专用 `49`。启动顺序建议先启动 Arena，再启动 ROS launch。没有 VR 命令、命令超时、急停或本地 GUI 急停时，底盘速度归零，关节保持当前位置。采集时按 GUI 的“开始采集”，完成后按“保存 Episode”；输出文件名包含时间戳，不会覆盖同名数据。

控制协议及阶段计划见 [docs/teleoperation_architecture.md](docs/teleoperation_architecture.md) 和 [docs/data_collection.md](docs/data_collection.md)。

按键：`I/K` 前进/后退，`J/L` 左移/右移，`U/O` 原地旋转；`Tab` 切左右臂，`1`–`7` 选当前臂关节，按住 `W/S` 微调；`R/F` 升降，`A/D` 头部 yaw，`Q/E` pitch，`Z/X` 开合左右夹爪，`Esc` 退出。底盘指令被控制器几何和速度/加速度限值约束；不发布 ROS 指令。建议低速、逐项验证。

## 开发与验证

### 不连接 VR：用 ROS 2 / RViz 控制 Isaac Lab 或 Arena

先在 ROS Humble 终端编译并启动通用桥接。它会启动 RViz、`robot_state_publisher`
和 UDP relay，不会启动真机 bringup 或 `ros2_control`：

```bash
cd "$MRS_ROBOT_SIM_ROOT"
source /opt/ros/humble/setup.bash
source "$OPENFLEX_WS_ROOT/install/setup.bash"
source install/setup.bash
colcon --log-base arena_benchmark/integrations/ros2/log build \
  --base-paths arena_benchmark/integrations/ros2/mrs_robot_arena_bridge \
  --build-base arena_benchmark/integrations/ros2/build \
  --install-base arena_benchmark/integrations/ros2/install \
  --symlink-install
source arena_benchmark/integrations/ros2/install/setup.bash
ros2 launch mrs_robot_arena_bridge ros_control.launch.py ros_domain_id:=49
```

RViz 面板沿用已有 ROS 命令接口：`/cmd_vel`、双臂 `*/commands`、
`/head_forward_position_controller/commands` 和 `/lift_position_controller/commands`。
双臂/头部/升降柱位置目标会保持并限速执行；底盘速度和升降 jog 在 0.25 秒无新消息后归零。
急停可通过 `/vr_estop_active`（`std_msgs/Bool`）触发。

另开终端，每次只启动一个仿真后端：

```bash
# Isaac Lab
cd "$MRS_ROBOT_SIM_ROOT/isaaclab_ext"
python scripts/teleop_ros_robot.py --device cuda:0 --viz kit

# 或 IsaacLab-Arena（停止上一个后端后再运行）
cd "$MRS_ROBOT_SIM_ROOT/arena_benchmark"
bash scripts/run_native_isaac.sh scripts/teleop_ros_openflex.py --device cuda:0 --viz kit
```

两种 Kit 进程都通过 `127.0.0.1:24102/24103` 与 ROS relay 通信，彼此不能同时占用这些端口；
ROS Humble 的 `rclpy` 留在 Python 3.10 relay 进程中，不会导入到 Isaac 的 Python 3.12 中。
Arena 每步仍由 Arena 原生 action 执行，ROS 只传输命令和关节状态。

快速单元测试不启动 Kit/Isaac Sim：

    PYTHONPATH=src:../isaaclab_ext/src:../sim_runtime/teleoperation/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q

导入 `mrs_arena` 包根和 contract reader 不需要 Isaac Sim。导入 `mrs_arena.embodiments.openflex` 前必须先初始化 SimulationApp/Kit；不得在普通单元测试或包的 `__init__.py` 中提前导入 Isaac Lab 模块。`mrs_robot_arena` 仅用于兼容既有导入路径。

机器人类使用 Arena 的 register_asset 注册机制；扩展运行入口应先确保该模块在 Arena 资产查询前被导入。外部环境的启动参数和自定义 TaskBase 继承方式请遵照官方 [External Tasks and Embodiments](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0/pages/arena_in_your_repo/external_tasks_and_embodiments.html) 文档。

新增任务后，至少应在锁定的 Arena runtime（原生或容器）中验证：USD articulation 加载、reset、动作维数和关节映射、夹爪 mimic、传感器坐标系、终止条件、多环境并行隔离，以及固定随机种子的评测结果。

## 变更约定

- 不修改仓库根 `third_party/IsaacLab-Arena`；少量确实无法通过扩展解决的兼容改动才放入 `patches/`，并记录适用的上游提交。
- USD 与 ROS 接口契约以 MRS_ROBOT_sim 为单一事实来源。
- ROS 可用于数据采集或外部集成，但不进入 Arena 每个仿真步执行的策略控制路径。
- checkpoints、运行日志、临时数据和机器本地绝对路径不提交到 Git。
- Benchmark 报告需要记录代码提交、机器人资产版本、配置、随机种子和指标定义。
