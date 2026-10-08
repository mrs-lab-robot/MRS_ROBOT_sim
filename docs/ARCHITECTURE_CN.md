# 仓库架构与边界

## 当前边界

`MRS_ROBOT_sim` 是 Isaac Sim + ROS 2 仿真的唯一 Git 边界。OpenFleX 主工作空间是运行和
构建 underlay，不拥有本仓库的仿真源码。

```text
MRS_ROBOT_sim/
├── sim_runtime/                     # USD、传感器、场景、ROS 2、通用遥操作
├── isaaclab_ext/                    # mrs_robot_lab：Articulation、Action、Observation、smoke
├── arena_benchmark/                 # mrs_arena：Embodiment、Scene、Task、Benchmark
├── third_party/IsaacLab-Arena/      # 固定提交的只读上游子模块
├── configs/versions.yaml            # 全局运行时版本锁
├── tests/                           # 分层新测试
├── test/                            # 历史运行/ROS/performance 测试，兼容期保留
├── reports/                         # 迁移、性能和运行证据
└── tools/                           # 资产与模型工具
```

`isaac_sim_core/`、`ros2_pkgs/openflex_isaac_sim/`、`arena/` 和 `arena_benchmark` 内的
`third_party/isaaclab-arena` 是旧接口兼容符号链接。新代码应依赖 canonical 路径，而不是旧入口。

## ROS 2 包

| 包 | 责任 |
| --- | --- |
| `openflex_isaac_description` | 生成 Isaac Sim 专用 URDF |
| `openflex_isaac_contract` | 动作、观测、关节顺序、topic 和频率合同 |
| `openflex_isaac_controllers` | Isaac Sim 的 ros2_control 适配与控制器 |
| `openflex_isaac_sensors` | 四路 RGB-D、MID360、IMU 和 ROS 2 发布适配 |
| `openflex_isaac_bridge` | Isaac Sim 与 ROS 2 的通用桥接层 |
| `openflex_isaac_bringup` | 全链路启动、RViz/VR 辅助入口和运行验证 |

旧 `isaacsim_*` 包已删除，不能继续从 `ros2_pkgs/control` 或
`ros2_pkgs/simulation_bridge` 引用源码。

## 依赖方向

```text
OpenFleX underlay ──> openflex_isaac_description ──> openflex_isaac_bringup
isaac_ros2_utils ──> openflex_isaac_controllers ───┘
sim_runtime ──> isaaclab_ext (mrs_robot_lab) ──> arena_benchmark (mrs_arena)
      │
      └── ROS 2 packages ──> OpenFleX underlay / simulator topics
```

`sim_runtime` 不依赖 Isaac Lab/Arena；`isaaclab_ext` 不依赖 benchmark；只有 Arena 顶层可以组合下层。
VR protocol/transport 位于 runtime，绝对关节命令到 Lab action 的适配位于 `mrs_robot_lab`。

OpenFleX underlay 提供机器人基础描述、swerve 控制器和应用层节点。第三方
`isaac_ros2_utils` 通过 `config/dependencies.repos` 获取，不能把嵌套 `.git` 目录提交进来。

## Canonical 文件

| 内容 | 唯一维护位置 |
| --- | --- |
| 机器人 USD | `sim_runtime/assets/robots/openflex_robot.usda` |
| 控制器 | `sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_bringup/config/controllers.isaac.mobile_base.yaml` |
| 动作/观测合同 | `sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml` |
| RealSense 配置 | `sim_runtime/config/sensors/realsense/` |
| MID360 配置 | `sim_runtime/config/sensors/mid360/` |
| 传感器 topic 合同 | `sim_runtime/config/sensors/sensors.isaac.yaml` |

`openflex_isaac_sensors/config/` 中的 RealSense 配置和 calibration 是指向上表 canonical 文件的符号链接；
ROS 包安装仍沿用自身的文件名和数据安装规则，不再保留第二份配置副本。

## 运行数据流

```text
Isaac physics -> /openflex/joint_states -> ros2_control -> /joint_states
ROS commands  -> controllers             -> /openflex/joint_command
Isaac odom    -> /odom                   -> /fastlio2/lio_odom compatibility relay
RGB-D         -> /cam_*/{color,depth}    -> VLA compressed image contract
MID360        -> raw PointCloud2         -> /livox/lidar CustomMsg
                                      \-> /livox/lidar_points and /scan
```

`/fastlio2/lio_odom` 目前是 `/odom` 的兼容转发，不是 FAST-LIO。`/livox/lidar` 已转换为
真机使用的 `livox_ros_driver2/msg/CustomMsg`，可直接作为兼容 FAST-LIO 的输入；但消息类型兼容
不代表已经运行 FAST-LIO，也不替代动态外参、时间同步和轨迹精度验收。

## VLA 边界

本仓库负责稳定仿真观测和动作接口，不负责数据集落盘格式、训练任务、checkpoint 或推理策略。
VLA 数据采集、训练和推理节点位于上层应用仓库，通过 ROS 2 合同接入。当前接口与控制检查已
通过，但在宣称完整 VLA 流程完成前，还需要对实际数据采集器、训练代码和推理节点分别验收。
