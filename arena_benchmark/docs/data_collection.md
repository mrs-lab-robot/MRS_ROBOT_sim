# VR 遥操作数据采集

## 启动与调试

推荐在现有 MRS_ROBOT 桌面程序中切换至“仿真”模式，然后进入“整机控制”页的“MRS_ROBOT_Arena VR 数据采集（本机）”卡片，点“一键启动 Arena + VR ROS 桥”。该卡片复用现有仿真进程托管和日志面板；Arena Kit 窗口中的采集面板负责 episode 操作。Kit 日志出现 `VR_TELEOP_GUI_READY` 表示 Arena 仿真和采集面板已就绪；ROS 日志出现 `Arena relay ready` 表示 ROS UDP relay 已启动。

该入口不调用实体机器人 bringup。若启动失败，先在仿真日志中看首个 `[Arena Kit]` 或 `[Arena ROS 桥]` 错误：Arena 侧通常与 CUDA/Kit/资产路径有关；ROS 侧通常与 ROS Humble、OpenFleX 工作空间安装或 colcon 构建有关。单独启停按钮可用于隔离定位。

## GUI 操作

`scripts/teleop_vr_openflex.py` 打开 Kit 仿真窗口和控制面板。推荐先启动 Arena，再启动 ROS VR launch。

- “开始采集”：清空内存中的当前 episode 缓冲并开始逐仿真步记录。
- “保存 Episode”：写出一个 HDF5 文件并关闭当前 episode。
- “丢弃”：丢弃尚未保存的 episode。
- “机器人复位”：丢弃正在采集的未保存 episode，reset Arena articulation 并重置命令 watchdog。
- “急停 / 解除”：仿真侧本地急停切换；解除后仍需有效 VR 数据才能恢复动作。

输出默认位于 `outputs/datasets/`，目录不进入 Git。已存在的同名文件会拒绝覆盖。

## HDF5 schema v1

每个文件是一条 episode，顶层包含：

- attributes：`format=mrs_robot_arena_teleop_v1`、`metadata_json`。
- `sim_time_ns[N]`、`command_seq[N]`；时间戳对应 pre-step observation。
- `action[N,22]`：实际传给 Arena `env.step()` 的 action。
- `joint_position[N,19]`、`joint_velocity[N,19]`：动作前状态；`next_joint_position[N,19]`：动作后状态，均为 MRS embodiment 契约顺序。
- `raw_command_json[N]`：对应 command frame，包括处理后 ROS setpoint 和可用的 VR 原始手柄输入。

HDF5 当前用于清晰可靠的逐步 teleop 采集；相机图像将使用 Arena camera observation/recorder 配置扩展，不把高带宽图像塞入 UDP 控制协议。

## 数据质量检查

最少检查每个 episode 的步数非零、动作维数 22、关节状态维数 19、仿真时间单调递增、值有限、reset 前后 episode 不混合。保存 metadata 中应保留协议版本、动作步长和 joint order。后续增加相机后，另验证 image shape、相机内外参、帧时间戳和关节状态同步误差。
