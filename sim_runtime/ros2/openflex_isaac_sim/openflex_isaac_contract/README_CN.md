# OpenFleX Embodiment Contract

这个包保存 OpenFleX 机器人 embodiment 的机器可读 contract，用于把机器人本体、动作、
观测、传感器角色和 ROS 2 真机话题接口从具体场景、task、SERF/VLA policy 中解耦。

## 边界

- 这个包可以被 Isaac runtime、OmniGibson robot registration、task provider 和 policy adapter 读取。
- 这个包不依赖 SERF、task-0021、BEHAVIOR dataset root 或 checkpoint。
- 这个包不包含 task-0021 的对象、reset 状态或 goal predicate。
- 这个包不定义 R1Pro 的 action、proprioception 或 checkpoint layout；任何 retargeting 都属于独立的 policy adapter。

当前 Isaac runtime 目标是 Isaac Sim 6.0，仿真 ROS domain 默认是 49，真机 domain 是 0。
这个包只提供共享 contract，不实现
OmniGibson robot import、task provider 或 SERF 接入。

## Canonical contract 内容

`config/embodiment.yaml` 是后续 robot registration 的唯一机器可读接口来源：

- `runtime` 和 `domains` 声明 Isaac Sim 6.0、真机 ROS domain 0、仿真 ROS domain 49；
  同时保留 `robot` 中的同名兼容字段。
- 每个 `actions` 条目定义 controller、topic、字段顺序、单位、逐字段范围、默认命令频率
  与安全裁剪语义。`lift_velocity` 是 VLA 的真机兼容速度入口，由仿真桥接到位置控制器，
  不重复计入 22 维主动作向量。关节范围来自生成后的 OpenFleX URDF；频率来自
  Isaac controller manager 的 `update_rate`。
- `base_twist` 的范围是 Isaac swerve 配置中 `max_wheel_speed` 和轮组几何的单轴运动学
  包络，不是实机额定值。多轴组合仍由 `swerve_drive_controller` 的轮速限制负责。
- `observation_layout` 为固定长度的 joint position/velocity、odom、twist 和 IMU 向量
  列出字段、顺序、维度和单位。`PointCloud2` 保持原始可变长度消息，不伪造固定维度。
- 所有 action、observation 和 sensor topic 都必须在 `ros2_topics` 中声明；validator 会
  检查引用、消息类型与收发方向。

后续 OmniGibson 注册或 policy adapter 必须消费这些字段，不得复制一份私有的 action 或
observation 向量定义。

## 验证

默认 verifier 会读取 `openflex_isaac_bringup` 的 controller/sensor YAML 对照配置。
因此应编译并 source contract、description 和 bringup 三个包后运行：

```bash
cd /path/to/MRS_ROBOT_sim
source /opt/ros/humble/setup.bash
source /path/to/openflex_ws/install/setup.bash
colcon build --symlink-install \
  --base-paths ros2_pkgs/openflex_isaac_sim .deps/src \
  --packages-select openflex_isaac_contract openflex_isaac_description openflex_isaac_bringup
source install/setup.bash
ros2 run openflex_isaac_contract verify_embodiment_contract.py
colcon test --packages-select openflex_isaac_contract --event-handlers console_direct+
```

期望 verifier 输出：

```text
[OK] OpenFleX embodiment contract matches Isaac controller and sensor configs
```

如果只 build/source `openflex_isaac_contract`，安装布局测试仍可验证 contract
自身的安装产物，但默认 verifier 会提示缺少 `openflex_isaac_bringup` 配置。此时可
显式传入配置文件：

```bash
ros2 run openflex_isaac_contract verify_embodiment_contract.py \
  --controllers /path/to/controllers.isaac.mobile_base.yaml \
  --sensors /path/to/sensors.isaac.yaml
```

## 后续阶段

下一阶段是工业任务 adapter：使用这个 contract 对齐 Isaac Sim 6.0 里的机器人动作、
观测、传感器角色和 ROS 2 topic，再接储氢罐阀门检测等场景任务。检测任务、
策略输入输出或数据集私有 layout 不能写回 contract 包。
