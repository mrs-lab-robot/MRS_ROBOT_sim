# tests/ — 测试地图（写给 AI agent 的）

本仓库**只有这一个测试根**。历史上分散在 `arena_benchmark/tests/`、`isaaclab_ext/tests/`、
`tests/`、`test/` 四棵树里（同内容不同目录跑出不同结果），现已合并为 `tests/`，内部按
`unit/` → `integration/` → `smoke/` 三层按**运行成本**分层，不按代码归属分。

- `pytest.ini` 里 `testpaths = tests`，从仓库根 `pytest` 和 `pytest tests` 等价。
- 全仓只有 `tests/conftest.py` 一个 conftest，3 行 `import bootstrap`；
  `tools/bootstrap.py` 负责 `sys.path` / `MRS_ROBOT_SIM_ROOT` / ROS-only 跳过清单。
- **不要**再新建第二棵测试树，也不要在子目录里加 `conftest.py`。

---

## 0. 30 秒上手

```bash
cd /media/y/workdisk/project/MRS_ROBOT/MRS_ROBOT_sim

# 全量（CPU 可跑的部分，约 13 秒）
env -u LD_LIBRARY_PATH -u PYTHONPATH -u ROS_DISTRO -u AMENT_PREFIX_PATH \
  arena_benchmark/.venv/bin/python -m pytest tests -q
```

解释器固定用 `arena_benchmark/.venv/bin/python`（Python 3.12，带 torch / isaaclab / pxr）。
系统 `python3` 跑不了 `tests/unit/lab`（缺 isaaclab）。

**必须**清掉 `LD_LIBRARY_PATH` / `PYTHONPATH` / `ROS_DISTRO` / `AMENT_PREFIX_PATH`，
原因见 §4。

---

## 1. 怎么跑

| 目的 | 命令 |
| --- | --- |
| 全量 | `pytest tests` |
| 只跑 CPU 单元层 | `pytest tests/unit` |
| 只跑集成层 | `pytest tests/integration` |
| 单目录 | `pytest tests/unit/arena` |
| 单文件 | `pytest tests/unit/lab/test_worker.py` |
| 单用例 | `pytest "tests/unit/lab/test_worker.py::TestWorkerInitialization::test_xxx"` |
| 按关键字 | `pytest tests -k "worker or teleop"` |
| 按 marker | `pytest tests -m requires_kit`（见下方说明，目前无测试带此 marker） |
| 失败即停 + 详细栈 | `pytest tests -x --tb=short` |
| 只看收集到什么 | `pytest tests --co -q` |
| 让收集错误不中断整轮 | `pytest tests --continue-on-collection-errors` |

> `pytest.ini` 的 `addopts` **故意不写 `-q`**：写死 `-q` 后命令行再传 `-q` 会变成 `-qq`，
> 全绿的运行会连 `N passed` 汇总行都不打印，很容易误判。

`markers` 目前只声明了 `requires_kit`，但**没有任何测试真正带这个 marker**——
Kit 门控实际走环境变量 `MRS_RUN_ISAAC_APP_TESTS=1`（见 §5）。

---

## 2. 目录地图

### `tests/unit/` — 纯 CPU，毫秒级，无 ROS / 无 Kit / 不启模拟器

| 路径 | 测什么 | 用例数 |
| --- | --- | --- |
| `unit/arena/` | Arena 任务层（`mrs_arena`）：任务注册表、场景清单、导航评估器/观测/策略适配器、双臂搬箱传输逻辑、MID360 挂载解析、遥操作 CLI 与脚本 helper | ~65 |
| `unit/lab/` | IsaacLab 适配层（`mrs_robot_lab`）：worker 状态机与 CLI、episode 契约/控制器/回放、LeRobot 导出与追加、mimic 标注、VR runtime/sidecar、swerve 底盘、task spec 与奖励、PPO 入口、关节/遥操作动作映射、传感器与相机 cfg | ~236 |
| `unit/sim_runtime/` | `sim_runtime` 运行时契约：IsaacSim 路径解析、URDF 关节命令接口、FastDDS 环境变量 | 3 |
| `unit/legacy_ros/` | 旧 ROS replica 栈的单元测试：运行时传感器控制 HTTP API、rviz 显示配置、相机分辨率对齐、VLA 契约桥、mid360 语义映射、legacy 脚手架已禁用 | ~34 |
| `unit/test_usd_negative_scale_normalization.py` | USD 负缩放烘焙后顶点数/法线（**当前 2 failed，另有处理**，见 §5） | 4 |

**`unit/` 根下那个散文件是有意的**：它原本在 `test/sim_unit/`，为了让 `REPO_ROOT = parents[2]`
这行**一个字节都不用改**，落在与旧路径同深度的 `tests/unit/` 根上。等负缩放问题修完后应挪进
`unit/legacy_ros/` 并把该行改成 `parents[3]`。

### `tests/integration/` — 跨模块接线，多数要外部运行时（ROS 2 / 已安装包 / 子进程）

| 路径 | 测什么 |
| --- | --- |
| `integration/architecture/` | 分层静态契约：`sim_runtime` → `isaaclab_ext` → `arena` 的目录/依赖方向、资产唯一真源、禁止反向 import（14 个用例，纯读文件，CPU 可跑） |
| `integration/teleoperation/` | 遥操作协议栈：命令帧、看门狗安全、UDP 链路、生命周期、ROS relay 聚合器 |
| `integration/ros2_contract/` | ROS 2 契约校验：安装布局、manifest schema、Isaac 启动脚本、`verify_*_contract` CLI 行为 |
| `integration/ros2_control/` | ROS 2 控制链：launch 静态契约（`robot_control_only` / `rviz_only` / `vr_teleop` / `lidar`）、直连 USD spawn、mobile base 运行时校验、stale VR 进程清理 |
| `integration/ros2_description/` | 集成 description 源文件名守卫 |
| `integration/ros2_sensors/` | 独立传感器组件（`openflex_isaac_sensors`）的 diagnostics / 性能审计 |

`integration/ros2_*` 是原来 `test/ros2_integration/{contract,control,description,sensors}/`
**原样搬过来**的（`git mv`，内容零改动）。目录名带 `ros2_` 前缀而不是塞进
`integration/ros2/<area>/`，是因为这些文件用 `parents[3]` 算仓库根，必须保持同一深度。

### `tests/smoke/` — 需要 Isaac Sim Kit / GPU

| 路径 | 测什么 |
| --- | --- |
| `smoke/test_dual_arm_box_task.py` | 启 `AppLauncher` 后构造双臂搬箱任务，验证接触传感器、成功判据阈值、mimic 未被提前认领。**默认跳过**，`MRS_RUN_ISAAC_APP_TESTS=1` 才跑 |

### `tests/performance/` — 不是测试

`*.py` 文件名不以 `test_` 开头，**不会被 pytest 收集**。是可选的人工性能探针
（空场景实时率、MID360/RealSense 独立性能、点云 authority restamper、robot state relay），
需要 ROS 2 / Kit / NVIDIA 渲染。输出写 `reports/`，不构成验收协议。

---

## 3. 运行环境分级

| 层级 | 需要什么 | 怎么判断 |
| --- | --- | --- |
| **CPU-only** | 只要 `arena_benchmark/.venv` | `tests/unit/**`（除下表列出的）、`tests/integration/architecture/`、`tests/integration/teleoperation/` 的非 ROS 部分 |
| **ROS 2 运行时** | `rclpy` + `ament_index_python` + `source install/setup.bash` | `tests/integration/ros2_*`；`tests/unit/legacy_ros/test_vla_contract_bridge.py`；`tests/unit/legacy_ros/mid360/test_mid360_semantic_mapper.py`；`tests/unit/sim_runtime/test_fastdds_launch_env.py`；`tests/unit/sim_runtime/test_isaac_arm_command_mode.py` |
| **Isaac Kit / GPU** | 锁定版 Isaac Sim + NVIDIA 驱动 + `MRS_RUN_ISAAC_APP_TESTS=1` | 只有 `tests/smoke/test_dual_arm_box_task.py` |
| **外部 Python 包** | `LeRobot`（+ NumPy/Pillow） | `tests/unit/lab/test_lerobot_export.py`、`test_lerobot_append.py` 的部分用例 |

**没有 rclpy 时**：`tools/bootstrap.py` 的 `ROS_ONLY_TESTS` 会把 7 个文件直接
`collect_ignore`（不收集，不是 skip，`N skipped` 里看不到它们）：

```
tests/integration/teleoperation/test_ros_relay_aggregator.py
tests/integration/ros2_contract/test_verify_openflex_contract.py
tests/integration/ros2_control/test_verify_mobile_base_runtime.py
tests/unit/sim_runtime/test_fastdds_launch_env.py
tests/unit/sim_runtime/test_isaac_arm_command_mode.py
tests/unit/legacy_ros/mid360/test_mid360_semantic_mapper.py
tests/unit/legacy_ros/test_vla_contract_bridge.py
```

装了 ROS 2 后 `has_ros()` 返回 True，清单自动失效，这些文件会被正常收集。

> 加进这张清单 = **这些用例彻底不跑**。所以只加「无 ROS 时连 import 都过不去」的文件。
> `tests/integration/ros2_control/test_direct_usd_spawn.py` 有 23 个用例、其中 21 个在纯 CPU
> venv 里能过，只因为 2 个要 Isaac python 布局而失败——**不要**把它整份扔进这张清单，
> 那会静默丢掉 21 个通过的用例。

---

## 4. 环境变量坑（最容易假错的地方）

**症状：明明代码没动，`import` 报错 / `lib*.so` 找不到 / 收集期就崩。**

这台机器上 shell 里通常已经 export 了 ROS 2 Humble 的 `LD_LIBRARY_PATH`、`PYTHONPATH`、
`AMENT_PREFIX_PATH`。它们会把 ROS 的 Python 3.10 路径塞进 Python 3.12 的 venv 前面，
导致 `numpy`/`torch`/`pxr` 加载到不兼容的 `.so`。

固定写法（每条命令都加）：

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH -u ROS_DISTRO -u AMENT_PREFIX_PATH \
  arena_benchmark/.venv/bin/python -m pytest <path>
```

其他坑：

- **`MRS_ROBOT_SIM_ROOT`**：由 `bootstrap.py` 用 `setdefault` 设置。如果你已经 export 了
  指向**别的 checkout** 的值，测试会去读别人的资产，症状是「文件存在但断言对不上」。
  怀疑时 `unset MRS_ROBOT_SIM_ROOT`。
- **`OPENFLEX_TEST_ROBOT_USD`**：`tests/unit/test_usd_negative_scale_normalization.py` 会用它
  覆盖被测资产路径；设成过期路径会得到与负缩放无关的失败。
- **`MRS_RUN_ISAAC_APP_TESTS=1`**：唯一让 `tests/smoke` 真正启动 Kit 的开关。**不要**在
  CPU 机器上开，会卡在 Kit 初始化或直接 core dump。
- **`PYTEST_DISABLE_PLUGIN_AUTOLOAD`**：老文档里出现过，**现在不要用**——会连
  `pytest-subtests` 一起禁掉，而多个文件用了 `subTest`，会静默少跑一堆用例。
- 从子目录（`cd tests && pytest`）跑是支持的，但 `rootdir` 仍是仓库根（靠 `pytest.ini`），
  路径参数要相对当前目录。

---

## 5. 已知 skip / 已知 fail 清单

### 5.1 skip（`N skipped` 里的，共 9 条）

| 文件 | 条数 | 原因 | 恢复条件 |
| --- | --- | --- | --- |
| `tests/smoke/test_dual_arm_box_task.py` | 3 | `@unittest.skipUnless(os.environ.get("MRS_RUN_ISAAC_APP_TESTS") == "1")` —— 需要 Isaac Sim Kit | GPU 主机上 `MRS_RUN_ISAAC_APP_TESTS=1`，或改用 `scripts/smoke_openflex_dual_arm_box.py` |
| `tests/unit/lab/test_lerobot_export.py` | 3 | `LeRobot`（以及 prepared-dataset 用例的 NumPy + Pillow）未安装 | `pip install lerobot`（版本要匹配 episode 格式），两条 prepared-writer 用例还要 NumPy + Pillow |
| `tests/unit/lab/test_lerobot_append.py` | 1 | 同上：`LeRobot is not installed` | 同上 |
| `tests/integration/ros2_contract/test_install_layout.py` | 2 | `No installed openflex_isaac_contract package found in AMENT_PREFIX_PATH` —— 需要真的 `colcon build` 过并 `source install/setup.bash` | 构建安装空间后 source 再跑 |

### 5.2 未收集（`collect_ignore`，不出现在任何计数里）

`tools/bootstrap.py::ROS_ONLY_TESTS` 的 7 个文件，宿主没有 `rclpy` 时直接不收集，见 §3。

### 5.3 已知失败

| 文件 | 条数 | 原因 | 状态 |
| --- | --- | --- | --- |
| `tests/unit/test_usd_negative_scale_normalization.py` | 2 | 资产法线问题：镜像后 `left_link1` 顶点数 2101 ≠ 右侧 2141，法线未重新烘焙 | **已另行处理，别在这个任务里修**。它同时贡献 2 passed，所以别整个文件 skip 掉 |
| `tests/integration/ros2_control/test_direct_usd_spawn.py` | 2 | `test_isaac_python_resolver_*` / `test_pip_isaac_launch_isolates_*` 需要机器上有 Isaac Sim 的 python 布局（`isaacsim`/`isaaclab` 的 pip 或 workstation 安装），纯 venv 里没有 | 该文件另外 21 个用例在 CPU 下正常通过；缺 Isaac 安装环境时只该失败这 2 条 |

### 5.4 顺序依赖（隐藏地雷，务必知道）

`tests/unit/legacy_ros/test_runtime_sensor_control.py` 在模块顶层 `import openflex_isaac_sensors`，
而该包**只有** `test_legacy_scaffolds_disabled.py` 在 import 时把
`sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_sensors` 插进 `sys.path` 才可导入。

- ✅ `pytest tests` / `pytest tests/unit/legacy_ros`：同目录按文件名排序，
  `test_legacy_scaffolds_disabled` 先导入 → 正常（34 passed）。
- ❌ `pytest tests/unit/legacy_ros/test_runtime_sensor_control.py`：**单独跑会收集失败**
  （`ModuleNotFoundError: openflex_isaac_sensors`），不是代码坏了。
- 修法（未做，属测试逻辑改动）：在该文件里自己 `sys.path.insert`，别依赖兄弟文件。

---

## 6. 重复 / 重叠的测试：保留哪个

合并时没有出现**同名**文件（历史遗留的 `test_worker.py`、`test_capture_api.py` 冲突在合并前
已被上游解决），但存在几组**语义重叠**的测试。新增用例前先看这张表，别再写第三份。

| 重叠组 | 涉及文件 | 重叠点 | 建议保留 |
| --- | --- | --- | --- |
| 双臂搬箱任务 | `tests/smoke/test_dual_arm_box_task.py`（Kit，3 个 skip）<br>`tests/unit/arena/test_box_transport_logic.py`（CPU，4 个）<br>`tests/unit/lab/test_dual_arm_task_spec.py`（CPU，6 个） | 同一任务的三层：Kit 里真实构造 / 传输进度判据 / TaskSpec 解析。**不是真重复**，`test_dual_arm_box_task.py` 自己的 docstring 就写着「CPU 上请用 transport_logic 那份」 | 三个都留；改任务逻辑时按层各改一处 |
| 导航目标判据 | `tests/unit/lab/test_task_logic.py`<br>`tests/unit/arena/test_navigation_logic.py` | 都覆盖 goal mask / 航向环绕（`wrapped heading`） | 两份都留：前者测 `NavigationGoalEvaluator` 纯函数，后者测从 yaml 载入 spec 后的行为。**新用例加到 `test_task_logic.py`** |
| 契约 / manifest schema | `tests/unit/lab/test_contract.py`<br>`tests/integration/ros2_contract/test_manifest_schema.py` | 都校验 embodiment 契约字段 | 留 `test_contract.py`（CPU、无 ROS）；`test_manifest_schema.py` 只在验证 ROS 安装侧行为时用 |
| rviz 显示配置 | `tests/unit/legacy_ros/test_rviz_sensor_displays.py`<br>`tests/integration/ros2_control/test_rviz_only_launch.py` | 都断言 `robot_control_only.rviz` 里的 lidar 显示与 SensorDataQoS | 留 `test_rviz_only_launch.py`（还覆盖 launch 环境变量与节点参数）；legacy 那份只剩「控制面板默认禁用」这条独有断言，可缩到 1 个用例 |
| 运行时传感器控制 API | `tests/unit/legacy_ros/test_runtime_sensor_control.py`<br>`tests/unit/lab/test_arena_sensor_api.py` | 都是 HTTP create/destroy + owner-thread 语义 | **优先 `test_arena_sensor_api.py`**（Arena 栈是当前目标栈）；legacy 那份只留 Isaac 5.1 兼容性相关用例 |
| LeRobot 写盘 | `tests/unit/lab/test_lerobot_export.py`<br>`tests/unit/lab/test_lerobot_append.py` | 都碰 dataset writer | 不重复：export = 新建数据集，append = 追加到已有。都留 |
| worker | `tests/unit/lab/test_worker.py`<br>`tests/unit/lab/test_worker_cli.py`<br>`tests/unit/lab/test_worker_vr_action_adapter.py` | 同一 worker 的状态机 / CLI 参数 / VR 动作适配 | 不重复，都留 |

---

## 7. 新增测试放哪

1. 不启模拟器、不连 ROS、毫秒级 → `tests/unit/<层>/`（arena 任务 → `unit/arena`，
   lab 适配 → `unit/lab`，旧 ROS replica 栈 → `unit/legacy_ros`，`sim_runtime` → `unit/sim_runtime`）。
2. 要起 ROS 2 节点/launch、要读已安装空间、或要跑子进程 → `tests/integration/`。
3. 要 `AppLauncher` / GPU → `tests/smoke/`，并**必须**用
   `@unittest.skipUnless(os.environ.get("MRS_RUN_ISAAC_APP_TESTS") == "1", ...)` 门控。
4. 文件名必须全局唯一（没有 `__init__.py`，pytest 按 basename 做模块名）。同语义的加前缀区分，
   别覆盖。
5. 用 `Path(__file__).resolve().parents[N]` 算仓库根时：本文件在 `tests/<layer>/<group>/x.py`
   → `parents[3]` 是仓库根，`tests/<layer>/x.py` → `parents[2]`。**别写相对 cwd 的路径**，
   pytest 可以从任何目录启动。
6. 需要 ROS 才能在无 ROS 环境 import 的，把路径加进 `tools/bootstrap.py::ROS_ONLY_TESTS`，
   不要在测试里 `try: import rclpy except ImportError: skip`。

---

## 8. 合并前的基线（存档）

合并前四棵树各跑一遍（同一 venv、同一环境变量清理方式）：

| 树 | passed | failed | skipped |
| --- | --- | --- | --- |
| `arena_benchmark/tests/` | 65 | 0 | 3 |
| `isaaclab_ext/tests/` | 232 | 0 | 4 |
| `tests/` | 33 | 0 | 0 |
| `test/` | 116 | 21 | 2 |
| **合计** | **446** | **21** | **9** |

合并后 `pytest tests` 实测：**447 passed, 4 failed, 9 skipped**（4 failed = §5.3 那 4 条）。
逐文件对比：79 个测试文件逐个单独跑，非 ROS 文件的 passed 数与合并前**完全一致**，
没有一条用例因为搬家而丢失或改判。总数比 446 多 1，来自
`tests/integration/ros2_control/test_direct_usd_spawn.py` 在此期间被修好了一条用例
（20 → 21 passed），与本目录结构无关。

> 判断「是不是合并把测试弄丢了」的正确方法不是看单次总数，而是 `pytest tests --co -q` 看收集数：
> 有文件被别的工作改动或加进 `ROS_ONLY_TESTS` 时，收集数会掉，那是文件内容/环境的事，
> 不是目录结构的事。目录结构的健康度用 `pytest tests --co -q | tail -1` 与本节表格对账。
