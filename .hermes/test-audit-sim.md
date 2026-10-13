# sim 仓库 test 脚本审查报告（Agent-sim 测试审查）

- 审查日期：2026-10-13
- 范围：仓库自写 test 文件（排除 `.venv` / `site-packages` / `build` / `install` / `third_party` / `extscache` / `__pycache__`）
- 方法：逐文件静态审查，对照当前源码 grep 校验 import 与符号是否还存在、是否与 canonical 分层测试重复
- **只读审查，未删除任何文件**，等用户拍板

## 一、总体统计

共 **121 个 test 文件 / ~2.0 万行**（用户口径 117 个可能未计 `isaaclab_ext/scripts` 与 `test/performance` 非 pytest 脚本）。

| 分类 | 文件数 | 行数 | 说明 |
|---|---|---|---|
| **A 保留** | 93 | ~15,431 | 还在验证当前核心功能（sim_runtime / isaaclab_ext / arena_benchmark 分层） |
| **B 可精简** | 13 | ~1,984 | 能跑但冗余/过碎/体量过大/与环境强耦合 |
| **C 删除** | 6 | ~1,023 | 失效（引用已删符号）/损坏/与 canonical 完全重复 |
| **D 合并** | 9 | ~1,929 | 与同模块其他测试重复，建议并入 |

> 结论：**77% 测试仍有价值（A），需要动刀的主要是 C（6 个）+ D（9 个）+ 少数 B**。test/ 里的性能脚本才是历史包袱大头（行数占了 ~8300 / 总 2.0 万）。

## 二、分组明细

### 2.1 arena_benchmark/tests（18 个，canonical unit 层）

`acceptance/`、`integration/` 子目录只有 README 占位，无测试文件（三层测试名不副实，实际只有 unit 一层）。

| 文件 | 行数 | 分类 | 理由 |
|---|---|---|---|
| tests/unit/test_box_transport_logic.py | 67 | A | 双臂搬运状态机核心逻辑，纯 CPU |
| tests/unit/test_navigation_logic.py | 105 | A | 导航核心逻辑/目标掩码/成功保持 |
| tests/unit/test_navigation_observation_adapter.py | 67 | A | 观测 ProxyArray→torch 适配，唯一覆盖 |
| tests/unit/test_navigation_policy_adapter.py | 32 | A | 动作→body twist 映射，唯一覆盖 |
| tests/unit/test_openflex_task_registry.py | 22 | A | 任务 profile 注册表，唯一覆盖 |
| tests/unit/test_scene_manifest.py | 73 | A | 场景清单构建与四元数换算 |
| tests/unit/test_teleop_script_helpers.py | 83 | A | teleop 脚本 helper，唯一覆盖 |
| tests/unit/test_livox_ros2_relay.py | 257 | A | MID360→Livox relay 链路，真实传感器链路 |
| tests/unit/test_mid360_sensor_mount.py | 278 | A | MID360 生命周期/挂载，唯一覆盖 |
| tests/unit/test_arena_navigation_evaluator.py | 155 | A | Arena 导航评测，唯一覆盖 |
| tests/unit/test_teleop_cli.py | 182 | A | graph_loader 与 teleop CLI 契约（可作为合并落点） |
| tests/unit/test_native_launcher.py | 86 | A | run_native_isaac.sh 环境契约（可作为合并落点） |
| tests/unit/test_arena_openflex_evaluator.py | 21 | B | 只断言一次透传，价值极低，建议并入其他 evaluator 测试 |
| tests/unit/test_dual_arm_box_task.py | 91 | B | 需 `MRS_RUN_ISAAC_APP_TESTS=1` 默认 skip，与 smoke 脚本重叠 |
| tests/unit/test_native_isaac_environment.py | 58 | D | 与 test_native_launcher 同测 run_native_isaac.sh，并入 |
| tests/unit/test_native_isaac_eula.py | 73 | D | 同上，EULA 分支并入 test_native_launcher |
| tests/unit/test_navigation_task_config.py | 22 | D | 与 test_navigation_logic 读同一 yaml，并入 |
| tests/unit/test_scene_only_graph.py | 37 | D | 与 test_teleop_cli 同测 graph_loader，并入 |

统计：A=12，B=2，C=0，D=4

### 2.2 isaaclab_ext/tests（42 个，canonical 分层测试）

| 文件 | 行数 | 分类 | 理由 |
|---|---|---|---|
| test_arena_sensor_api.py | 217 | A | Arena 传感器 HTTP 控制面，唯一覆盖 |
| test_asset_resolver.py | 94 | A | 资源解析与越界校验，唯一覆盖 |
| test_camera_cfg.py | 100 | A | 相机挂载/参数构建，唯一覆盖 |
| test_capture_api.py | 139 | A | 采集 HTTP 控制服务器，唯一覆盖 |
| test_contract.py | 78 | A | Embodiment 契约加载，唯一覆盖 |
| test_dual_arm_task_spec.py | 189 | A | task_spec 完整归属（作 D 合并落点） |
| test_episode_controller.py | 126 | A | EpisodeController 生命周期，唯一覆盖 |
| test_episode_replay.py | 188 | A | EpisodeReplayValidator，唯一覆盖 |
| test_joint_target_adapter.py | 42 | A | 关节目标归一化，唯一覆盖 |
| test_joint_teleop.py | 93 | A | 键盘 jog/夹爪/限位，唯一覆盖 |
| test_learning_proxy_state_methods.py | 152 | A | 学习任务 rewards/dones，唯一覆盖 |
| test_learning_task_interfaces.py | 38 | A | EnvCfg 观测/动作维度契约，唯一覆盖 |
| test_lerobot_append.py | 250 | A | append 幂等/溯源/manifest，唯一覆盖 |
| test_lerobot_export_bridge_integration.py | 136 | A | 双解释器端到端（opt-in 默认 skip），唯一覆盖 |
| test_lerobot_export.py | 225 | A | HDF5→LeRobot 映射，唯一覆盖 |
| test_lerobot_worker_export.py | 226 | A | worker_cli + lerobot CLI，唯一覆盖 |
| test_navigation_policy_evaluation.py | 675 | A | 评估 harness 全链路，唯一覆盖 |
| test_navigation_prim_path_regression.py | 124 | A | 环境装配回归，唯一覆盖 |
| test_navigation_reward_shaping.py | 127 | A | reward shaping（已核实 step_dt 为 isaaclab 属性） |
| test_ppo_training_entrypoint.py | 178 | A | PPO 入口/checkpoint manifest，唯一覆盖 |
| test_robot_interface.py | 32 | A | 契约常量断言（脆弱但有效） |
| test_runtime_rates.py | 41 | A | 频率换算，唯一覆盖 |
| test_smoke_environment.py | 130 | A | 无任务 smoke 环境，唯一覆盖 |
| test_swerve.py | 68 | A | SWERVE 配置/解算标量版，唯一覆盖 |
| test_swerve_tensor.py | 65 | A | SWERVE 张量版，唯一覆盖 |
| test_task_configuration.py | 111 | A | 任务配置加载，唯一覆盖 |
| test_task_logic.py | 95 | A | 搬箱/导航状态机，唯一覆盖 |
| test_task_logic_torch.py | 131 | A | 批量版状态机，唯一覆盖 |
| test_teleop_action_mapper.py | 100 | A | VR/ROS→动作映射，唯一覆盖 |
| test_vr_runtime.py | 261 | A | VR sidecar 运行时，唯一覆盖 |
| test_vr_sidecar.py | 138 | A | VR sidecar 配置，唯一覆盖 |
| test_worker_cli.py | 291 | A | worker CLI 流程，唯一覆盖 |
| test_worker_vr_action_adapter.py | 68 | A | VR→Lab 动作边界，唯一覆盖 |
| test_episode_contract_v1_red.py | 289 | B | 有效但文件名/注释"red 预期失败"已过时，改名去注释即可 |
| test_learning_quaternion_conventions.py | 125 | B | 有效，两例 DeprecationWarning 用例与上文件重复，可精简 |
| test_teleop_recording.py | 125 | B | 有效，部分 Recorder 用例与 episode_contract 重叠 |
| test_worker.py | 1024 | B | 39 个用例体量过大，建议拆分精简 |
| test_dual_arm_mimic_adapter.py | 127 | C | 依赖外部 `isaaclab_arena`，本包声明不依赖 Arena，标准环境必挂 |
| test_synthetic_vr.py | 83 | C | **已损坏**：空壳用例 + 引用未定义变量 `first/later` → NameError |
| test_dual_arm_box_task.py | 171 | D | 只测 load_task_spec，被 test_dual_arm_task_spec 完整覆盖 |
| test_dual_arm_observation_proxy_arrays.py | 72 | D | 与 test_learning_proxy_state_methods 同主题，并入 |
| test_lerobot_prepared_writer_integration.py | 69 | D | 与 test_lerobot_export 内 prepared-writer 用例重复，并入 |

统计：A=33，B=4，C=2，D=3

### 2.3 isaaclab_ext/scripts（4 个，Kit 冒烟入口）

| 文件 | 行数 | 分类 | 理由 |
|---|---|---|---|
| scripts/test_joint_control.py | 12 | B | 只是 `spawn_robot --no-cameras` 的三行别名 |
| scripts/test_camera.py | 12 | B | 只是 `spawn_robot` 无参别名 |
| scripts/test_actions.py | 12 | B | 只是 `spawn_robot --no-cameras --validate-actions` 别名 |
| scripts/test_remove_api_on_reference.py | 133 | C | 一次性 Kit 复现调试脚本，无断言，会启动 Kit，非测试 |

### 2.4 sim_runtime 各 ROS2 包自带 test/（11 个，当前核心）

| 文件 | 行数 | 分类 | 理由 |
|---|---|---|---|
| ros2/mrs_robot_arena_bridge/test/test_camera_relay.py | 112 | A | 相机 relay 核心链路（VR/远程预览） |
| ros2/mrs_robot_arena_bridge/test/test_relay_shutdown.py | 23 | A | relay 关停容错，唯一覆盖 |
| ros2/…/bringup/test/test_task_loader.py | 320 | A | TaskLoader YAML→运行时配置 |
| ros2/…/bringup/test/test_scene_spec_loader.py | 428 | A | SceneLoader + SceneSpec（注：文件名与实际模块 scene_loader.py 不符） |
| ros2/…/bringup/test/test_scene_environment_switch.py | 333 | A | start_robot_control_sim 环境切换 |
| ros2/…/bringup/test/test_rviz_sensor_displays.py | 209 | A | RViz 显示布局（与根 test/sim_unit/test_rviz_sensor_displays.py 同主题，分属两层） |
| ros2/…/contract/test/test_scene_task_spec_contract.py | 319 | A | SceneSpec/TaskSpec YAML 契约 |
| ros2/…/contract/test/test_scene_task_validation.py | 235 | A | spec 校验规则，唯一覆盖 |
| ros2/…/contract/test/test_session_config.py | 368 | A | SessionConfig/FrequencyConfig，唯一覆盖 |
| ros2/…/contract/test/test_unified_session_contract.py | 131 | A | 统一会话/预检/Embodiment，唯一覆盖 |
| teleoperation/test/test_camera_stream.py | 108 | A | CameraFrameAssembler + 预览注册（与 bridge 的 camera_relay 有部分重叠） |

统计：A=11，B=0，C=0，D=0

### 2.5 tests/（根，canonical 分层测试，14 个）

全部验证当前分层核心结构/接口，引用路径与符号均存在，**全部 A**。

| 文件 | 行数 | 分类 | 理由 |
|---|---|---|---|
| migration/test_layered_architecture.py | 166 | A | 三层架构独立性/资源不重复，唯一覆盖 |
| sim_runtime/teleoperation/test_ros_relay_aggregator.py | 66 | A | CommandAggregator |
| sim_runtime/teleoperation/test_teleop_lifecycle.py | 46 | A | teleop 生命周期 |
| sim_runtime/teleoperation/test_teleop_protocol.py | 49 | A | 协议帧 |
| sim_runtime/teleoperation/test_teleop_safety.py | 62 | A | 指令看门狗 |
| sim_runtime/teleoperation/test_udp_link.py | 68 | A | UDP 遥操作链路 |
| sim_runtime/test_fastdds_launch_env.py | 91 | A | sim.launch 环境清理契约 |
| sim_runtime/test_isaac_arm_command_mode.py | 75 | A | generate_isaac_urdf 指令模式 |
| sim_runtime/test_isaacsim_path.py | 94 | A | sim.launch isaac 路径解析 |
| sim_runtime/test_isaac_urdf_control_contract.py | 67 | A | URDF 各关节 command interface 契约 |

统计：A=14，B=0，C=0，D=0

### 2.6 test/（根，历史 runtime suite，32 个 —— README 明言"暂留"）

| 文件 | 行数 | 分类 | 理由 |
|---|---|---|---|
| performance/empty_scene_performance.py | 280 | A | 空场景性能基准，引用均存在 |
| performance/mid360_standalone_performance.py | 898 | A | MID360 独立性能，当前核心传感器 |
| performance/pointcloud_authority_restamper.py | 166 | A | 点云重署 ROS 工具节点 |
| performance/realsense_standalone_performance.py | 371 | A | RealSense 独立性能 |
| performance/robot_state_snapshot_relay.py | 149 | A | 机器人状态 UDP 转发工具 |
| performance/robot_control_performance.py | 434 | **C** | **已损坏**：L401 调用未定义的 `run_mode()` → NameError |
| performance/robot_control_full_performance.py | 810 | **D** | 引用已删 `robot_control_only.launch.py` + 参数不被 sim.launch.py 接受，无法运行 |
| performance/robot_control_isolated_camera_performance.py | 617 | **D** | `import robot_control_full_performance` + 引用已删 launch，同上族 |
| performance/test_robot_control_performance_metrics.py | 108 | B | 纯测 TopicSamples 可独立跑，但绑定上面 C 的脚本 |
| ros2_integration/contract/test_install_layout.py | 65 | A | description 包安装布局 |
| ros2_integration/contract/test_manifest_schema.py | 263 | A | manifest schema |
| ros2_integration/contract/test_simulation_app_startup.py | 120 | A | 启动脚本环境清理 |
| ros2_integration/contract/test_verify_openflex_contract.py | 275 | A | embodiment contract 校验器 |
| ros2_integration/control/test_direct_usd_spawn.py | 804 | A | USD 直出（需 pxr 环境，README 裸跑声明不符） |
| ros2_integration/control/test_lidar_contract_launch.py | 200 | A | lidar 话题契约 |
| ros2_integration/control/test_robot_control_only_launch.py | 76 | A | 实测 sim.launch.py 契约 |
| ros2_integration/control/test_rviz_only_launch.py | 172 | A | rviz_only launch |
| ros2_integration/control/test_stop_stale_vr_processes.py | 60 | A | 停 VR 残留进程脚本 |
| ros2_integration/control/test_verify_mobile_base_runtime.py | 89 | A | 移动底盘运行验证 |
| ros2_integration/control/test_vr_teleop_launch.py | 88 | A | vr_teleop launch |
| ros2_integration/sensors/test_standalone_components.py | 194 | A | 传感器独立组件（绑定 mid360 性能脚本常量） |
| ros2_integration/description/test_integrated_description_source.py | 51 | B | include 外部 underlay `openarmx_integrated_description`，本机未装即挂 |
| sim_unit/mid360/test_mid360_semantic_mapper.py | 117 | A | MID360 语义映射标签订单 |
| sim_unit/test_legacy_scaffolds_disabled.py | 83 | A | legacy 脚手架被禁用，唯一覆盖 |
| sim_unit/test_runtime_sensor_control.py | 1085 | A | 最大单文件，runtime sensor 全量契约 |
| sim_unit/test_rviz_sensor_displays.py | 82 | A | RViz 显示配置断言 |
| sim_unit/test_usd_negative_scale_normalization.py | 175 | A | USD 负缩放归一化（需 pxr 环境） |
| sim_unit/test_vla_contract_bridge.py | 108 | A | VLA 契约桥，当前核心 |
| sim_unit/test_robot_camera_resolution_matches_hardware.py | 44 | B | 依赖兄弟仓库 openflex_ws 文件 + 相对路径拼接，换机器即挂 |
| sim_unit/description/test_generate_isaac_urdf.py | 196 | **C** | 与 canonical tests/sim_runtime/test_isaac_urdf_control_contract.py + test_isaac_arm_command_mode.py 重复 |
| migration/test_stages_1_to_3_layout.py | 50 | **C** | 与 canonical tests/migration/test_layered_architecture.py 高度重复 |

统计：A=23，B=4，C=3，D=2

## 三、建议优先处理清单（待拍板）

1. **C 删 6 个（~1000 行）**：
   - `isaaclab_ext/tests/test_synthetic_vr.py`（已损坏 NameError）
   - `isaaclab_ext/tests/test_dual_arm_mimic_adapter.py`（依赖外部 isaaclab_arena）
   - `isaaclab_ext/scripts/test_remove_api_on_reference.py`（一次性的无断言 Kit 调试脚本）
   - `test/performance/robot_control_performance.py`（run_mode 未定义）
   - `test/sim_unit/description/test_generate_isaac_urdf.py`（canonical 已覆盖）
   - `test/migration/test_stages_1_to_3_layout.py`（canonical 已覆盖）
2. **D 合并 9 个**：主要并入落点已标注（test_native_launcher / test_teleop_cli / test_dual_arm_task_spec / test_learning_proxy_state_methods / test_lerobot_export）。其中 `robot_control_full/isolated_camera_performance.py` 两个"性能脚本"实际已坏死（引用已删 launch），所谓合并 = 同步删除，建议按 C 处理。
3. **B 中体量最大的**：`isaaclab_ext/tests/test_worker.py`（1024 行 / 39 用例）建议拆分精简；`test_episode_contract_v1_red.py` 改名去陈旧注释。
4. **结构与口径问题（当心）**：
   - `test/` 与 `tests/` 双目录并存，canonical 是 `tests/`（README 明言 test/ 暂留）——C/D 之外，B 类测试建议逐步迁移时同步核对。
   - `arena_benchmark/tests/acceptance|integration` 只有 README 空壳，三层实际只有 unit 一层；真正集成层是 `arena_benchmark/scripts/smoke_openflex*.py`。
   - 若干测试需要 pxr/USD 或 ROS 环境（test_direct_usd_spawn / test_usd_negative_scale_normalization 等），与 README"裸 python3 -m pytest test"的环境声明不符，跑通过的机器变数大。
   - `test_dual_arm_mimic_adapter.py` 若删，确认无测试再依赖其覆盖。

## 四、复核方法说明

- 关键存疑点均已二次 grep 核实：`robot_control_only.launch.py` 确已不在 bringup/launch（只剩 sim/rviz_only/vr_teleop/isaac_sim）；`run_mode()` 全仓库无定义；URDF 生成测试在 test/ 与 tests/ 两处确实重复。
- 未实际运行任何测试（符合本机环境约定：不跑全量 pytest / Qt GUI 测试）。
- arena_benchmark/.venv 内 sympy/torch/warp 等第三方测试未纳入。