# sim 仓库 test 清理交付说明

- 执行日期：2026-10-13
- 依据：`.hermes/test-audit-sim.md`（仅按报告的 C/D/B 分类执行，未扩大范围）
- 交付物：删除 12 个测试文件、合并 6+2 个文件的唯一覆盖、精简 test_worker.py
- 未提交 git

## 一、C 类删除（报告 6 个 + 删除前 grep 确认无引用）

| 文件 | 理由 |
|---|---|
| `isaaclab_ext/tests/test_synthetic_vr.py` | 已损坏（NameError） |
| `isaaclab_ext/tests/test_dual_arm_mimic_adapter.py` | 依赖外部 isaaclab_arena |
| `isaaclab_ext/scripts/test_remove_api_on_reference.py` | 一次性无断言 Kit 调试脚本 |
| `test/performance/robot_control_performance.py` | `run_mode` 未定义，已损坏 |
| `test/sim_unit/description/test_generate_isaac_urdf.py` | canonical `tests/sim_runtime/` 已覆盖 |
| `test/migration/test_stages_1_to_3_layout.py` | canonical `tests/migration/test_layered_architecture.py` 已覆盖 |

## 二、D 类合并（报告 9 个，逐项并入后删除被并入文件）

| 被并入文件 | 落点 | 说明 |
|---|---|---|
| `test_native_isaac_environment.py` | `test_native_launcher.py` | humble 库环境契约并入 |
| `test_native_isaac_eula.py` | 同上 | EULA 3 用例 + helper 并入 |
| `test_navigation_task_config.py` | `test_navigation_logic.py` | 读同一 yaml，缺 backend 断言并入 |
| `test_scene_only_graph.py` | `test_teleop_cli.py` | graph_loader 契约 2 用例并入 |
| `test_dual_arm_box_task.py` | `test_dual_arm_task_spec.py` | 仅测 load_task_spec，被完整覆盖，直接删 |
| `test_dual_arm_observation_proxy_arrays.py` | `test_learning_proxy_state_methods.py` | _get_observations 唯一覆盖并入 |
| `test_lerobot_prepared_writer_integration.py` | `test_lerobot_export.py` | append + LeRobotDataset 可重开用例并入 |
| `test/performance/robot_control_full_performance.py` | 删除 | 引用已删 `robot_control_only.launch.py`，无法运行 |
| `test/performance/robot_control_isolated_camera_performance.py` | 删除 | 依赖上者 + 已删 launch |

## 三、B 类精简

- `isaaclab_ext/tests/test_worker.py`：**1024 行 / 39 用例 → 967 行 / 38 用例**。
  提取 `_launch_with_mocked_stages()` 消除 8 处重复的 6 连 MagicMock+launch 样板；
  删除与 `test_launch_failure_exposes_failure_reason_in_status` 重复的
  `test_launch_failure_transitions_to_failed`。覆盖保留。
- `arena_benchmark/tests/unit/test_arena_openflex_evaluator.py`（B，仅断言一次透传）：
  并入 `test_arena_navigation_evaluator.py` 后删除。
- `test/performance/test_robot_control_performance_metrics.py`（B，绑定已删 C 脚本）：
  脚本删除后其唯一被测对象 TopicSamples 不存在，同步删除。

## 四、删除引发的引用残留处理

- `test/ros2_integration/control/test_reporting_layout.py` 中 2 个用例直接 `read_text` 已删性能脚本，
  已删除该 2 用例；其余 4 用例（报告规范 / empty_scene / 报告文件）保留并通过。
- 全仓 grep 复查：不再有任何 `.py` 引用已删模块（`test_scene_only_graph` 仅命中并入后的函数名）。

## 五、验证结果（本机未跑全量 pytest）

1. `python3 -m py_compile`：本次改动的 8 个保留/合并文件全部通过。
2. arena_benchmark `/tests/unit` 全量（60 tests）：**1 失败、3 skip**
   - 唯一失败 `test_vr_teleop_keeps_fabric_enabled_for_kit_rendering` 为**既有失败**（脚本已改为
     `disable_fabric=_ARENA_DISABLE_FABRIC`，断言仍期望 `ast.Constant False`），与本次清理无关（本文件 diff 仅 +41 行新增）。
3. isaaclab_ext 精简子集：
   - `test_learning_proxy_state_methods.py`：5/5 OK
   - `test_lerobot_export.py`：收集正常，2 OK + 3 skip
   - `test_worker.py`：38 OK + 1 skip（共 39 项）
4. `test_reporting_layout.py`：4/4 OK
5. 其余 isaaclab_ext / sim_runtime / test 大套件未跑（本机内存约束，按约定只跑精简后子集）。

## 六、后续建议（未执行，避免扩大范围）

- `test_episode_contract_v1_red.py` 改名去旧注释、`test_learning_quaternion_conventions.py` 去重复用例等
  其他 B 项，报告已列，留待后续单独处理。
- `test_teleop_cli.py` 既有失败（`disable_fabric` 断言）与脚本已脱节，建议到期单独核对。