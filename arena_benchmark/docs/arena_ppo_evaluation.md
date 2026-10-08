# Arena PPO 检查点评估

此模块在 Arena 注册的环境中评估 IsaacLab PPO 导航检查点。

## 功能特性

- ✅ 加载 `isaaclab_ext` 训练的 RSL-RL PPO 检查点
- ✅ 在 Arena 的 `OpenFlexNavigationEmbodiment` 环境中评估
- ✅ 完整的来源验证（manifest/checkpoint/task/scene/contract SHA256）
- ✅ 8D 导航观测适配（goal-relative odometry）
- ✅ 归一化动作到 SI 单位的自动转换
- ✅ 确定性种子范围评估
- ✅ 与 `isaaclab_ext/scripts/evaluate_navigation_ppo.py` 输出格式兼容

## 架构设计

### 模块职责

```
arena_benchmark/
├── src/mrs_arena/learning/arena_navigation_evaluator.py
│   └── 核心评估逻辑（模拟器无关）
│       ├── run_arena_navigation_episode()      # Episode 执行
│       ├── load_checkpoint_metadata()          # 检查点加载和验证
│       └── make_arena_policy_wrapper()         # RSL-RL → Arena 适配
│
├── scripts/evaluate_arena_navigation_ppo.py
│   └── 命令行评估脚本（需要 Isaac Sim）
│
└── tests/unit/test_arena_navigation_evaluator.py
    └── 模拟器无关的单元测试
```

### 依赖关系

- **Arena 环境**: 使用已注册的 `OpenFlexNavigationTask` 和 `OpenFlexNavigationEmbodiment`
- **观测适配**: 复用 `mrs_arena.adapters.navigation_observation`
- **动作适配**: 复用 `mrs_arena.policies.navigation.policy_actions_to_body_twist`
- **结果构建**: 复用 `mrs_robot_lab.environments.learning.navigation_evaluation`
- **检查点验证**: 复用 `mrs_robot_lab` 的 `validate_checkpoint_provenance`

## 使用方法

### 1. 运行单元测试（不需要 GPU/Isaac Sim）

```bash
cd arena_benchmark
PYTHONPATH=src python3 -m unittest discover -s tests/unit -p "test_arena_navigation_evaluator.py" -v
```

或使用测试脚本：

```bash
cd arena_benchmark
python3 run_evaluator_tests.py
```

### 2. 评估检查点（需要 Isaac Sim）

```bash
cd arena_benchmark
./scripts/evaluate_arena_navigation_ppo.py \
    --checkpoint /path/to/run/checkpoints/model_5000.pt \
    --num-episodes 10 \
    --seed-start 1000 \
    --headless
```

**参数说明**：
- `--checkpoint`: 训练完成的检查点文件路径（必需）
- `--num-episodes`: 评估的 episode 数量（默认 10）
- `--seed-start`: 起始随机种子（默认 1000）
- `--headless`: 无头模式运行（推荐）
- `--device`: 计算设备，如 `cuda:0`（默认从 AppLauncher 继承）

### 3. 评估输出

评估结果保存到 `<run_dir>/evaluations/arena_eval_<timestamp>.json`：

```json
{
  "schema_version": 2,
  "evaluation_backend": "arena",
  "checkpoint_sha256": "abc123...",
  "arena_environment": {
    "embodiment": "OpenFlexNavigationEmbodiment",
    "task": "OpenFlexNavigationTask",
    "num_envs": 1
  },
  "navigation_target": {
    "position_xy": [3.0, 0.0],
    "yaw_rad": 0.0
  },
  "aggregate": {
    "total_episodes": 10,
    "success_count": 9,
    "success_rate": 0.9,
    "mean_steps": 245.3
  },
  "episodes": [
    {"index": 0, "seed": 1000, "steps": 250, "success": true, ...},
    ...
  ]
}
```

## 实现细节

### 观测空间

8D 导航观测（goal-relative odometry）：
```python
[
    dx,           # 目标相对位置 x (米)
    dy,           # 目标相对位置 y (米)
    cos(dθ),      # 目标相对朝向 cos
    sin(dθ),      # 目标相对朝向 sin
    vx,           # 机器人世界坐标系速度 x (m/s)
    vy,           # 机器人世界坐标系速度 y (m/s)
    cos(θ),       # 机器人朝向 cos
    sin(θ),       # 机器人朝向 sin
]
```

### 动作空间

归一化动作 `[-1, 1]^3` → SI 单位 body twist：
```python
policy_actions_to_body_twist([vx_norm, vy_norm, ω_norm])
→ [vx_m/s, vy_m/s, ω_rad/s]  # 受 SWERVE_CONFIG 限制
```

### 检查点验证

评估前验证：
- ✅ `run_manifest.json` 的 `status == "completed"`
- ✅ `task_id == "navigation_to_goal"`
- ✅ `schema_version == 2`
- ✅ 检查点路径匹配
- ✅ 检查点文件 SHA256 匹配
- ✅ Task/scene/contract 配置 SHA256 匹配（如提供）

## 与 isaaclab_ext 的差异

| 特性 | isaaclab_ext | arena_benchmark |
|------|--------------|-----------------|
| 环境构建 | `make_navigation_task_env()` | `ArenaEnvBuilder(make_openflex_navigation_environment())` |
| 观测构建 | 环境内置 | `navigation_policy_observation()` 适配 |
| 动作适配 | 环境内置 | `policy_actions_to_body_twist()` 适配 |
| 结果构建 | `run_navigation_episode()` | `run_arena_navigation_episode()` |
| 注册验证 | 不需要 | 验证 Arena 注册 |

## 限制和已知问题

### ⚠️ 当前限制

1. **固定初始状态**: 评估使用 task_spec 中定义的固定初始位置，不测试泛化能力
2. **单环境**: 仅支持 `num_envs=1`（Arena 评估协议要求）
3. **确定性推理**: 策略运行在 `torch.inference_mode()` 下

### 🔴 已知阻塞问题

**无** - 所有核心功能已实现并在 arena_benchmark 范围内

### 📋 未来改进方向

1. 支持随机初始状态的泛化评估
2. 支持自定义评估场景
3. 添加可视化和调试模式
4. 支持批量评估多个检查点

## 测试覆盖

### 单元测试（模拟器无关）

- ✅ `test_imports_exist`: 模块导入验证
- ✅ `test_run_episode_requires_single_env`: 单环境要求验证
- ✅ `test_run_episode_validates_max_length`: 最大步数验证
- ✅ `test_load_checkpoint_metadata_validates_path`: 路径验证
- ✅ `test_load_checkpoint_metadata_validates_manifest`: Manifest 存在验证
- ✅ `test_load_checkpoint_metadata_validates_manifest_schema`: Schema 验证
- ✅ `test_make_arena_policy_wrapper_builds_callable`: 策略包装器验证

### 集成测试（需要 Isaac Sim）

运行 `scripts/evaluate_arena_navigation_ppo.py` 对实际检查点评估即为集成测试。

## 贡献指南

1. 所有修改必须遵循 TDD（测试驱动开发）原则
2. 单元测试必须是模拟器无关的（可以在 `PYTHONPATH=src python3 -m unittest` 下运行）
3. 严格在 `arena_benchmark` 范围内工作，不修改 `isaaclab_ext` 或其他模块
4. 复用现有的适配器和结果构建器，避免重复实现

## 参考文档

- `isaaclab_ext/scripts/evaluate_navigation_ppo.py` - 原始评估脚本
- `mrs_arena/adapters/navigation_observation.py` - 观测适配器
- `mrs_arena/policies/navigation.py` - 动作适配器
- `mrs_robot_lab/environments/learning/navigation_evaluation.py` - 共享评估逻辑
