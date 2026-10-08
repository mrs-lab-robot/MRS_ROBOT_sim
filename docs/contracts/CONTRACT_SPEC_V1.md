# MRS_ROBOT 契约规范 v1.0 - DRAFT

## 版本信息

- **版本**: 1.0
- **状态**: **DRAFT** (仅阶段1基线完成，接口未冻结)
- **最后更新**: 2026-10-06

## 已实现的契约组件

### 1. EmbodimentContract ✅ 已实现并测试

**源文件**: `sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml`  
**加载器**: `isaaclab_ext/src/mrs_robot_lab/contracts.py::load_embodiment_contract()`  
**测试**: `isaaclab_ext/tests/test_contract.py` (3个测试)  
**验证测试**: `isaaclab_ext/tests/test_episode_contract_v1_red.py::TestContractExtensions::test_contract_exposes_action_units`

#### 1.1 动作空间（已验证）

**总维度**: 22D  
**动作组**（已从embodiment.yaml解析）:
```yaml
actions:
  - name: base_twist
    dimension: 3
    fields: [linear.x, linear.y, angular.z]
    units: [m/s, m/s, rad/s]
  
  - name: lift_position
    dimension: 1
    joints: [lift_joint]
    limits: [{minimum: -0.650, maximum: 0.300}]
    units: [m]
  
  - name: head_position
    dimension: 2
    joints: [head_yaw_joint, head_pitch_joint]
    units: [rad, rad]
  
  - name: left_arm_position
    dimension: 7
    joints: [left_joint1, ..., left_joint7]
    units: [rad, rad, rad, rad, rad, rad, rad]
  
  - name: right_arm_position
    dimension: 7
    joints: [right_joint1, ..., right_joint7]
    units: [rad, rad, rad, rad, rad, rad, rad]
  
  - name: left_gripper_position
    dimension: 1
    units: [m]
  
  - name: right_gripper_position
    dimension: 1
    units: [m]
```

**已实现的契约API**（在contracts.py中）:
- `contract.action_dimensions: dict[str, int]` - 动作维度映射
- `contract.action_fields: dict[str, tuple[str, ...]]` - 字段名称
- `contract.action_joints: dict[str, tuple[str, ...]]` - 关节名称
- `contract.action_limits: dict[str, tuple[tuple[float, float], ...]]` - 限制
- `contract.action_units: dict[str, tuple[str, ...]]` - 物理单位 ✅ **新增**
- `contract.joint_action_names: tuple[str, ...]` - 关节动作名称列表

#### 1.2 观测空间（已验证）

**测量的关节状态维度**: 19个关节 × 2 (position + velocity) = 38D

**观测布局**（已从embodiment.yaml解析）:
```yaml
observation_layout:
  - observation: joint_state
    dimension: 38
    fields: [position.lift_joint, ..., velocity.right_gripper_joint]
  
  - observation: base_odom
    dimension: 7
    fields: [position.x, position.y, position.z, orientation.x, orientation.y, orientation.z, orientation.w]
  
  - observation: base_twist
    dimension: 6
    fields: [linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]
  
  - observation: livox_imu
    dimension: 10
    fields: [orientation.x, ..., linear_acceleration.z]
```

**已实现的契约API**:
- `contract.observation_dimensions: dict[str, int]`
- `contract.observation_fields: dict[str, tuple[str, ...]]`

**传感器能力**（从embodiment.yaml的sensors部分声明，不是任务必需字段）:
- `base_camera`, `head_camera`, `left_wrist_camera`, `right_wrist_camera`
- `lidar`, `imu`, `odom`

**注意**: 这些是机器人硬件能力，不是每个Episode必须包含所有传感器。

---

### 2. Episode记录格式 ✅ 已实现并测试

**实现**: `isaaclab_ext/src/mrs_robot_lab/recorders/teleop_episode.py::TeleopEpisodeRecorder`  
**测试**: 
- `isaaclab_ext/tests/test_teleop_recording.py` (4个旧测试)
- `isaaclab_ext/tests/test_episode_contract_v1_red.py` (6个新测试)

#### 2.1 三种动作语义 ✅ 已实现

**HDF5数据集**（已验证保存）:
1. `operator_command` (T, 22) - 原始输入指令（未裁剪）
2. `applied_target` (T, 22) - 裁剪后的目标指令
3. `action` (T, 22) - 向后兼容字段（等于applied_target）

**测试**: `test_episode_contract_v1_red.py::TestEpisodeActionSemantics::test_saved_episode_contains_operator_and_applied_actions`

#### 2.2 时序索引 ✅ 已实现

**HDF5数据集**:
- `step_index` (T,) - 动作步进索引（0-based，单调递增）
- `sim_time_ns` (T,) - 仿真时间戳（纳秒，单调递增）
- `camera_frames/<camera_name>/step_index` (N,) - 相机帧对应的步进索引
- `camera_frames/<camera_name>/sim_time_ns` (N,) - 相机帧时间戳

**验证**: 时间戳和索引单调性在`append()`中强制执行  
**测试**: `test_episode_contract_v1_red.py::TestEpisodeTemporalConstraints`

#### 2.3 溯源元数据 ✅ 已实现

**必需字段**（当提供完整溯源时）:
```python
metadata = {
    "episode_id": str,           # 唯一标识符
    "config_hash": str,          # 场景/任务配置的SHA256
    "contract_hash": str,        # embodiment.yaml的SHA256
    "random_seed": int,          # 随机种子
    "input_source": str,         # 数据来源标识
}
```

**Schema版本标记**:
- 当所有5个必需字段存在时，saver自动添加 `"episode_schema_version": "1.0"`
- 旧格式（`metadata={}`）保持兼容，不添加schema_version

**测试**: `test_episode_contract_v1_red.py::TestEpisodeProvenance::test_saved_episode_metadata_includes_all_provenance_fields`

#### 2.4 相机灵活性 ✅ 已实现

**能力**: Episode可包含任意相机子集（不要求全部4个相机）  
**约束**: 同一Episode内相机名称集合必须一致  
**测试**: `test_episode_contract_v1_red.py::TestEpisodeCameraFlexibility::test_episode_allows_per_task_camera_subset`

#### 2.5 API兼容性 ✅ 已实现

**旧API**（向后兼容）:
```python
recorder.append(
    sim_time_ns=...,
    action=[...],  # 22D
    joint_position=[...],  # 19D
    joint_velocity=[...],  # 19D
    next_joint_position=[...],
    command_seq=...,
    camera_frames={...},  # 可选
)
# step_index自动使用len(samples)
```

**新API**（显式动作语义）:
```python
recorder.append(
    sim_time_ns=...,
    step_index=...,  # 必需
    operator_command=[...],  # 原始输入
    applied_target=[...],    # 裁剪后目标
    joint_position=[...],
    joint_velocity=[...],
    next_joint_position=[...],
    command_seq=...,
    camera_frames={...},
)
```

---

## 未实现组件（阶段2-3待实现）

### 3. SceneSpec ⏳ 待实现

**负责Agent**: `sim-runtime` (阶段2.2)  
**目标**: 定义统一的场景描述格式，跨sim_runtime、isaaclab_ext、arena_benchmark

**TODO**:
- [ ] 定义ObjectPlacement数据结构
- [ ] 定义SceneSpec数据结构
- [ ] YAML序列化/反序列化
- [ ] sim_runtime场景加载器
- [ ] isaaclab_ext场景构建器
- [ ] arena_benchmark场景加载器

### 4. TaskSpec ⏳ 待实现

**负责Agent**: `isaaclab-learning` (阶段2.3), `arena-evaluation` (阶段2.4)  
**目标**: 定义统一的任务描述格式

**TODO**:
- [ ] 定义SuccessCriterion数据结构
- [ ] 定义TaskSpec数据结构
- [ ] 实现物理可行性验证（不是状态注入）
- [ ] 实现dual_arm_box_transport任务
- [ ] 实现base_navigation_target任务

### 5. LeRobot数据集导出 ⏳ 待实现

**负责Agent**: `gui-data` (阶段2.1)  
**目标**: 将HDF5 teleop录制转换为LeRobot训练格式

**TODO**:
- [ ] 实现lerobot_export.py转换器
- [ ] 验证与LeRobot训练流程兼容
- [ ] 测试ACT模型加载

### 6. Mimic数据增广 ⏳ 待实现

**负责Agent**: `isaaclab-learning` (阶段2.3)  
**目标**: Isaac Lab中的Sim示教重放和Mimic增广

**TODO**:
- [ ] 实现示教轨迹重放
- [ ] 实现Mimic增广策略
- [ ] 验证增广数据质量

---

## 已验证的事实摘要

### 测试通过记录（阶段1完成）

**契约解析**（3个测试）:
- ✅ 加载embodiment.yaml并解析维度
- ✅ 暴露关节组和限制
- ✅ 拒绝维度不匹配的契约

**Episode录制**（10个测试）:
- ✅ 4个旧API兼容性测试（test_teleop_recording.py）
- ✅ 6个新契约验证测试（test_episode_contract_v1_red.py）
  - 动作单位暴露
  - operator/applied动作语义
  - 溯源元数据完整性
  - 时间戳单调性
  - step_index关联
  - 相机子集灵活性

### 已知限制

1. **Arena契约同步**: arena_benchmark尚未实现contracts.py（阶段2.4）
2. **场景/任务规范**: SceneSpec和TaskSpec数据结构未定义
3. **LeRobot兼容性**: 转换器未实现，兼容性未验证
4. **跨层集成测试**: 三层联合测试未建立

---

## 版本管理

### 版本号规则

格式: `{major}.{minor}`
- major: 不兼容的破坏性变更
- minor: 向后兼容的扩展

### 当前版本路径

- **1.0-DRAFT**: 阶段1基线（EmbodimentContract + Episode）
- **1.0-RC**: 阶段2完成后（SceneSpec + TaskSpec + 四层实现）
- **1.0**: 阶段3验收通过（双任务训练+真机闭环）

### 冻结条件

接口冻结需满足：
1. 所有三层（sim_runtime、isaaclab_ext、arena_benchmark）实现完成
2. 跨层集成测试通过
3. 至少一个完整的训练-评测流程验收通过

**当前状态**: 未冻结，接口可能在阶段2调整

---

## 文件清单（已验证存在）

### 契约相关
- ✅ `sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml`
- ✅ `isaaclab_ext/src/mrs_robot_lab/contracts.py`
- ✅ `isaaclab_ext/src/mrs_robot_lab/recorders/teleop_episode.py`

### 测试文件
- ✅ `isaaclab_ext/tests/test_contract.py`
- ✅ `isaaclab_ext/tests/test_teleop_recording.py`
- ✅ `isaaclab_ext/tests/test_episode_contract_v1_red.py`

### 待创建（阶段2）
- ⏳ `sim_runtime/scenarios/task_scenarios/scene_spec.py`
- ⏳ `isaaclab_ext/src/mrs_robot_lab/recorders/lerobot_export.py`
- ⏳ `arena_benchmark/src/mrs_robot_arena/contracts.py`
- ⏳ `arena_benchmark/src/mrs_robot_arena/tasks/task_spec.py`

---

## 附录：契约API参考

### EmbodimentContract类

```python
@dataclass
class EmbodimentContract:
    robot_id: str
    action_dimensions: dict[str, int]
    action_fields: dict[str, tuple[str, ...]]
    action_joints: dict[str, tuple[str, ...]]
    action_limits: dict[str, tuple[tuple[float, float], ...]]
    action_units: dict[str, tuple[str, ...]]  # ✅ 新增
    joint_action_names: tuple[str, ...]
    observation_dimensions: dict[str, int]
    observation_fields: dict[str, tuple[str, ...]]
```

### TeleopEpisodeRecorder API

```python
class TeleopEpisodeRecorder:
    def start() -> None: ...
    
    def append(
        *,
        sim_time_ns: int,
        step_index: int | None = None,  # 新API必需，旧API可选
        operator_command: list[float] | None = None,  # 新API
        applied_target: list[float] | None = None,    # 新API
        action: list[float] | None = None,            # 旧API
        joint_position: list[float],
        joint_velocity: list[float],
        next_joint_position: list[float],
        command_seq: int,
        raw_command: dict[str, Any] | None = None,
        camera_frames: dict[str, bytes] | None = None,
    ) -> None: ...
    
    def save(output_path: str | Path, *, metadata: dict[str, Any]) -> Path: ...
    
    def discard() -> None: ...
```

---

**文档结束** - 2026-10-06
