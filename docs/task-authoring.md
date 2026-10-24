# 建新任务：三条规矩 + 落地流程

> 目标：**换参数只改 YAML，不改 Python**。这样训练和评测用的是同一份事实，成功率才可比。
>
> 不新造格式：场景/任务沿用仓库已有的 `sim_runtime/config/{scenes,tasks}/*.yaml`（`spec_version: "1.0"`），
> 由 `mrs_arena.environments.openflex_graph_loader` 转成上游 IsaacLab-Arena 的 env graph。
> Arena 自带的那套 graph YAML 是"格式"，本仓这层 YAML 是"参数"，两者不要混用。

## 三条规矩

1. **物体位姿只能来自 scene/task YAML。**
   起点、朝向、目标位置、随机化幅度，一律读 `initial_state.*_pose` / `success_criteria.params.target_position` /
   `randomization_config`。Python 里**不允许**出现坐标常量。

2. **判据函数只吃数据，不读 YAML。**
   判据签名收"当前状态 + 阈值参数"，阈值由装配层从 YAML 读好再传进去。
   这样同一份判据既能给训练环境用，也能给评测用（见"判据只有一份"）。

3. **任务类只做装配。**
   `TaskBase` 子类负责：读 spec → 组 scene cfg → 挂判据 → 返回 metrics。
   不写死任何数值，不复制别处已有的判据实现。

## 落地流程（新建一个任务时照着做）

| 步 | 改哪里 | 做什么 |
|---|---|---|
| 1 | `sim_runtime/config/scenes/<scene>.yaml` | 写场景：地面、桌面、物体、灯光、随机化范围 |
| 2 | `sim_runtime/config/tasks/<task>.yaml` | 写任务：初始位姿、目标、成功判据参数、所需传感器 |
| 3 | `arena_benchmark/src/mrs_arena/tasks/<task>.py` | 照 `_template_task.py` 写装配类（数值全部从 spec 读） |
| 4 | `arena_benchmark/src/mrs_arena/environments/openflex_task_registry.py` | 加一条 `OpenFlexTaskProfile`（task_id / scene_id / factory） |
| 5 | `tests/integration/…` | 加一条"改 YAML 数值 → cfg 跟着变"的测试 |
| 6 | GUI | 任务下拉框从 registry 自动读，不用改 |

## 取数值用现成工具，别自己解析 YAML

```python
from mrs_arena.tasks.spec_placement import (
    object_placement_from_spec,   # initial_state.<id>_pose -> (位置, 四元数)
    target_position_from_spec,    # success_criteria.params.target_position
    randomization_from_spec,      # randomization_config -> (xy_m, yaw_rad)
)
```

## 自检清单

- [ ] 在 YAML 里把物体挪 5 cm，重新建 cfg，位置真的变了
- [ ] `grep -n "0\.[0-9]" <task>.py` 里没有坐标常量（阈值常量可以有，但要能追溯到 spec 字段）
- [ ] 判据函数没有 import YAML / 没有读文件
- [ ] 新增的 task 在 registry 里能查到，GUI 下拉框能看到
- [ ] `pytest tests -q` 仍然全绿

## 反例（不要照抄）

`dual_arm_box_transport.py` 把箱子起点/终点写成模块级常量
（`TASK_INITIAL_BOX_POSITION = (0.65, 0.0, 0.80)`），YAML 里改了也不生效，
`randomization_config` 也没实现。这个任务是测试用的临时场景，**保持原样不改**，
但新任务不要学它——照 `_template_task.py` 写。
