# 仓库架构与边界

> 本仓库仍在构建中，接口与目录可能变化，不是稳定版本。它只提供仿真侧组件；完整 GUI 与真机应用由独立的配套仓库提供。

## 组件

| 目录 | 职责 |
| --- | --- |
| `sim_runtime/` | USD 资产、场景与传感器配置、ROS 2 运行时及传输工具 |
| `isaaclab_ext/` | Isaac Lab 适配、动作/观测配置及仿真工具 |
| `arena_benchmark/` | Arena 环境、任务、策略适配及评估工具 |
| `third_party/IsaacLab-Arena/` | 锁定版本的上游依赖，只读使用 |
| `test/`、`tests/` | 分层测试；运行条件依测试类型而异 |

## 依赖方向

```text
sim_runtime  →  isaaclab_ext  →  arena_benchmark
      └────── ROS 2 overlay / simulation-side transport
```

ROS 2 与 Isaac Sim 运行在不同 Python 环境中：目标分别为 ROS 2 Humble/Python 3.10 与 Isaac Sim/Python 3.12。不要把两者的解释器或动态库路径混用。

## 外部依赖与运行边界

- ROS 2 构建需要 Humble，以及配套 GUI/机器人工作区提供的 underlay 依赖；当前仓库不包含该工作区。
- Arena 依赖版本由 [`configs/versions.yaml`](../configs/versions.yaml) 锁定；上游源码位于 Git 子模块。
- 真机 GUI、硬件驱动、实体机器人安全控制、完整数据采集/训练/推理流程不属于本仓库的独立交付范围。
- 当前接口、配置和部署步骤均可能调整；通过静态检查或单元测试不代表完成 Isaac Sim、ROS、GUI 或真机联合验收。

## 资源维护

优先从现有 canonical 资源和配置中读取，不复制出第二份权威文件。修改 ROS 接口、资产结构或版本锁时，应同步检查对应的构建与契约测试，并记录尚未覆盖的集成场景。