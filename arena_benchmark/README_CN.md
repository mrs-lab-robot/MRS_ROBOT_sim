# Arena 仿真与评估层

> **开发状态：**本仓库仍在构建中，不是稳定版本；接口、任务和运行方式都可能变化。
>
> 完整桌面工作流需要配套的真机 GUI 仓库。该 GUI 不包含在本仓库中，双方当前也没有在这里固定兼容提交。

本目录提供 Arena 环境、任务实现、策略适配和评估工具。它读取同仓库仿真组件及上游 Arena 子模块中的资源，不包含桌面 GUI，也不提供真机安全控制。

## 目标运行环境

精确版本见 [`../configs/versions.yaml`](../configs/versions.yaml)。当前开发目标为 Linux x86_64、Python 3.12、Isaac Sim 6.0.0.1、Isaac Lab 3.0.0b2 和锁定的 Arena 提交。这是开发基线，不构成稳定兼容承诺；其他架构、驱动版本和最低 GPU 配置尚未声明支持。

## 安装

在仓库根目录初始化上游源码：

```bash
git submodule update --init third_party/IsaacLab-Arena
```

然后在本目录创建 Python 环境并安装可编辑包：

```bash
uv python install 3.12
UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
  uv sync --project ../third_party/IsaacLab-Arena --no-default-groups --group isaaclab-from-wheel
uv pip install --python .venv/bin/python --no-deps --editable ../third_party/IsaacLab-Arena
uv pip install --python .venv/bin/python --editable ../sim_runtime/teleoperation
uv pip install --python .venv/bin/python --editable ../isaaclab_ext
uv pip install --python .venv/bin/python --editable .
```

首次启动 Kit 时可能出现 NVIDIA/Omniverse 许可提示。操作者应先阅读条款，再自行决定是否接受；仓库不会自动接受。

## 测试与限制

在本目录运行不启动模拟器的单元测试：

```bash
CONTRACT_DIR="$(printf '%s\n' ../sim_runtime/ros2/*_isaac_sim/*_contract)"
PYTHONPATH="src:$CONTRACT_DIR" .venv/bin/python -m unittest discover -s tests/unit -v
```

Isaac Sim/Kit 集成检查需要锁定的运行时与合适的 GPU/驱动。单元测试不验证渲染、物理接触、ROS 数据流、GUI 集成或完整 benchmark 协议。目标机验收完成前，现有任务和评估脚本均应视为实验性实现。

整体部署边界见[根目录 README](../README.md)。