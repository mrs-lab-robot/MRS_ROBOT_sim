# 仿真与评测工具链

> **状态：当前仍在构建中，尚非稳定版本。**接口、配置格式和部署步骤可能变化，不应作为生产就绪版本使用。
>
> **功能边界：完整功能需要与配套的真机 GUI 仓库组合部署。**本仓库只提供仿真侧组件，不包含桌面 GUI、真机应用或真机安全控制；配套 GUI 仓库的兼容提交目前也未在此固定。

## 仓库内容

- `sim_runtime/`：仿真资产、配置、ROS 2 包和传输工具。
- `isaaclab_ext/`：Isaac Lab 适配和仿真侧工具。
- `arena_benchmark/`：Arena 环境、任务和评估工具。
- `third_party/IsaacLab-Arena/`：锁定版本的上游子模块；不要直接修改其内容。
- `tests/`：单元与集成测试。GPU/Kit 验收与不启动模拟器的测试是不同范围。
- `test/`：额外的 ROS/运行时/性能检查；运行条件取决于具体测试。性能探针为可选工具，不构成稳定验收协议。

为兼容现有代码，运行时内部包名和资产标识暂时保留；它们不构成稳定的公共 API。

## 目标软件栈

当前目标版本见 [`configs/versions.yaml`](configs/versions.yaml)：ROS 进程使用 ROS 2 Humble/Python 3.10，Isaac Sim 与 Arena 环境使用 Python 3.12。版本锁是开发基线，不是稳定版本兼容承诺。当前没有声明最低 GPU/驱动要求，也没有声明支持非 x86_64 平台。

部署还需要 Linux x86_64、可用的 NVIDIA 驱动与 Vulkan/RTX、ROS 2 Humble、配套 GUI/机器人工作区提供的 ROS underlay，以及锁定版本的 Isaac Sim、Isaac Lab 和 Arena 依赖。构建步骤需要 `git`、`colcon`、`vcs` 和 `uv`。

本仓库不包含 Isaac Sim，也没有维护中的项目专用容器镜像。首次启动时可能出现 NVIDIA/Omniverse 许可提示，操作者应自行阅读并决定是否接受。

## 检出

先将维护者提供的克隆地址设为环境变量 `REPOSITORY_URL`，再执行：

```bash
: "${REPOSITORY_URL:?请先设置仓库克隆地址}"
git clone "$REPOSITORY_URL" simulation-stack
cd simulation-stack
git submodule update --init third_party/IsaacLab-Arena
```

上游子模块可能包含可选源码子模块；只有使用源码开发依赖组时才需要继续初始化。

## 构建 ROS 2 overlay

先安装并 source ROS 2 Humble，再构建配套 GUI/机器人工作区。随后将它作为仿真包的 underlay：

```bash
source /opt/ros/humble/setup.bash
source /path/to/companion-gui-workspace/install/setup.bash
export ISAACSIM_ROBOT_ROOT="$(pwd)"

# 若 underlay 未提供锁定的源码依赖，再导入该依赖。
mkdir -p .deps/src
vcs import .deps/src < config/dependencies.repos

colcon --log-base log/simulation build \
  --build-base build/simulation \
  --install-base install \
  --symlink-install \
  --base-paths sim_runtime/ros2/*_isaac_sim
source install/setup.bash
```

`vcs import` 只检出源码，不会构建依赖。若 underlay 缺少某个包，应按其自身文档先构建并 source，再构建本 overlay。

## 创建 Arena Python 环境

以下命令使用上游锁定的 Python 3.12 wheel 依赖组：

```bash
cd arena_benchmark
uv python install 3.12
UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
  uv sync --project ../third_party/IsaacLab-Arena --no-default-groups --group isaaclab-from-wheel
uv pip install --python .venv/bin/python --no-deps --editable ../third_party/IsaacLab-Arena
uv pip install --python .venv/bin/python --editable ../sim_runtime/teleoperation
uv pip install --python .venv/bin/python --editable ../isaaclab_ext
uv pip install --python .venv/bin/python --editable .
```

首次启动 Kit 时可能出现交互式许可提示。应先阅读条款，再由操作者决定是否接受；无人值守安装不得默认替用户接受。

## 检查

不启动模拟器的 Arena 单元测试：

```bash
cd arena_benchmark
CONTRACT_DIR="$(printf '%s\n' ../sim_runtime/ros2/*_isaac_sim/*_contract)"
PYTHONPATH="src:$CONTRACT_DIR" .venv/bin/python -m unittest discover -s tests/unit -v
```

Isaac Sim/Kit 冒烟与集成检查需要锁定运行时、兼容 GPU/驱动和相应启动环境。单元测试通过不代表渲染、传感器、ROS、GUI 或真机链路已完成端到端验收。

## 部署限制

以上命令是部署起点，不是经过验证的一键安装器。本 README 尚未证明干净克隆部署或 GUI 到仿真的完整联调。完整流程应按配套真机 GUI 仓库自己的部署说明执行，并在组合前确认两边版本兼容。

根目录 [`LICENSE`](LICENSE) 为 MIT；第三方代码、资产、模拟器二进制和子模块可能适用各自条款，重新分发前应分别核对。