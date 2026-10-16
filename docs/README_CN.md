# 文档索引

本仓库仍在开发中，不是稳定版本。完整桌面工作流需要另行部署配套的真机 GUI 仓库；本仓库不包含 GUI，且当前没有固定双方的兼容提交。

- [根目录 README](../README.md)：范围、目标版本、部署前置条件与初步构建步骤。
- [架构与边界](ARCHITECTURE_CN.md)：仿真、ROS 2、Isaac Lab 与 Arena 的职责划分。
- [仿真运行时](../sim_runtime/README.md)：ROS 2 overlay 的构建边界。
- [Isaac Lab 扩展](../isaaclab_ext/README.md)：Python 环境与测试边界。
- [Arena 扩展](../arena_benchmark/README_CN.md)：Arena 环境、依赖安装与单元测试。

部署命令尚未经过干净克隆的完整验收；请按目标机的依赖和 GUI 仓库版本复核。