"""仓库级引导：sys.path / 环境变量 / 需要 ROS 的测试清单。

为什么单独一个模块：测试统一收在 tests/ 一棵树下（unit / integration / smoke 三层，
见 tests/README.md），从仓库根跑和从子目录跑 rootdir 不一样，所以 tests/conftest.py
只做 3 行 `import bootstrap` 把这件事一次性做对。同样这套约定也给脚本用
（`from bootstrap import REPO_ROOT, SOURCE_ROOTS`），逐步替掉 50+ 处手写
`_configure_project_paths()`。

只依赖标准库：本模块要在任何 venv / 系统解释器下都能 import。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 源码根（存在才插入，不覆盖既有环境）
SOURCE_ROOTS = (
    REPO_ROOT / "arena_benchmark",              # `import scripts.teleop_vr_openflex`
    REPO_ROOT / "arena_benchmark" / "src",      # `import mrs_arena`
    REPO_ROOT / "isaaclab_ext" / "src",         # `import mrs_robot_lab`
    REPO_ROOT / "sim_runtime" / "teleoperation" / "src",
    REPO_ROOT / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_contract",
)

# 需要 ROS 2 运行时的测试：没装 rclpy 的环境里连 import 都过不了，直接跳过收集。
# 路径相对仓库根，与 tests/ 合并后的新位置一致。
ROS_ONLY_TESTS = (
    "tests/integration/teleoperation/test_ros_relay_aggregator.py",
    "tests/integration/ros2_contract/test_verify_openflex_contract.py",
    "tests/integration/ros2_control/test_verify_mobile_base_runtime.py",
    "tests/unit/sim_runtime/test_fastdds_launch_env.py",
    "tests/unit/sim_runtime/test_isaac_arm_command_mode.py",
    "tests/unit/legacy_ros/mid360/test_mid360_semantic_mapper.py",
    "tests/unit/legacy_ros/test_vla_contract_bridge.py",
)


def has_ros() -> bool:
    """宿主是否装了 ROS 2 Python 运行时。"""
    try:
        return importlib.util.find_spec("rclpy") is not None
    except Exception:  # pragma: no cover - 探测失败按没有 ROS 处理
        return False


def configure() -> None:
    """把源码根挂进 sys.path，并补上 MRS_ROBOT_SIM_ROOT。"""
    os.environ.setdefault("MRS_ROBOT_SIM_ROOT", str(REPO_ROOT))
    for root in SOURCE_ROOTS:
        entry = str(root)
        if root.exists() and entry not in sys.path:
            sys.path.insert(0, entry)


def collect_ignore() -> list[str]:
    """没有 ROS 时要跳过的测试文件绝对路径（给 conftest 用）。"""
    if has_ros():
        return []
    return [str(REPO_ROOT / path) for path in ROS_ONLY_TESTS]


configure()
