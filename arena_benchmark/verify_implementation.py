#!/usr/bin/env python3
"""验证 Arena PPO 评估器实现的完整性。

此脚本执行静态检查以验证所有必需的文件和函数都已实现。
不需要 Isaac Sim 或 GPU。
"""

from pathlib import Path
import sys


def check_file_exists(path: Path, description: str) -> bool:
    """检查文件是否存在。"""
    if path.exists():
        print(f"  ✓ {description}: {path.name}")
        return True
    else:
        print(f"  ✗ {description}: {path} (不存在)")
        return False


def check_function_in_file(path: Path, function_name: str) -> bool:
    """检查文件中是否定义了指定函数。"""
    try:
        content = path.read_text(encoding="utf-8")
        if f"def {function_name}(" in content:
            print(f"    ✓ 函数 {function_name}() 已定义")
            return True
        else:
            print(f"    ✗ 函数 {function_name}() 未找到")
            return False
    except Exception as e:
        print(f"    ✗ 读取文件失败: {e}")
        return False


def main() -> int:
    """主验证流程。"""
    arena_root = Path(__file__).parent
    print("验证 Arena PPO 评估器实现")
    print("=" * 60)
    print(f"Arena 根目录: {arena_root}\n")

    all_checks_passed = True

    # 1. 检查核心模块
    print("1. 核心评估模块")
    evaluator_path = arena_root / "src/mrs_arena/learning/arena_navigation_evaluator.py"
    if check_file_exists(evaluator_path, "评估器模块"):
        all_checks_passed &= check_function_in_file(evaluator_path, "run_arena_navigation_episode")
        all_checks_passed &= check_function_in_file(evaluator_path, "load_checkpoint_metadata")
        all_checks_passed &= check_function_in_file(evaluator_path, "make_arena_policy_wrapper")
    else:
        all_checks_passed = False

    print()

    # 2. 检查测试文件
    print("2. 单元测试")
    test_path = arena_root / "tests/unit/test_arena_navigation_evaluator.py"
    if check_file_exists(test_path, "单元测试文件"):
        all_checks_passed &= check_function_in_file(test_path, "test_imports_exist")
        all_checks_passed &= check_function_in_file(test_path, "test_run_episode_requires_single_env")
        all_checks_passed &= check_function_in_file(test_path, "test_load_checkpoint_metadata_validates_path")
    else:
        all_checks_passed = False

    print()

    # 3. 检查评估脚本
    print("3. 评估脚本")
    script_path = arena_root / "scripts/evaluate_arena_navigation_ppo.py"
    if check_file_exists(script_path, "评估脚本"):
        all_checks_passed &= check_function_in_file(script_path, "main")
    else:
        all_checks_passed = False

    print()

    # 4. 检查启动脚本
    print("4. 辅助脚本")
    run_script = arena_root / "scripts/run_arena_evaluation.sh"
    check_file_exists(run_script, "快速启动脚本")
    test_runner = arena_root / "run_evaluator_tests.py"
    check_file_exists(test_runner, "测试运行器")

    print()

    # 5. 检查文档
    print("5. 文档")
    doc_path = arena_root / "docs/arena_ppo_evaluation.md"
    check_file_exists(doc_path, "评估文档")

    print()
    print("=" * 60)

    if all_checks_passed:
        print("✓ 所有核心组件已实现")
        print("\n下一步:")
        print("  1. 运行单元测试: python3 run_evaluator_tests.py")
        print("  2. 评估检查点: ./scripts/run_arena_evaluation.sh <checkpoint.pt>")
        return 0
    else:
        print("✗ 部分组件缺失或不完整")
        return 1


if __name__ == "__main__":
    sys.exit(main())
