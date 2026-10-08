#!/usr/bin/env python3
"""运行 arena_navigation_evaluator 的模拟器无关单元测试。"""

import subprocess
import sys
from pathlib import Path

def main():
    arena_benchmark_root = Path(__file__).parent
    test_file = "tests/unit/test_arena_navigation_evaluator.py"

    print(f"运行 Arena 导航评估器单元测试...")
    print(f"工作目录: {arena_benchmark_root}")
    print(f"测试文件: {test_file}\n")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/unit",
            "-p",
            "test_arena_navigation_evaluator.py",
            "-v",
        ],
        cwd=arena_benchmark_root,
        env={"PYTHONPATH": str(arena_benchmark_root / "src")},
        capture_output=False,
    )

    if result.returncode == 0:
        print("\n✓ 所有测试通过")
    else:
        print(f"\n✗ 测试失败 (exit code: {result.returncode})")

    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
