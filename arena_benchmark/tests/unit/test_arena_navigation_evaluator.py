"""模拟器无关的 Arena 导航评估器单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json
import sys
from pathlib import Path


class ArenaNavigationEvaluatorTest(unittest.TestCase):
    """测试 Arena 导航评估器的核心逻辑（不涉及 GPU/Kit）。"""

    def test_imports_exist(self) -> None:
        """验证评估模块可以导入。"""
        try:
            # 直接导入模块避免触发 mrs_arena.__init__
            import importlib.util
            module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
            spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
            if spec is None or spec.loader is None:
                self.fail(f"arena_navigation_evaluator module must exist at {module_path}")
            arena_navigation_evaluator = importlib.util.module_from_spec(spec)
            sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
            spec.loader.exec_module(arena_navigation_evaluator)
            self.assertTrue(hasattr(arena_navigation_evaluator, "run_arena_navigation_episode"))
        except ImportError as error:
            self.fail(f"arena_navigation_evaluator module must exist: {error}")

    def test_run_episode_requires_single_env(self) -> None:
        """评估要求单环境配置。"""
        import importlib.util
        module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
        spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
        arena_navigation_evaluator = importlib.util.module_from_spec(spec)
        sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
        spec.loader.exec_module(arena_navigation_evaluator)

        mock_env = MagicMock()
        mock_env.num_envs = 2
        mock_policy = MagicMock()

        with self.assertRaises(ValueError) as context:
            arena_navigation_evaluator.run_arena_navigation_episode(
                mock_env,
                mock_policy,
                target_xy=(1.0, 0.0),
                target_yaw_rad=0.0,
                asset_cfg=MagicMock(),
                index=0,
                seed=1000,
            )
        self.assertIn("exactly one env", str(context.exception))

    def test_run_episode_validates_max_length(self) -> None:
        """评估要求正的最大步数。"""
        import importlib.util
        module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
        spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
        arena_navigation_evaluator = importlib.util.module_from_spec(spec)
        sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
        spec.loader.exec_module(arena_navigation_evaluator)

        mock_env = MagicMock()
        mock_env.num_envs = 1
        mock_env.max_episode_length = 0
        mock_policy = MagicMock()

        with self.assertRaises(ValueError) as context:
            arena_navigation_evaluator.run_arena_navigation_episode(
                mock_env,
                mock_policy,
                target_xy=(1.0, 0.0),
                target_yaw_rad=0.0,
                asset_cfg=MagicMock(),
                index=0,
                seed=1000,
            )
        self.assertIn("must be positive", str(context.exception))

    def test_load_checkpoint_metadata_validates_path(self) -> None:
        """检查点加载器验证路径。"""
        import importlib.util
        module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
        spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
        arena_navigation_evaluator = importlib.util.module_from_spec(spec)
        sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
        spec.loader.exec_module(arena_navigation_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "missing.pt"
            with self.assertRaises(FileNotFoundError):
                arena_navigation_evaluator.load_checkpoint_metadata(nonexistent)

    def test_load_checkpoint_metadata_validates_manifest(self) -> None:
        """检查点加载器验证 run_manifest.json 存在。"""
        import importlib.util
        module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
        spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
        arena_navigation_evaluator = importlib.util.module_from_spec(spec)
        sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
        spec.loader.exec_module(arena_navigation_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint_file = checkpoint_dir / "model.pt"
            checkpoint_file.write_bytes(b"fake checkpoint")

            with self.assertRaises(FileNotFoundError):
                arena_navigation_evaluator.load_checkpoint_metadata(checkpoint_file)

    def test_load_checkpoint_metadata_validates_manifest_schema(self) -> None:
        """检查点加载器验证 manifest schema。"""
        import importlib.util
        module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
        spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
        arena_navigation_evaluator = importlib.util.module_from_spec(spec)
        sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
        spec.loader.exec_module(arena_navigation_evaluator)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint_file = checkpoint_dir / "model.pt"
            checkpoint_file.write_bytes(b"fake checkpoint")

            manifest_file = run_dir / "run_manifest.json"
            manifest_file.write_text(json.dumps({"status": "failed"}))

            with self.assertRaises(ValueError) as context:
                arena_navigation_evaluator.load_checkpoint_metadata(checkpoint_file)
            self.assertIn("status must be 'completed'", str(context.exception))

    def test_make_arena_policy_wrapper_builds_callable(self) -> None:
        """策略包装器将 RSL-RL 策略转换为 Arena 兼容接口。"""
        import importlib.util
        module_path = Path(__file__).parent.parent.parent / "src/mrs_arena/learning/arena_navigation_evaluator.py"
        spec = importlib.util.spec_from_file_location("arena_navigation_evaluator", module_path)
        arena_navigation_evaluator = importlib.util.module_from_spec(spec)
        sys.modules["arena_navigation_evaluator"] = arena_navigation_evaluator
        spec.loader.exec_module(arena_navigation_evaluator)

        mock_rsl_policy = MagicMock()
        mock_rsl_policy.return_value = MagicMock()

        wrapper = arena_navigation_evaluator.make_arena_policy_wrapper(mock_rsl_policy)
        self.assertTrue(callable(wrapper))


if __name__ == "__main__":
    unittest.main()
