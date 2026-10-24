"""回归测试：NavigationTaskEnvironment._setup_scene 必须解析 ENV_REGEX_NS 占位符。

这个测试验证了一个具体的 bug：当 _setup_scene 手动构造 Articulation(robot_cfg) 时，
InteractiveScene 不会自动展开 {ENV_REGEX_NS}，因此传递给 Articulation 的路径必须是
使用当前场景命名空间解析后的绝对路径（例如 /World/envs/env_.*/Robot），
而不是未解析的 {ENV_REGEX_NS}/Robot。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from mrs_robot_lab.environments.learning.navigation_task import NavigationTaskEnvironment
    IMPORT_ERROR = None
except ImportError as error:
    NavigationTaskEnvironment = None
    IMPORT_ERROR = error


class NavigationPrimPathRegressionTest(unittest.TestCase):
    """验证 NavigationTaskEnvironment._setup_scene 正确解析 prim_path 占位符。"""

    def setUp(self) -> None:
        if IMPORT_ERROR is not None:
            self.skipTest(f"NavigationTaskEnvironment 无法导入: {IMPORT_ERROR}")

    @patch("mrs_robot_lab.environments.learning.navigation_task.Articulation")
    @patch("mrs_robot_lab.environments.learning.navigation_task.sim_utils.UsdFileCfg")
    def test_setup_scene_resolves_env_regex_ns_before_articulation_construction(
        self,
        mock_usd_cfg_class: MagicMock,
        mock_articulation_class: MagicMock,
    ) -> None:
        """_setup_scene 必须将 {ENV_REGEX_NS}/Robot 解析为绝对路径后再传递给 Articulation。

        回归场景：
        1. NavigationTaskEnvironment._setup_scene 在第117行设置 robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        2. 在第120行直接调用 Articulation(robot_cfg)
        3. 由于是手动构造而非通过 InteractiveSceneCfg 声明，InteractiveScene 不会展开占位符
        4. Articulation 必须接收解析后的绝对路径（例如 /World/envs/env_.*/Robot），而不是字面的 {ENV_REGEX_NS}

        该测试验证传递给 Articulation.__init__ 的 cfg.prim_path 是否已解析为绝对路径。
        """
        # 使用 object.__new__ 分配对象，不调用 __init__
        env = object.__new__(NavigationTaskEnvironment)
        env._is_closed = True  # 防止析构函数访问未初始化的属性

        # 准备最小化的 runtime_config
        mock_criterion = MagicMock()
        mock_criterion.criterion_type = "base_pose_at_target"
        mock_criterion.params = {"target_position": [1.0, 2.0, 0.0], "target_yaw_rad": 0.0}
        mock_criterion.tolerance = 0.5

        mock_task_spec = MagicMock()
        mock_task_spec.success_criteria = [mock_criterion]
        mock_task_spec.episode_length_s = 30.0
        mock_task_spec.initial_state = {"robot_pose": {"position": [0.0, 0.0, 0.5], "yaw_rad": 0.0}}
        mock_task_spec.reward_config = {}

        mock_runtime_config = MagicMock()
        mock_runtime_config.task = mock_task_spec
        mock_runtime_config.task_spec_path = Path("/fake/task.yaml")
        mock_runtime_config.scene_spec_path = Path("/fake/scene.yaml")
        mock_runtime_config.base_stage_path = Path("/fake/stage.usd")
        mock_runtime_config.scene.robot_spawn_pose = ([0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0])

        # 手动设置必要的实例属性
        env.runtime_config = mock_runtime_config
        env._initial_position = [0.0, 0.0, 0.5]
        env._initial_yaw = 0.0
        env._camera_frequencies_hz = {}

        # 创建模拟的场景对象
        mock_scene = MagicMock()
        mock_scene.env_regex_ns = r"/World/envs/env_.*"  # InteractiveScene 提供的命名空间正则
        mock_scene.env_ns = "/World/envs"
        mock_scene.articulations = {}
        mock_scene.clone_environments = MagicMock()
        env.scene = mock_scene

        # 模拟 UsdFileCfg 构造函数
        mock_usd_cfg_instance = MagicMock()
        mock_usd_cfg_instance.func = MagicMock()
        mock_usd_cfg_class.return_value = mock_usd_cfg_instance

        # 直接调用 _setup_scene
        env._setup_scene()

        # 验证 Articulation 被调用
        self.assertTrue(
            mock_articulation_class.called,
            "Articulation 构造函数应该在 _setup_scene 中被调用"
        )

        # 提取传递给 Articulation 的配置对象
        articulation_call_args = mock_articulation_class.call_args
        self.assertIsNotNone(articulation_call_args, "Articulation 必须被调用")

        robot_cfg = articulation_call_args[0][0]  # 第一个位置参数
        actual_prim_path = robot_cfg.prim_path

        # 关键断言：prim_path 必须是解析后的绝对路径，不能包含未解析的 {ENV_REGEX_NS}
        self.assertNotIn(
            "{ENV_REGEX_NS}",
            actual_prim_path,
            f"Articulation 接收到未解析的 prim_path: {actual_prim_path}。"
            f"_setup_scene 必须在构造 Articulation 之前将 {{ENV_REGEX_NS}} 解析为场景命名空间正则。"
        )

        # 验证路径格式：应该是 /World/envs/env_.*/Robot
        expected_path = "/World/envs/env_.*/Robot"
        self.assertEqual(
            actual_prim_path,
            expected_path,
            f"prim_path 应该是 {expected_path}，实际: {actual_prim_path}"
        )



if __name__ == "__main__":
    unittest.main()
