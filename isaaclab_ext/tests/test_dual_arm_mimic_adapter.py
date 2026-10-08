"""测试双臂箱子搬运任务的 Mimic 数据生成适配器"""

from __future__ import annotations

import unittest


class DualArmMimicAdapterTest(unittest.TestCase):
    """验证 isaaclab_ext 提供的双臂 Mimic 配置符合 Isaac Lab Mimic 和 Arena 约定"""

    def test_mimic_config_exists_for_dual_arm_mode(self):
        """双臂模式下必须返回有效的 MimicEnvCfg"""
        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="transport_box")

        self.assertIsInstance(cfg, MimicEnvCfg)
        self.assertIsNotNone(cfg.datagen_config)
        self.assertIsNotNone(cfg.subtask_configs)

    def test_dual_arm_mode_defines_left_and_right_eef_subtasks(self):
        """双臂模式必须为左右末端执行器各定义子任务列表"""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="transport_box")

        self.assertIn("left", cfg.subtask_configs)
        self.assertIn("right", cfg.subtask_configs)
        self.assertIsInstance(cfg.subtask_configs["left"], list)
        self.assertIsInstance(cfg.subtask_configs["right"], list)
        self.assertGreater(len(cfg.subtask_configs["left"]), 0)
        self.assertGreater(len(cfg.subtask_configs["right"]), 0)

    def test_subtasks_reference_transport_box_object(self):
        """每个子任务必须引用任务中配置的搬运箱子对象"""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        box_name = "test_transport_box"
        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name=box_name)

        for eef_name in ["left", "right"]:
            for subtask in cfg.subtask_configs[eef_name]:
                self.assertEqual(subtask.object_ref, box_name)

    def test_left_and_right_arms_have_matching_subtask_count(self):
        """左右臂的子任务数量必须匹配，以支持协作约束"""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="transport_box")

        left_count = len(cfg.subtask_configs["left"])
        right_count = len(cfg.subtask_configs["right"])
        self.assertEqual(left_count, right_count, "左右臂子任务数量必须相等以支持双臂协作")

    def test_coordination_constraints_exist_for_dual_arm_grasping(self):
        """双臂抓取阶段必须定义协作约束"""
        from isaaclab.envs.mimic_env_cfg import SubTaskConstraintType
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="transport_box")

        self.assertGreater(len(cfg.task_constraint_configs), 0, "双臂任务必须定义至少一个协作约束")
        has_coordination = any(
            constraint.constraint_type == SubTaskConstraintType.COORDINATION
            for constraint in cfg.task_constraint_configs
        )
        self.assertTrue(has_coordination, "必须包含 COORDINATION 类型的约束以同步左右臂动作")

    def test_each_subtask_has_valid_termination_signal_or_none(self):
        """每个子任务必须定义有效的终止信号名称或显式为 None（最后一个子任务）"""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="transport_box")

        for eef_name, subtasks in cfg.subtask_configs.items():
            for index, subtask in enumerate(subtasks):
                is_last = index == len(subtasks) - 1
                if is_last:
                    self.assertIsNone(
                        subtask.subtask_term_signal,
                        f"{eef_name} 的最后一个子任务的终止信号必须为 None"
                    )
                else:
                    self.assertIsInstance(
                        subtask.subtask_term_signal,
                        str,
                        f"{eef_name} 的第 {index} 个子任务必须定义终止信号"
                    )
                    self.assertGreater(
                        len(subtask.subtask_term_signal),
                        0,
                        f"{eef_name} 的第 {index} 个子任务的终止信号不能为空字符串"
                    )

    def test_adapter_validates_box_object_name_parameter(self):
        """适配器必须验证 box_object_name 参数不为空"""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        with self.assertRaises(ValueError) as context:
            get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="")

        self.assertIn("box_object_name", str(context.exception))

    def test_datagen_config_uses_reasonable_defaults(self):
        """数据生成配置必须设置合理的默认值"""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from mrs_robot_lab.adapters.dual_arm_mimic_adapter import get_dual_arm_box_transport_mimic_cfg

        cfg = get_dual_arm_box_transport_mimic_cfg(arm_mode=ArmMode.DUAL_ARM, box_object_name="transport_box")

        self.assertTrue(cfg.datagen_config.generation_guarantee)
        self.assertFalse(cfg.datagen_config.generation_keep_failed)
        self.assertGreater(cfg.datagen_config.generation_num_trials, 0)
        self.assertGreaterEqual(cfg.datagen_config.seed, 0)
        self.assertGreater(cfg.datagen_config.max_num_failures, 0)


if __name__ == "__main__":
    unittest.main()
