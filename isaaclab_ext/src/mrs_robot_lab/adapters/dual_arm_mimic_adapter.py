"""Isaac Lab Mimic 数据生成适配器：双臂箱子搬运任务

此模块提供双臂协作箱子搬运任务的 Mimic 环境配置。
遵循 Isaac Lab Mimic 和 Arena 的约定，为左右末端执行器定义子任务序列和协作约束。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 动态添加第三方库路径以支持单元测试环境
_sim_root = Path(__file__).resolve().parents[5]
_arena_path = _sim_root / "third_party" / "IsaacLab-Arena"
_isaaclab_path = _arena_path / "submodules" / "IsaacLab" / "source" / "isaaclab"
if _isaaclab_path.exists() and str(_isaaclab_path.parent) not in sys.path:
    sys.path.insert(0, str(_isaaclab_path.parent))
if _arena_path.exists() and str(_arena_path) not in sys.path:
    sys.path.insert(0, str(_arena_path))

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
    SubTaskConstraintCoordinationScheme,
)
from isaaclab.utils.configclass import configclass

try:
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
except ImportError:
    ArmMode = None
    MIMIC_DATAGEN_CONFIG_DEFAULTS = {
        "generation_guarantee": True,
        "generation_keep_failed": False,
        "generation_num_trials": 100,
        "generation_select_src_per_subtask": False,
        "generation_select_src_per_arm": False,
        "max_num_failures": 25,
        "seed": 1,
    }


def get_dual_arm_box_transport_mimic_cfg(
    arm_mode: ArmMode,
    box_object_name: str,
) -> MimicEnvCfg:
    """构建双臂箱子搬运任务的 Mimic 环境配置

    Args:
        arm_mode: 臂模式，必须为 DUAL_ARM
        box_object_name: 任务中搬运对象的名称

    Returns:
        配置完整的 MimicEnvCfg 实例

    Raises:
        ValueError: 如果 arm_mode 不是 DUAL_ARM 或 box_object_name 为空
    """
    if not box_object_name or not isinstance(box_object_name, str):
        raise ValueError("box_object_name 必须是非空字符串")

    if ArmMode is not None and arm_mode != ArmMode.DUAL_ARM:
        raise ValueError(f"双臂箱子搬运任务仅支持 DUAL_ARM 模式，当前为 {arm_mode}")

    return DualArmBoxTransportMimicEnvCfg(box_object_name=box_object_name)


@configclass
class DualArmBoxTransportMimicEnvCfg(MimicEnvCfg):
    """双臂箱子搬运任务的 Isaac Lab Mimic 环境配置

    此配置为双臂协作箱子搬运定义子任务序列和约束：
    - 左臂和右臂各自的子任务序列
    - 双臂在抓取阶段的协作约束
    - 基于最近邻的源演示选择策略
    """

    box_object_name: str = "transport_box"

    def __post_init__(self):
        super().__post_init__()

        # 配置数据生成参数
        self.datagen_config.name = "demo_src_dual_arm_box_transport_isaac_lab_D0"
        for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
            setattr(self.datagen_config, key, value)

        # 左臂子任务序列
        left_subtasks = [
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal="left_approach_box",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="左臂接近箱子",
                next_subtask_description="左臂抓取箱子",
            ),
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal="left_grasp_box",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="左臂抓取箱子",
                next_subtask_description="双臂协同搬运箱子",
            ),
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal="bimanual_transport_complete",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="双臂协同搬运箱子",
                next_subtask_description="左臂释放箱子",
            ),
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="左臂释放箱子并稳定",
                next_subtask_description="",
            ),
        ]

        # 右臂子任务序列（与左臂对称）
        right_subtasks = [
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal="right_approach_box",
                subtask_term_offset_range=(5, 15),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="右臂接近箱子",
                next_subtask_description="右臂抓取箱子",
            ),
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal="right_grasp_box",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="右臂抓取箱子",
                next_subtask_description="双臂协同搬运箱子",
            ),
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal="bimanual_transport_complete",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="双臂协同搬运箱子",
                next_subtask_description="右臂释放箱子",
            ),
            SubTaskConfig(
                object_ref=self.box_object_name,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.005,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="右臂释放箱子并稳定",
                next_subtask_description="",
            ),
        ]

        self.subtask_configs["left"] = left_subtasks
        self.subtask_configs["right"] = right_subtasks

        # 定义双臂协作约束
        # 约束1: 双臂抓取阶段需要协作（索引1对应grasp子任务）
        grasp_coordination = SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left", 1), ("right", 1)],
            constraint_type=SubTaskConstraintType.COORDINATION,
            coordination_scheme=SubTaskConstraintCoordinationScheme.REPLAY,
            coordination_scheme_pos_noise_scale=0.01,
            coordination_scheme_rot_noise_scale=0.05,
            coordination_synchronize_start=True,
        )

        # 约束2: 双臂搬运阶段需要协作（索引2对应transport子任务）
        transport_coordination = SubTaskConstraintConfig(
            eef_subtask_constraint_tuple=[("left", 2), ("right", 2)],
            constraint_type=SubTaskConstraintType.COORDINATION,
            coordination_scheme=SubTaskConstraintCoordinationScheme.REPLAY,
            coordination_scheme_pos_noise_scale=0.01,
            coordination_scheme_rot_noise_scale=0.05,
            coordination_synchronize_start=True,
        )

        self.task_constraint_configs = [grasp_coordination, transport_coordination]
