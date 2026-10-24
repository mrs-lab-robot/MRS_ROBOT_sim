"""Arena registration that composes the robot configuration owned by mrs_robot_lab."""

from __future__ import annotations

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg

from mrs_robot_lab.assets.asset_resolver import AssetResolver
from mrs_robot_lab.assets.mrs_robot_cfg import (
    JOINT_STATE_NAMES,
    MRS_ROBOT_CFG,
    OpenFlexActionsCfg,
    OpenFlexObservationsCfg,
)
from mrs_robot_lab.contracts import ContractError, load_embodiment_contract
from mrs_robot_lab.sensors.camera_cfg import build_camera_cfgs


# 相机配置**懒加载**：原来这里是模块级 `_OPENFLEX_CAMERA_CFGS = build_camera_cfgs()`，
# import 本模块就会读挂点 YAML + 标定、构造 4 个 CameraCfg —— 任何一个文件缺失/解析
# 失败都会让 import 直接炸（实测会让 Kit 启动阶段的 import 链提前退出），也无法按
# run 改传感器集合（采集只要 2 路时也只能吃这 4 路）。改成首次访问时构造并缓存。
_CAMERA_CFG_CACHE: dict[str, CameraCfg] | None = None


def _openflex_camera_cfgs() -> dict[str, CameraCfg]:
    """按运行期挂点契约构建 4 路相机配置（首次调用时读 YAML，之后走缓存）。"""

    global _CAMERA_CFG_CACHE
    if _CAMERA_CFG_CACHE is None:
        _CAMERA_CFG_CACHE = build_camera_cfgs()
    return _CAMERA_CFG_CACHE


@configclass
class OpenFlexSceneCfg:
    """Robot articulation slot merged into an Arena scene."""

    robot: ArticulationCfg | None = None


@configclass
class OpenFlexEventCfg:
    """No custom reset randomization is enabled in the initial embodiment."""

    pass


@configclass
class OpenFlexCameraCfg(ArenaCameraCfg):
    """Four RGB/depth cameras sourced from the canonical runtime mount contract."""

    # 字段默认值留 None，实际配置在 __post_init__ 里懒加载填进来。
    base_d435: CameraCfg | None = None
    head_d435: CameraCfg | None = None
    left_wrist_d405: CameraCfg | None = None
    right_wrist_d405: CameraCfg | None = None

    def __post_init__(self) -> None:
        cfgs = _openflex_camera_cfgs()
        for name in ("base_d435", "head_d435", "left_wrist_d405", "right_wrist_d405"):
            setattr(self, name, cfgs[name])


@register_asset
class OpenFlexEmbodiment(EmbodimentBase):
    """Dual-arm OpenFleX embodiment using the shared Isaac Lab robot config."""

    name = "openflex"
    tags = ["embodiment", "dual_arm", "openflex"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose=None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        collision_mode=None,
    ):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            concatenate_observation_terms=concatenate_observation_terms,
            arm_mode=arm_mode,
            collision_mode=collision_mode,
        )
        contract = load_embodiment_contract(AssetResolver().resolve("openflex_embodiment_contract"))
        if contract.robot_id != "openflex":
            raise ContractError(f"expected the OpenFleX contract, got robot id {contract.robot_id!r}")
        position_fields = tuple(f"position.{name}" for name in JOINT_STATE_NAMES)
        velocity_fields = tuple(f"velocity.{name}" for name in JOINT_STATE_NAMES)
        if contract.observation_fields.get("joint_state") != position_fields + velocity_fields:
            raise ContractError("joint_state field order differs from the shared Isaac Lab observation mapping")

        self.scene_config = OpenFlexSceneCfg()
        self.scene_config.robot = MRS_ROBOT_CFG.copy()
        self.camera_config = OpenFlexCameraCfg()
        self.action_config = OpenFlexActionsCfg()
        self.observation_config = OpenFlexObservationsCfg()
        self.observation_config.policy.concatenate_terms = concatenate_observation_terms
        self.event_config = OpenFlexEventCfg()
