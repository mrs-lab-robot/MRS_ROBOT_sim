"""Base-only OpenFlex Arena interface matching the navigation PPO contract."""

from __future__ import annotations

from isaaclab.managers import ActionTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode

from mrs_robot_lab.actions.base_actions import SwerveBaseActionCfg
from mrs_arena.embodiments.openflex import OpenFlexEmbodiment
from mrs_arena.adapters.navigation_observation import navigation_policy_observation
from mrs_arena.tasks.navigation_logic import (
    NavigationParameters,
)


@configclass
class OpenFlexNavigationActionsCfg:
    """Only the three physical body-twist commands used by the PPO policy."""

    base_twist_action: ActionTermCfg = SwerveBaseActionCfg(asset_name="robot")


@configclass
class OpenFlexNavigationObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        navigation_state: ObservationTermCfg = ObservationTermCfg(
            func=navigation_policy_observation,
            params={
                "target_xy": (2.0, 0.0),
                "target_yaw_rad": 0.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@register_asset
class OpenFlexNavigationEmbodiment(OpenFlexEmbodiment):
    """OpenFlex robot with the exact 8-observation/3-action navigation interface."""

    name = "openflex_navigation"
    tags = ["embodiment", "mobile_base", "navigation", "openflex"]
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(self, parameters: NavigationParameters, initial_pose, enable_cameras: bool = False):
        super().__init__(
            enable_cameras=enable_cameras,
            initial_pose=initial_pose,
            concatenate_observation_terms=True,
            arm_mode=ArmMode.DUAL_ARM,
        )
        self.action_config = OpenFlexNavigationActionsCfg()
        self.observation_config = OpenFlexNavigationObservationsCfg()
        term_params = self.observation_config.policy.navigation_state.params
        term_params["target_xy"] = parameters.target_xy
        term_params["target_yaw_rad"] = parameters.target_yaw_rad
