"""Adapters from external teleoperation commands to Isaac Lab actions."""

from mrs_robot_lab.adapters.dual_arm_mimic_adapter import (
    DualArmBoxTransportMimicEnvCfg,
    get_dual_arm_box_transport_mimic_cfg,
)

__all__ = [
    "DualArmBoxTransportMimicEnvCfg",
    "get_dual_arm_box_transport_mimic_cfg",
]
