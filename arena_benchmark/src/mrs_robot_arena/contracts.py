"""Compatibility import for the contract reader moved to ``mrs_robot_lab``."""

from mrs_robot_lab.contracts import ContractError, EmbodimentContract, load_embodiment_contract

__all__ = ["ContractError", "EmbodimentContract", "load_embodiment_contract"]
