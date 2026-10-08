"""MRS robot integrations for Isaac Lab-Arena.

The package root intentionally imports no Isaac Sim or Kit modules. Import
simulator-dependent embodiment modules only after SimulationApp is running.
"""

from mrs_robot_lab.contracts import ContractError, EmbodimentContract, load_embodiment_contract

__version__ = "0.1.0"
__all__ = ["ContractError", "EmbodimentContract", "load_embodiment_contract", "__version__"]
