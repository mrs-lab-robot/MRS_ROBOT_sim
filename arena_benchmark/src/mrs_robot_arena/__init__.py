"""Compatibility namespace; project code now lives in :mod:`mrs_arena`."""

from mrs_arena import ContractError, EmbodimentContract, load_embodiment_contract, __version__

__all__ = ["ContractError", "EmbodimentContract", "load_embodiment_contract", "__version__"]
