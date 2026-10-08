"""Runtime-neutral helpers for first-party Arena policy evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def make_episode_result(
    *,
    task_id: str,
    index: int,
    seed: int,
    steps: int,
    success: bool,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": str(task_id),
        "index": int(index),
        "seed": int(seed),
        "steps": int(steps),
        "success": bool(success),
        "extras": dict(extras or {}),
    }


def policy_input(observations: Any) -> Any:
    """Select Arena's policy observation group while preserving other APIs."""
    if isinstance(observations, Mapping) and "policy" in observations:
        return observations["policy"]
    return observations


def single_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if value.numel() != 1:
            raise ValueError("expected one boolean value")
        return bool(value.item())
    except AttributeError as error:
        raise TypeError("termination value must be bool or a scalar tensor") from error
