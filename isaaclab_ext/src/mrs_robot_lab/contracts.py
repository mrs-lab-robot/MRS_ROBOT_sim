"""Reader and validator for the canonical MRS_ROBOT_sim embodiment contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


class ContractError(ValueError):
    """The robot interface contract is missing required fields or is inconsistent."""


@dataclass(frozen=True)
class EmbodimentContract:
    """Validated, simulator-independent view of the MRS robot contract."""

    robot_id: str
    action_dimensions: dict[str, int]
    action_fields: dict[str, tuple[str, ...]]
    action_joints: dict[str, tuple[str, ...]]
    action_limits: dict[str, tuple[tuple[float, float], ...]]
    action_units: dict[str, tuple[str, ...]]
    observation_dimensions: dict[str, int]
    observation_fields: dict[str, tuple[str, ...]]
    observation_units: dict[str, tuple[str, ...]]
    joint_action_names: tuple[str, ...]


def load_embodiment_contract(path: str | Path) -> EmbodimentContract:
    """Load a YAML contract and check dimensions used by Arena adapters.

    Action fields may describe either joint commands or non-joint commands such as
    base Twist. Only names under the joints key are exposed as joint action names.
    """

    contract_path = Path(path)
    try:
        with contract_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"could not read embodiment contract {contract_path}: {error}") from error

    if not isinstance(document, dict):
        raise ContractError("embodiment contract root must be a mapping")
    robot = document.get("robot")
    if not isinstance(robot, dict) or not isinstance(robot.get("id"), str):
        raise ContractError("embodiment contract must define robot.id")

    actions = document.get("actions")
    if not isinstance(actions, list):
        raise ContractError("embodiment contract must define actions as a list")
    action_dimensions: dict[str, int] = {}
    action_fields: dict[str, tuple[str, ...]] = {}
    action_joints: dict[str, tuple[str, ...]] = {}
    action_limits: dict[str, tuple[tuple[float, float], ...]] = {}
    action_units: dict[str, tuple[str, ...]] = {}
    joint_names: list[str] = []
    seen_joint_names: set[str] = set()
    for action in actions:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            raise ContractError("each action must define a name")
        name = action["name"]
        dimension = action.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ContractError(f"action {name} must define a positive integer dimension")
        if name in action_dimensions:
            raise ContractError(f"duplicate action name: {name}")
        action_dimensions[name] = dimension

        fields = action.get("fields")
        if fields is not None:
            _validate_dimension(name, dimension, fields)
            action_fields[name] = tuple(fields)
        joints = action.get("joints")
        if joints is not None:
            if not isinstance(joints, list) or not all(isinstance(joint, str) and joint for joint in joints):
                raise ContractError(f"action {name} joints must be a list of non-empty names")
            action_joints[name] = tuple(joints)
            if action.get("counts_toward_action", True):
                for joint in joints:
                    if joint not in seen_joint_names:
                        seen_joint_names.add(joint)
                        joint_names.append(joint)

        limits = action.get("limits")
        if limits is not None:
            action_limits[name] = _validate_action_limits(name, dimension, limits)

        units = action.get("units")
        if units is not None:
            if not isinstance(units, list) or len(units) != dimension:
                raise ContractError(f"action {name} units must contain exactly {dimension} entries")
            if not all(isinstance(u, str) and u for u in units):
                raise ContractError(f"action {name} units must be non-empty strings")
            action_units[name] = tuple(units)

    layouts = document.get("observation_layout")
    if not isinstance(layouts, list):
        raise ContractError("embodiment contract must define observation_layout as a list")
    observation_dimensions: dict[str, int] = {}
    observation_fields: dict[str, tuple[str, ...]] = {}
    observation_units: dict[str, tuple[str, ...]] = {}
    for layout in layouts:
        if not isinstance(layout, dict) or not isinstance(layout.get("observation"), str):
            raise ContractError("each observation layout must define an observation name")
        name = layout["observation"]
        dimension = layout.get("dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ContractError(f"observation {name} must define a positive integer dimension")
        if name in observation_dimensions:
            raise ContractError(f"duplicate observation layout: {name}")
        fields = layout.get("fields")
        _validate_dimension(name, dimension, fields)
        observation_dimensions[name] = dimension
        observation_fields[name] = tuple(fields)

        units = layout.get("units")
        if units is not None:
            if not isinstance(units, list) or len(units) != dimension:
                raise ContractError(f"observation {name} units must contain exactly {dimension} entries")
            if not all(isinstance(u, str) and u for u in units):
                raise ContractError(f"observation {name} units must be non-empty strings")
            observation_units[name] = tuple(units)

    return EmbodimentContract(
        robot_id=robot["id"],
        action_dimensions=action_dimensions,
        action_fields=action_fields,
        action_joints=action_joints,
        action_limits=action_limits,
        action_units=action_units,
        observation_dimensions=observation_dimensions,
        observation_fields=observation_fields,
        observation_units=observation_units,
        joint_action_names=tuple(joint_names),
    )


def _validate_dimension(name: str, dimension: int, fields: Any) -> None:
    if not isinstance(fields, list) or not all(isinstance(field, str) and field for field in fields):
        raise ContractError(f"{name} fields must be a list of non-empty names")
    if len(fields) != dimension:
        field_word = "field" if len(fields) == 1 else "fields"
        raise ContractError(f"{name} declares dimension {dimension} but has {len(fields)} {field_word}")


def _validate_action_limits(name: str, dimension: int, limits: Any) -> tuple[tuple[float, float], ...]:
    if not isinstance(limits, list) or len(limits) != dimension:
        raise ContractError(f"action {name} limits must contain exactly {dimension} entries")
    validated: list[tuple[float, float]] = []
    for index, limit in enumerate(limits):
        if not isinstance(limit, dict):
            raise ContractError(f"action {name} limit {index} must be a mapping")
        minimum = limit.get("minimum")
        maximum = limit.get("maximum")
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum >= maximum
        ):
            raise ContractError(f"action {name} limit {index} must have finite minimum < maximum")
        validated.append((float(minimum), float(maximum)))
    return tuple(validated)
