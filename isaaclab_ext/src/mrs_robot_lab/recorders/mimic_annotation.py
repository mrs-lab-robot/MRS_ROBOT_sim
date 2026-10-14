"""Per-step Mimic annotation collection for teleoperation episodes.

A1 of the mimic-redirect plan: while an episode records, capture the three
annotation groups the Isaac Lab Mimic data pipeline needs on top of the
existing action/joint/timestamp streams:

- object_pose: world pose of the primary manipulated object(s) — the same
  semantics as the Mimic API ``get_object_poses()`` (which returns rigid and
  articulated scene objects), restricted to the tracked objects of the task.
- eef_pose / target_eef_pose: bimanual end-effector world poses — the
  ``get_robot_eef_pose()`` semantics — plus the eef pose the current action
  commands, derived from the action via the contract's joint-target mapping
  and the arm Jacobian (``action_to_target_eef_pose()`` semantics for
  joint-space actions).
- subtask_signal: current subtask id plus per-subtask completion flags —
  the ``get_subtask_term_signals()`` semantics. The Arena task layer exposes
  its progress phase through ``env.extras["task_phase"]``.

Everything is collected from the live Arena scene directly, so recording
works before the per-task MimicEnv wrappers (A2/A3) land; the collector
mirrors the Mimic API names so those wrappers can replace the derivation
one function at a time.
"""

from __future__ import annotations

import math
from typing import Any

from mrs_robot_lab.adapters.teleop_adapter import (
    ACTION_JOINTS_BY_TERM,
    joint_targets_from_action,
)


EEF_BODY_NAMES: dict[str, str] = {
    "left": "openarmx_left_link7",
    "right": "openarmx_right_link7",
}

# task_id → (primary rigid-object scene key, subtask labels ordered by phase id).
# The label list is the subtask sequence the box-transport success logic
# (phase 0 before bilateral grasp, 1 after grasp, 2 after valid lift) and the
# dual-arm Mimic subtask configs describe; the final label has no completion
# signal (Mimic excludes the final subtask signal).
_TASK_OBJECT_HINTS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "openflex_dual_arm_box_transport": (
        "transport_box_rigid_body",
        ("approach_and_grasp", "lift", "carry_and_place"),
    ),
    "openflex_pick_cube": (
        "pick_cube",
        ("approach_and_lift", "settle"),
    ),
    "openflex_navigation_to_goal": (
        None,
        ("navigate_to_goal",),
    ),
}

_POSE_WIDTH = 7  # position xyz + quaternion xyzw (world frame)


def _finite_pose(position: Any, orientation: Any) -> list[float]:
    import torch

    pos = position.detach().to(device="cpu", dtype=torch.float32).reshape(3)
    quat = orientation.detach().to(device="cpu", dtype=torch.float32).reshape(4)
    values = [float(value) for value in (*pos, *quat)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("mimic annotation pose contains non-finite values")
    return values


def _resolve_eef_body_indices(robot: Any) -> dict[str, int]:
    """Map each arm side to its end-effector body index on the articulation."""

    resolution: dict[str, int] = {}
    index_by_name: dict[str, int] = {}
    find_bodies = getattr(robot, "find_bodies", None)
    if callable(find_bodies):
        try:
            names, indices = find_bodies([".*_link7$"])
            index_by_name = {str(name): int(index) for name, index in zip(names, indices)}
        except (RuntimeError, ValueError):
            index_by_name = {}
    for side, body_name in EEF_BODY_NAMES.items():
        index = index_by_name.get(body_name)
        if index is None:
            matches = [
                matched_index
                for name, matched_index in index_by_name.items()
                if name.endswith(body_name) or name.endswith(f"{side}_link7")
            ]
            if not matches:
                raise RuntimeError(f"robot has no {side} end-effector body ({body_name})")
            index = matches[0]
        resolution[side] = int(index)
    return resolution


def _arm_delta_targets(
    actions: list[float] | tuple[float, ...],
    current_by_name: dict[str, float],
) -> dict[str, list[float]]:
    """Return per-arm target-minus-current joint deltas for the left/right arms."""

    targets = joint_targets_from_action(list(actions), current_by_name)
    deltas: dict[str, list[float]] = {}
    for side, term in (("left", "left_arm_action"), ("right", "right_arm_action")):
        deltas[side] = [
            float(targets[joint_name]) - float(current_by_name[joint_name])
            for joint_name in ACTION_JOINTS_BY_TERM[term]
        ]
    return deltas


class MimicAnnotationCollector:
    """Collect one Mimic annotation sample per recorded simulation step."""

    def __init__(
        self,
        robot: Any,
        env: Any,
        *,
        task_id: str = "",
        subtask_labels: tuple[str, ...] | None = None,
    ) -> None:
        import torch  # noqa: F401  # fail fast before recording starts

        self._robot = robot
        self._env = env
        self._task_id = str(task_id or "")
        self._eef_indices = _resolve_eef_body_indices(robot)
        hint_key, hint_labels = _TASK_OBJECT_HINTS.get(self._task_id, (None, ()))
        self._primary_object_key = hint_key
        self._subtask_labels: tuple[str, ...] = (
            tuple(subtask_labels) if subtask_labels is not None else hint_labels
        )
        self.object_names: tuple[str, ...] = self._resolve_object_names()

    def _resolve_object_names(self) -> tuple[str, ...]:
        scene = self._env.unwrapped.scene
        rigid_objects = getattr(scene, "rigid_objects", None)
        if not rigid_objects:
            return ()
        names = sorted(str(name) for name in rigid_objects.keys())
        primary = self._primary_object_key
        if primary is not None and primary in names:
            return (primary,)
        tracked = tuple(name for name in names if "contact" not in name)
        return tracked or names

    @property
    def subtask_labels(self) -> tuple[str, ...]:
        return self._subtask_labels

    def collect(
        self,
        *,
        actions: list[float] | tuple[float, ...],
        current_by_name: dict[str, float],
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Capture one annotation sample aligned with the current recorded row."""

        sample: dict[str, Any] = {
            "object_pose": self._collect_object_poses(),
            "object_names": self.object_names,
            "subtask_labels": self._subtask_labels,
        }
        sample["eef_pose"], sample["target_eef_pose"] = self._collect_eef_poses(
            actions, current_by_name
        )
        subtask_signal, subtask_flags = self._collect_subtask_signal(extras)
        sample["subtask_signal"] = subtask_signal
        sample["subtask_term_flags"] = subtask_flags
        return sample

    def _collect_object_poses(self) -> list[list[float]]:
        import torch

        scene = self._env.unwrapped.scene
        origin = scene.env_origins[0].detach().to(device="cpu", dtype=torch.float32)
        poses: list[list[float]] = []
        for name in self.object_names:
            asset = scene[name]
            position = asset.data.root_pos_w[0].detach().to(
                device="cpu", dtype=torch.float32
            ) - origin
            quat = asset.data.root_quat_w[0].detach().to(device="cpu", dtype=torch.float32)
            poses.append(_finite_pose(position, quat))
        return poses

    def _collect_eef_poses(
        self,
        actions: list[float] | tuple[float, ...],
        current_by_name: dict[str, float],
    ) -> tuple[list[float], list[float]]:
        import torch
        import isaaclab.utils.math as pose_utils

        data = self._robot.data
        body_state = data.body_state_w
        origin = self._env.unwrapped.scene.env_origins[0].detach().to(
            device="cpu", dtype=torch.float32
        )
        eef_poses: list[float] = []
        target_poses: list[float] = []
        arm_delta_by_side = _arm_delta_targets(actions, current_by_name)
        joint_names = [str(name) for name in self._robot.joint_names]
        joint_index_by_name = {name: index for index, name in enumerate(joint_names)}
        jacobian = getattr(data, "jacobian", None)
        for side in ("left", "right"):
            body_index = self._eef_indices[side]
            state = body_state[0, body_index].detach().to(device="cpu", dtype=torch.float32)
            position = state[0:3] - origin
            quat = state[3:7]
            eef_poses.extend(_finite_pose(position, quat))

            # action_to_target_eef_pose() semantics for joint-space actions:
            # propagate the commanded joint delta through the arm Jacobian.
            target_position = position.clone()
            target_quat = quat.clone()
            if jacobian is not None:
                delta = torch.tensor(arm_delta_by_side[side], dtype=torch.float32)
                columns = [
                    joint_index_by_name[joint_name]
                    for joint_name in ACTION_JOINTS_BY_TERM[f"{side}_arm_action"]
                ]
                body_jacobian = jacobian[0, body_index, :, :][:, columns].detach().to(
                    device="cpu", dtype=torch.float32
                )
                twist = body_jacobian @ delta
                linear = twist[0:3]
                angular = twist[3:6]
                target_position = position + linear
                angle = float(torch.linalg.vector_norm(angular))
                if angle > 1e-9:
                    axis = (angular / angle).unsqueeze(0)
                    rotation_delta = pose_utils.quat_from_angle_axis(
                        torch.tensor([angle], dtype=torch.float32), axis
                    ).reshape(4)
                    target_quat = pose_utils.quat_mul(
                        rotation_delta.unsqueeze(0), quat.unsqueeze(0)
                    ).reshape(4)
            target_poses.extend(_finite_pose(target_position, target_quat))
        if len(eef_poses) != 2 * _POSE_WIDTH or len(target_poses) != 2 * _POSE_WIDTH:
            raise ValueError("bimanual eef annotation must contain both arms")
        return eef_poses, target_poses

    def _collect_subtask_signal(
        self, extras: dict[str, Any] | None
    ) -> tuple[int, list[int]]:
        flags = [0] * len(self._subtask_labels)
        phase = None
        if extras:
            raw_phase = extras.get("task_phase")
            if raw_phase is not None:
                try:
                    phase = int(raw_phase)
                except (TypeError, ValueError):
                    phase = None
        if phase is None or phase < 0:
            return -1, flags
        # Phase p means subtask p is active: subtasks 0..p-1 completed.
        for index in range(len(flags)):
            if phase >= index + 1:
                flags[index] = 1
        return phase, flags


def mimic_annotation_metadata(collector: "MimicAnnotationCollector") -> dict[str, Any]:
    """Episode-level metadata block describing the annotation group layout."""

    return {
        "enabled": True,
        "object_names": list(collector.object_names),
        "subtask_labels": list(collector.subtask_labels),
        "eef_body_names": dict(EEF_BODY_NAMES),
        "pose_layout": "position_xyz + quaternion_xyzw (world frame)",
        "eef_order": ["left", "right"],
        "version": 1,
    }


__all__ = [
    "EEF_BODY_NAMES",
    "MimicAnnotationCollector",
    "mimic_annotation_metadata",
]
