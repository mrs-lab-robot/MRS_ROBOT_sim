"""Arena-side adapter over the single bilateral box success criterion.

The criterion for "grasp the box with both grippers, lift, carry, place, release
and let it settle" has exactly **one** implementation, and it lives in the
learning layer:
:class:`mrs_robot_lab.environments.learning.task_logic_torch.BatchedBoxTransportEvaluator`
(the one the training env uses). This module keeps the arena benchmark's public
API — :func:`create_transport_progress_state` / :func:`update_transport_progress`
— but only translates inputs and re-orders the return value. It must never grow
a second copy of the state machine: if the two backends disagree, the same task
stops being comparable between lab training and arena evaluation.

Layering: arena may import lab, never the other way round. That is why the shared
criterion lives in ``mrs_robot_lab`` and the delegation happens here.
"""

from __future__ import annotations

import math

import torch

from mrs_robot_lab.environments.learning.task_logic_torch import BatchedBoxTransportEvaluator


# arena 的调用方在 create 时还没有阈值（阈值是每步传进来的），而 lab 的 evaluator
# 在构造时就要求一套合法参数。这里先给一组占位值，只为让它拥有缓冲区；
# 第一次 update 一定会先被真实参数覆盖（见 update_transport_progress）。
_PENDING_CRITERIA = {
    "position_tolerance_m": 1e-3,
    "min_lift_height_m": 1e-3,
    "min_carry_distance_m": 1e-3,
    "stable_seconds": 1e-3,
    "max_linear_speed_mps": 1e-3,
}

_STATE_KEYS = (
    "bilateral_grasp_seen",
    "lift_seen",
    "carry_seen",
    "placed_seen",
    "release_stable_seconds",
)


class TransportProgressState(dict):
    """Per-episode state for the bilateral box task.

    The five tensors are **live aliases** of the lab evaluator's buffers, not
    copies. arena 的调用方用「把 state 里的张量清零」来表示 reset
    （``value.zero_()`` / ``value[env_ids] = 0``），别名让这种清零直接落在唯一
    的那份历史上。不要重新赋值这些键，也不要在别处另建一份计数器。
    """

    def __init__(self, evaluator: BatchedBoxTransportEvaluator) -> None:
        super().__init__(
            bilateral_grasp_seen=evaluator.has_grasped,
            lift_seen=evaluator.has_lifted,
            carry_seen=evaluator.has_carried,
            placed_seen=evaluator.has_placed,
            release_stable_seconds=evaluator.stable_elapsed,
        )
        self.evaluator = evaluator
        self.criteria: dict[str, float] | None = None
        self.target_position: tuple[float, float, float] | None = None


def create_transport_progress_state(
    num_envs: int, *, device: str | torch.device
) -> TransportProgressState:
    """Create the per-episode state consumed by :func:`update_transport_progress`.

    The returned dict carries the same five keys as before (plus ``placed_seen``,
    which the lab criterion tracks and the old arena copy dropped). Thresholds and
    the target pose are supplied per step, so they are not needed here.
    """
    if isinstance(num_envs, bool) or int(num_envs) <= 0:
        raise ValueError("num_envs must be a positive integer")
    evaluator = BatchedBoxTransportEvaluator(
        initial_position=(0.0, 0.0, 0.0),
        target_position=(0.0, 0.0, 0.0),
        num_envs=int(num_envs),
        device=device,
        **_PENDING_CRITERIA,
    )
    return TransportProgressState(evaluator)


def update_transport_progress(
    state: TransportProgressState,
    *,
    box_position: torch.Tensor,
    box_linear_speed: torch.Tensor,
    initial_position: torch.Tensor,
    target_position: tuple[float, float, float],
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    dt: float,
    min_lift_height_m: float,
    min_carry_distance_m: float,
    position_tolerance_m: float,
    max_linear_speed_mps: float,
    stable_seconds: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the shared criterion and return ``(success, phase)`` per environment.

    This return order is the fixed arena contract (``success`` first); the lab
    evaluator returns ``(phase, success)`` and the swap happens here and nowhere
    else. Both tensors are shape ``(num_envs,)``, batch dimension first.

    ``success`` (``bool``): the box was grasped by **both** grippers, lifted above
    ``min_lift_height_m``, carried ``min_carry_distance_m`` horizontally, placed
    within ``position_tolerance_m`` of the target, released by **both** grippers,
    and then held at the target slower than ``max_linear_speed_mps`` for
    ``stable_seconds``.

    ``phase`` (``int64``) is the lab progress ladder, 0..6 — a monotone progress
    indicator, **not** a subtask index (it never goes backwards inside an episode
    and a single step may skip levels):

    * ``0`` APPROACH — no bilateral grasp seen yet
    * ``1`` GRASPED  — both grippers have contacted the box
    * ``2`` LIFTED   — box rose ``min_lift_height_m`` above its initial pose
    * ``3`` CARRIED  — box moved ``min_carry_distance_m`` horizontally
    * ``4`` PLACED   — box is within ``position_tolerance_m`` of the target
    * ``5`` RELEASED — placed, and both grippers have let go
    * ``6`` SUCCESS  — released and settled at the target (``success`` is True)

    ``initial_position`` is the per-environment reference pose (shape
    ``(num_envs, 3)``); ``box_linear_speed`` is a non-negative scalar speed per
    environment, which is what arena's terminations observe.
    """

    _validate_inputs(
        state,
        box_position,
        box_linear_speed,
        initial_position,
        left_contact,
        right_contact,
        dt,
        min_lift_height_m,
        min_carry_distance_m,
        position_tolerance_m,
        max_linear_speed_mps,
        stable_seconds,
    )
    target = _target_position(target_position)
    evaluator = state.evaluator
    criteria = {
        "min_lift_height_m": float(min_lift_height_m),
        "min_carry_distance_m": float(min_carry_distance_m),
        "position_tolerance_m": float(position_tolerance_m),
        "max_linear_speed_mps": float(max_linear_speed_mps),
        "stable_seconds": float(stable_seconds),
    }
    if state.criteria != criteria or state.target_position != target:
        # 只写参数，不动历史：arena 每步都传阈值，但真正变更只应发生在配置切换时。
        evaluator.configure(target_position=target, **criteria)
        state.criteria = criteria
        state.target_position = target
    reference = initial_position.to(device=evaluator.device, dtype=torch.float32)
    if not torch.equal(evaluator.initial_position, reference):
        evaluator.configure(initial_position=reference)

    # lab 的判据只吃速度向量（取模长）。把标量速度放在 x 轴上：IEEE-754 下
    # sqrt(round(s*s)) == s，所以模长与原标量逐位相等，不会在阈值边界上引入偏差。
    velocity = torch.zeros_like(box_position, dtype=torch.float32)
    velocity[:, 0] = box_linear_speed.to(device=box_position.device, dtype=torch.float32)

    phase, success = evaluator.update(
        box_position,
        velocity,
        left_contact,
        right_contact,
        dt=dt,
    )
    return success, phase


def _target_position(values: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError("target_position must contain three finite values")
    result = tuple(float(component) for component in values)
    if not all(math.isfinite(component) for component in result):
        raise ValueError("target_position must contain three finite values")
    return result  # type: ignore[return-value]


def _validate_inputs(
    state: TransportProgressState,
    box_position: torch.Tensor,
    box_linear_speed: torch.Tensor,
    initial_position: torch.Tensor,
    left_contact: torch.Tensor,
    right_contact: torch.Tensor,
    dt: float,
    min_lift_height_m: float,
    min_carry_distance_m: float,
    position_tolerance_m: float,
    max_linear_speed_mps: float,
    stable_seconds: float,
) -> None:
    if not isinstance(state, TransportProgressState) or set(state) != set(_STATE_KEYS):
        raise ValueError("transport state must come from create_transport_progress_state")
    if box_position.ndim != 2 or box_position.shape[1] != 3:
        raise ValueError("box_position must have shape (num_envs, 3)")
    count = box_position.shape[0]
    if initial_position.shape != box_position.shape:
        raise ValueError("initial_position must have the same shape as box_position")
    if box_linear_speed.shape != (count,) or left_contact.shape != (count,) or right_contact.shape != (count,):
        raise ValueError("speed and contact inputs must have shape (num_envs,)")
    if any(value.shape != (count,) for value in state.values()):
        raise ValueError("transport state environment count does not match input tensors")
    numeric = (
        dt,
        min_lift_height_m,
        min_carry_distance_m,
        position_tolerance_m,
        max_linear_speed_mps,
        stable_seconds,
    )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("transport thresholds must be finite")
    if dt <= 0 or min_lift_height_m <= 0 or min_carry_distance_m <= 0 or position_tolerance_m <= 0:
        raise ValueError("time step, lift, carry, and position thresholds must be positive")
    if max_linear_speed_mps < 0 or stable_seconds <= 0:
        raise ValueError("speed threshold must be non-negative and stable_seconds positive")
    if not bool(torch.isfinite(box_position).all().item()) or not bool(torch.isfinite(initial_position).all().item()):
        raise ValueError("box positions must be finite")
    if not bool(torch.isfinite(box_linear_speed).all().item()) or bool((box_linear_speed < 0).any().item()):
        raise ValueError("box_linear_speed must be finite and non-negative")
