"""Collect existing ROS controller outputs into the Arena UDP command schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class RelayCommandFrame:
    seq: int
    source_time_ns: int
    session_id: int
    deadman: bool
    estop: bool
    base_twist: tuple[float, float, float]
    left_controller_pose: tuple[float, ...] | None
    right_controller_pose: tuple[float, ...] | None
    left_grip: tuple[float, ...] | None
    right_grip: tuple[float, ...] | None
    left_trigger: tuple[float, ...] | None
    right_trigger: tuple[float, ...] | None
    left_joystick: tuple[float, ...] | None
    right_joystick: tuple[float, ...] | None
    button_states: dict[str, bool]
    left_arm_position: tuple[float, ...] | None
    right_arm_position: tuple[float, ...] | None
    lift_position: tuple[float, ...] | None
    lift_velocity: tuple[float, ...] | None
    lift_action_position: tuple[float, ...] | None
    head_position: tuple[float, ...] | None
    left_gripper_position: tuple[float, ...] | None
    right_gripper_position: tuple[float, ...] | None

    def to_dict(self) -> dict:
        values = {
            "magic": "MRSAT",
            "version": 1,
            "type": "command",
            "session_id": self.session_id,
            "seq": self.seq,
            "source_time_ns": self.source_time_ns,
            "deadman": self.deadman,
            "estop": self.estop,
        }
        for name in (
            "base_twist", "left_controller_pose", "right_controller_pose", "left_grip", "right_grip",
            "left_trigger", "right_trigger", "left_joystick", "right_joystick", "left_arm_position", "right_arm_position", "lift_position", "lift_velocity",
            "head_position", "left_gripper_position", "right_gripper_position",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = list(value)
        values["button_states"] = dict(self.button_states)
        return values


class CommandAggregator:
    """ROS-independent stateful aggregation logic, unit-testable without ROS."""

    def __init__(
        self,
        *,
        deadman_timeout: float = 0.25,
        session_id: int = 1,
        control_mode: str = "vr",
    ) -> None:
        self.deadman_timeout = float(deadman_timeout)
        self.session_id = int(session_id)
        if control_mode not in ("vr", "ros"):
            raise ValueError("control_mode must be 'vr' or 'ros'")
        self.control_mode = control_mode
        self.seq = 0
        self.last_vr_pose_time: float | None = None
        self.last_base_twist_time: float | None = None
        self.last_lift_velocity_time: float | None = None
        self.estop = False
        self.base_twist = (0.0, 0.0, 0.0)
        self.left_controller_pose = None
        self.right_controller_pose = None
        self.left_grip = None
        self.right_grip = None
        self.left_trigger = None
        self.right_trigger = None
        self.left_joystick = None
        self.right_joystick = None
        self.button_states = {}
        self.left_arm_position = None
        self.right_arm_position = None
        self.left_gripper_position = None
        self.right_gripper_position = None
        self.lift_position = None
        self.lift_velocity = None
        self.head_position = None
        self._lift_action_target: float | None = None
        self._latest_lift_position: float | None = None
        self._lift_action_velocity = 0.0
        self._lift_action_velocity_active = False
        self._lift_action_last_command_time: float | None = None
        self._lift_action_last_update_time: float | None = None
        self._lift_action_min_position = -0.650
        self._lift_action_max_position = 0.300
        self._lift_action_max_velocity = 0.100
        self._lift_action_command_timeout = 0.250

    @staticmethod
    def _values(values, count: int, label: str) -> tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if len(result) != count:
            raise ValueError(f"{label} must contain {count} values")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{label} values must be finite")
        return result

    def update_vr_pose(self, *, now: float) -> None:
        self.last_vr_pose_time = float(now)

    def update_estop(self, active: bool) -> None:
        self.estop = bool(active)

    def update_controller_pose(self, side: str, values) -> None:
        pose = self._values(values, 7, f"{side}_controller_pose")
        if side == "left":
            self.left_controller_pose = pose
        elif side == "right":
            self.right_controller_pose = pose
        else:
            raise ValueError(f"unknown controller side: {side}")

    def update_hand_value(self, side: str, field: str, value: float) -> None:
        target = f"{side}_{field}"
        if side not in ("left", "right") or field not in ("grip", "trigger"):
            raise ValueError(f"unsupported VR input: {target}")
        setattr(self, target, self._values((value,), 1, target))

    def update_joystick(self, side: str, x: float, y: float) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"unknown controller side: {side}")
        setattr(self, f"{side}_joystick", self._values((x, y), 2, f"{side}_joystick"))

    def update_button(self, name: str, pressed: bool) -> None:
        self.button_states[str(name)] = bool(pressed)

    def update_base_twist(
        self, vx: float, vy: float, yaw_rate: float, *, now: float | None = None
    ) -> None:
        self.base_twist = self._values((vx, vy, yaw_rate), 3, "base_twist")
        self.last_base_twist_time = time.monotonic() if now is None else float(now)

    def update_arm(self, side: str, values) -> None:
        arm_values = self._values(values, len(values), f"{side}_arm_position")
        if len(arm_values) not in (7, 8):
            raise ValueError(f"{side} arm command must contain 7 joints and optional gripper")
        position = arm_values[:7]
        gripper = (max(0.0, min(0.044, arm_values[7])),) if len(arm_values) == 8 else None
        if side == "left":
            self.left_arm_position, self.left_gripper_position = position, gripper or self.left_gripper_position
        elif side == "right":
            self.right_arm_position, self.right_gripper_position = position, gripper or self.right_gripper_position
        else:
            raise ValueError(f"unknown arm side: {side}")

    def update_head(self, values) -> None:
        self.head_position = self._values(values, 2, "head_position")

    def update_lift_velocity(self, value: float, *, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        velocity = self._values((value,), 1, "lift_velocity")[0]
        self.lift_velocity = (velocity,)
        self.last_lift_velocity_time = timestamp
        bounded_velocity = max(
            -self._lift_action_max_velocity,
            min(self._lift_action_max_velocity, velocity),
        )
        self._lift_action_last_command_time = timestamp
        if abs(bounded_velocity) > 1.0e-9:
            if not self._lift_action_velocity_active:
                if self._latest_lift_position is not None:
                    self._lift_action_target = self._latest_lift_position
            self._lift_action_velocity_active = True
            self._lift_action_velocity = bounded_velocity
            if self._lift_action_last_update_time is None:
                self._lift_action_last_update_time = timestamp
        else:
            self._lift_action_velocity_active = False
            self._lift_action_velocity = 0.0
            self._lift_action_last_update_time = timestamp

    def update_lift_state(self, position: float, *, now: float | None = None) -> float:
        """Track simulated lift state and expose its current absolute action label."""
        timestamp = time.monotonic() if now is None else float(now)
        observed = self._values((position,), 1, "lift_state")[0]
        self._latest_lift_position = observed
        if not self._lift_action_velocity_active:
            self._lift_action_target = self._bound_lift_position(observed)
            self._lift_action_last_update_time = timestamp
        return self._lift_action_target if self._lift_action_target is not None else observed

    def update_lift_position(self, value: float) -> None:
        position = self._values((value,), 1, "lift_position")[0]
        self.lift_position = (position,)
        self._lift_action_target = self._bound_lift_position(position)
        self._lift_action_velocity_active = False
        self._lift_action_velocity = 0.0
        self._lift_action_last_update_time = time.monotonic()

    def _bound_lift_position(self, position: float) -> float:
        return max(
            self._lift_action_min_position,
            min(self._lift_action_max_position, float(position)),
        )

    def _step_lift_action(self, now: float) -> tuple[float, ...] | None:
        target = self._lift_action_target
        if target is None:
            return None
        previous = self._lift_action_last_update_time
        elapsed = 0.0 if previous is None else max(0.0, min(0.1, now - previous))
        self._lift_action_last_update_time = now
        if self._lift_action_velocity_active:
            command_time = self._lift_action_last_command_time
            if (
                command_time is None
                or now - command_time > self._lift_action_command_timeout
            ):
                self._lift_action_velocity_active = False
                self._lift_action_velocity = 0.0
            else:
                target = self._bound_lift_position(
                    target + self._lift_action_velocity * elapsed
                )
                self._lift_action_target = target
        return (target,)

    def command_frame(self, *, now: float, source_time_ns: int | None = None) -> RelayCommandFrame:
        self.seq += 1
        lift_action_position = self._step_lift_action(float(now))
        fresh = self.last_vr_pose_time is not None and now - self.last_vr_pose_time <= self.deadman_timeout
        base_fresh = (
            self.last_base_twist_time is not None
            and now - self.last_base_twist_time <= self.deadman_timeout
        )
        lift_fresh = (
            self.last_lift_velocity_time is not None
            and now - self.last_lift_velocity_time <= self.deadman_timeout
        )
        return RelayCommandFrame(
            seq=self.seq,
            source_time_ns=source_time_ns if source_time_ns is not None else 0,
            session_id=self.session_id,
            deadman=bool((self.control_mode == "ros" or fresh) and not self.estop),
            estop=self.estop,
            base_twist=self.base_twist if base_fresh else (0.0, 0.0, 0.0),
            left_controller_pose=self.left_controller_pose,
            right_controller_pose=self.right_controller_pose,
            left_grip=self.left_grip,
            right_grip=self.right_grip,
            left_trigger=self.left_trigger,
            right_trigger=self.right_trigger,
            left_joystick=self.left_joystick,
            right_joystick=self.right_joystick,
            button_states=dict(self.button_states),
            left_arm_position=self.left_arm_position,
            right_arm_position=self.right_arm_position,
            lift_position=self.lift_position,
            lift_velocity=(self.lift_velocity if lift_fresh else ((0.0,) if self.lift_velocity is not None else None)),
            lift_action_position=lift_action_position,
            head_position=self.head_position,
            left_gripper_position=self.left_gripper_position,
            right_gripper_position=self.right_gripper_position,
        )
