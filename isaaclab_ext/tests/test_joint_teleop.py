import math

from mrs_robot_lab.runners.joint_teleop import (
    action_vector_for_key,
    clamp_joint_target,
    physical_jog_delta,
    toggle_binary_gripper,
    validate_joint_targets,
)


def test_left_arm_jog_maps_to_the_canonical_action_slice():
    action = action_vector_for_key("left_arm", 2, 1.0)

    assert action == [0.0] * 5 + [1.0] + [0.0] * 16


def test_right_arm_jog_maps_to_the_right_seven_joint_slice():
    action = action_vector_for_key("right_arm", 6, -1.0)

    assert action == [0.0] * 16 + [-1.0] + [0.0] * 5


def test_gripper_commands_are_binary_and_unknown_groups_are_rejected():
    assert action_vector_for_key("left_gripper", 0, 1.0) == [0.0] * 20 + [1.0] + [0.0]

    try:
        action_vector_for_key("base", 0, 1.0)
    except ValueError as error:
        assert "unsupported" in str(error)
    else:
        raise AssertionError("unsupported action group was accepted")


def test_held_key_uses_a_small_normalized_increment():
    action = action_vector_for_key("left_arm", 0, 0.02)

    assert action == [0.0] * 3 + [0.02] + [0.0] * 18


def test_binary_gripper_uses_positive_for_open_and_negative_for_close():
    close_command = toggle_binary_gripper(1.0)

    assert close_command == -1.0
    assert toggle_binary_gripper(close_command) == 1.0


def test_base_twist_maps_into_the_three_leading_action_slots():
    action = action_vector_for_key("base_twist", 1, 0.2)

    assert action == [0.0, 0.2] + [0.0] * 20


def test_base_twist_uses_si_units_and_joint_actions_remain_normalized():
    assert action_vector_for_key("base_twist", 2, 0.35)[2] == 0.35
    try:
        action_vector_for_key("left_arm", 0, 1.01)
    except ValueError as error:
        assert "normalized" in str(error)
    else:
        raise AssertionError("joint action accepted a non-normalized command")


def test_direct_joint_jog_clamps_to_contract_limits():
    assert clamp_joint_target(0.9, 0.2, (-1.0, 1.0)) == 1.0
    assert clamp_joint_target(-0.9, -0.2, (-1.0, 1.0)) == -1.0


def test_direct_joint_jog_rejects_non_finite_values_and_invalid_limits():
    for args in ((float("nan"), 0.1, (-1.0, 1.0)), (0.0, float("inf"), (-1.0, 1.0)), (0.0, 0.1, (1.0, 1.0))):
        try:
            clamp_joint_target(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid direct joint jog was accepted: {args}")


def test_joint_target_validation_rejects_unknown_non_finite_and_out_of_limit_commands():
    limits = {"arm_joint": (-1.0, 1.0)}
    assert validate_joint_targets({"arm_joint": 0.5}, limits) == {"arm_joint": 0.5}
    for targets in ({"other_joint": 0.0}, {"arm_joint": float("nan")}, {"arm_joint": 1.01}):
        try:
            validate_joint_targets(targets, limits)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid joint targets were accepted: {targets}")


def test_physical_jog_delta_preserves_per_second_speed_when_control_frequency_changes():
    assert math.isclose(physical_jog_delta(0.02, 0.25, control_dt=1.0 / 90.0), 0.005)
    assert math.isclose(physical_jog_delta(0.02, 0.25, control_dt=1.0 / 120.0), 0.00375)
