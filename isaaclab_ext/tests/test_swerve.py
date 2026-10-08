import math

from mrs_robot_lab.controllers.swerve import SwerveConfig, compute_swerve_targets, load_swerve_config


CONFIG = SwerveConfig(
    wheel_radius=0.075,
    max_wheel_speed=0.8,
    wheel_positions=((0.21, 0.2735), (0.21, -0.2735), (-0.21, 0.2735), (-0.21, -0.2735)),
    twist_limits=((-0.8, 0.8), (-0.8, 0.8), (-2.320037210795224, 2.320037210795224)),
)


def test_forward_and_lateral_twists_map_to_module_commands():
    forward = compute_swerve_targets((0.3, 0.0, 0.0), CONFIG)
    lateral = compute_swerve_targets((0.0, 0.3, 0.0), CONFIG)

    assert forward.steering_angles == (0.0, 0.0, 0.0, 0.0)
    assert forward.wheel_angular_velocities == (4.0, 4.0, 4.0, 4.0)
    assert all(math.isclose(angle, math.pi / 2, abs_tol=1e-12) for angle in lateral.steering_angles)
    assert lateral.wheel_angular_velocities == (4.0, 4.0, 4.0, 4.0)


def test_combined_twist_is_clipped_and_steering_stays_within_joint_limits():
    targets = compute_swerve_targets((0.8, 0.8, 2.0), CONFIG)

    assert max(abs(speed) for speed in targets.wheel_angular_velocities) <= 0.8 / 0.075 + 1e-12
    assert all(-math.pi / 2 <= angle <= math.pi / 2 for angle in targets.steering_angles)


def test_zero_twist_stops_all_wheels_and_straightens_modules():
    targets = compute_swerve_targets((0.0, 0.0, 0.0), CONFIG)

    assert targets.steering_angles == (0.0, 0.0, 0.0, 0.0)
    assert targets.wheel_angular_velocities == (0.0, 0.0, 0.0, 0.0)


def test_load_swerve_config_reads_geometry_and_limits(tmp_path):
    controller = tmp_path / "controller.yaml"
    controller.write_text(
        "swerve_drive_controller:\n  ros__parameters:\n"
        "    steering_joint_names: [fl_steering_joint, fr_steering_joint, bl_steering_joint, br_steering_joint]\n"
        "    wheel_joint_names: [fl_wheel_joint, fr_wheel_joint, bl_wheel_joint, br_wheel_joint]\n"
        "    wheel_radius: 0.075\n    max_wheel_speed: 0.8\n    wheel_accel_limit: 0.2\n"
        "    fl_pos_x: 0.21\n    fl_pos_y: 0.2735\n    fr_pos_x: 0.21\n    fr_pos_y: -0.2735\n"
        "    bl_pos_x: -0.21\n    bl_pos_y: 0.2735\n    br_pos_x: -0.21\n    br_pos_y: -0.2735\n"
        "    fl_steering_min: -1.5708\n    fl_steering_max: 1.5708\n"
        "    fr_steering_min: -1.5708\n    fr_steering_max: 1.5708\n"
        "    bl_steering_min: -1.5708\n    bl_steering_max: 1.5708\n"
        "    br_steering_min: -1.5708\n    br_steering_max: 1.5708\n",
        encoding="utf-8",
    )
    contract = tmp_path / "embodiment.yaml"
    contract.write_text(
        "actions:\n  - name: base_twist\n    limits:\n"
        "      - {minimum: -0.8, maximum: 0.8}\n      - {minimum: -0.8, maximum: 0.8}\n"
        "      - {minimum: -2.32, maximum: 2.32}\n",
        encoding="utf-8",
    )

    cfg = load_swerve_config(controller, contract)

    assert cfg.wheel_radius == 0.075
    assert cfg.max_wheel_speed == 0.8
    assert cfg.wheel_accel_limit == 0.2
    assert cfg.wheel_positions == CONFIG.wheel_positions
    assert cfg.steering_limits[0] == (-1.5708, 1.5708)
    assert cfg.twist_limits[2] == (-2.32, 2.32)
