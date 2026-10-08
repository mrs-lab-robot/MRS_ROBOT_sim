# MRS Robot simulation runtime

`sim_runtime` is the simulator-facing foundation: canonical USD/stages, sensor and runtime configuration, scenarios,
ROS 2 packages, and ROS-independent teleoperation protocol/transport. It does not import Isaac Lab-Arena or define
benchmark tasks.

## Canonical resources

- Robot USD: `assets/robots/openflex_robot.usda`
- Sensor and mount configuration: `config/sensors/`
- ROS 2 workspace packages: `ros2/openflex_isaac_sim/`
- VR command protocol and UDP/safety primitives: `teleoperation/src/mrs_teleoperation/`
- Arena ROS relay package (transport adapter only): `ros2/mrs_robot_arena_bridge/`

RealSense configs in the ROS package are symlinks to `config/sensors/realsense/`, so ROS packaging uses the same
canonical data as Isaac Lab. `../isaac_sim_core` and `../ros2_pkgs/openflex_isaac_sim` are compatibility symlinks for
existing launchers and GUIs; new code should use these canonical paths.

## Build and test

Build the six Isaac Sim ROS packages from the repository root after sourcing ROS Humble and the OpenFleX underlay:

```bash
colcon build --symlink-install --base-paths sim_runtime/ros2/openflex_isaac_sim
```

The optional VR relay is built separately from `arena_benchmark/integrations/ros2/mrs_robot_arena_bridge` so its
generated colcon outputs do not mix with the main runtime workspace. The simulator-independent protocol and safety
tests live at `../tests/sim_runtime/teleoperation`; they use the project teleoperation source package directly.
