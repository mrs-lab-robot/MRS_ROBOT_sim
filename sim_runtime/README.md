# Simulation Runtime

This layer contains simulation assets and configuration, ROS 2 packages, and transport utilities shared by the Isaac Sim-facing components. It does not contain the desktop GUI or real-robot application, and its interfaces are still under development.

## Build boundary

Build the ROS 2 packages after sourcing ROS 2 Humble and the compatible companion GUI/robot workspace underlay:

```bash
source /opt/ros/humble/setup.bash
source /path/to/companion-gui-workspace/install/setup.bash
export ISAACSIM_ROBOT_ROOT="$(pwd)"
colcon build --symlink-install --base-paths sim_runtime/ros2/*_isaac_sim
source install/setup.bash
```

The companion workspace and its dependencies are separate repositories. If a dependency is not available in the underlay, resolve and build it according to its own versioned instructions before building this overlay.

The ROS and Isaac Sim processes use different Python environments (targeting Python 3.10 and 3.12 respectively). Do not source ROS Python paths into the simulator process. Generated `build/`, `install/`, and `log/` outputs are local build artifacts, not source files.

For complete desktop and real-robot workflows, deploy a compatible revision of the companion GUI repository separately. This runtime documentation does not claim end-to-end acceptance.