# ROS 2 Simulation Bringup

> This package is still under development. Its launch parameters and interfaces are not a stable API. The desktop GUI and real-robot application are maintained in a separate companion repository.

## Prerequisites

- Linux with ROS 2 Humble and the dependencies supplied by a compatible companion workspace.
- Isaac Sim 6.0 target installation, with `ISAACSIM_PATH` pointing to the directory containing `python.sh`.
- Working NVIDIA/Vulkan/RTX access. A graphical session is required when launching with `headless:=false`.
- The simulation ROS 2 overlay built and sourced after the companion workspace.

## Build

Run from the simulation repository root:

```bash
export ISAACSIM_ROBOT_ROOT="$(pwd)"
source /opt/ros/humble/setup.bash
source /path/to/companion-gui-workspace/install/setup.bash
colcon --log-base log/simulation build \
  --build-base build/simulation \
  --install-base install \
  --symlink-install \
  --base-paths sim_runtime/ros2/*_isaac_sim
source install/setup.bash
```

## Launch

The launch package name is defined by the package manifest in this directory. After sourcing the overlay, select the installed bringup package by its suffix:

```bash
BRINGUP_PACKAGE="$(ros2 pkg list | awk '/_isaac_bringup$/ {print; exit}')"
test -n "$BRINGUP_PACKAGE" || { echo 'bringup package is not installed' >&2; exit 1; }
ros2 launch "$BRINGUP_PACKAGE" sim.launch.py \
  headless:=true \
  render_hz:=30 \
  physics_hz:=120 \
  sensor_profile:=none
```

The main launch starts the simulation core; RViz is a separate entrypoint. Camera, LiDAR, and IMU resources are runtime-managed and may require RTX rendering even in headless mode. Review the launch files and current configuration for available arguments before use.

## Validation boundary

Check controller and sensor state using the current ROS contract files. A launch process starting successfully does not prove that images contain valid pixels, a point cloud is non-empty, the timing target is met, or the companion GUI works end to end. Performance reports and machine-specific run logs are not included as release evidence.