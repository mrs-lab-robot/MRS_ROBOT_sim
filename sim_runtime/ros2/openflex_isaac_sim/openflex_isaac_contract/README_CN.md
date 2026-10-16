# ROS 2 Embodiment Contract

This package contains the machine-readable interface contract used by the simulation-side ROS and Isaac components. The file `config/embodiment.yaml` is the current source of truth for runtime metadata, actions, observations, sensor roles, and topic mappings.

The schema and values are still under development. They are not a frozen public API and may change between revisions. Downstream consumers should validate against the exact repository revision they use rather than copying these definitions.

## Verify

Build and source the ROS 2 overlay after ROS Humble and the compatible companion workspace:

```bash
source /opt/ros/humble/setup.bash
source /path/to/companion-gui-workspace/install/setup.bash
colcon build --symlink-install --base-paths sim_runtime/ros2/*_isaac_sim
source install/setup.bash
```

Find the installed contract package and run its verifier:

```bash
CONTRACT_PACKAGE="$(ros2 pkg list | awk '/_isaac_contract$/ {print; exit}')"
test -n "$CONTRACT_PACKAGE" || { echo 'contract package is not installed' >&2; exit 1; }
ros2 run "$CONTRACT_PACKAGE" verify_embodiment_contract.py
```

The verifier may compare the contract with controller and sensor configuration installed by the same build. A passing contract check does not validate a running simulator, GUI integration, physical robot behavior, timing, or safety.