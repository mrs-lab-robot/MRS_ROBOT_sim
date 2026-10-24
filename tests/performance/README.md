# Performance and Runtime Checks

The scripts in this directory are optional runtime measurement probes, not part of the default unit-test suite or a stable acceptance protocol.

They may require ROS 2, Isaac Sim/Kit, NVIDIA rendering access, sensor configuration, or a running simulation. Use only the configuration and target hardware appropriate to the experiment; review each script's arguments before running it.

Generated summaries and raw outputs are written under `reports/`, which is ignored by Git. No local performance result in that directory should be treated as a product guarantee. Record the simulator version, code revision, configuration, seed, measurement window, and hardware when comparing results.