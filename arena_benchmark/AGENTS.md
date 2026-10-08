# Repository guidance

- Keep `third_party/isaaclab-arena` unmodified. Implement integrations in `src/mrs_robot_arena` or `compat/`.
- Keep ROS 2 optional for Arena-native rollout. Do not publish commands over ROS from the per-step policy path.
- Resolve USD and contract resources through `AssetResolver`; do not copy the canonical robot USD into this repository.
- Preserve the distinction between robot assets, embodiments, scenes, tasks, policies, and benchmark protocols.
- Run simulator-independent tests with `PYTHONPATH=src python3 -m unittest discover -s tests/unit -v`.
- Run Isaac Sim/Arena integration tests only in the matching upstream Arena runtime/container.
- Import Kit/Isaac Lab-dependent modules only after the simulation application is initialized.
- Keep output data, checkpoints, and local absolute paths out of Git.
