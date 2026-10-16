# Runtime Scripts

Scripts in this directory are development and integration entrypoints. They are not a stable command-line API.

- The native runtime wrapper prepares the pinned Isaac Sim/Kit Python environment and then invokes a selected script.
- Smoke and teleoperation scripts require the matching simulator runtime; many also require a compatible NVIDIA GPU/driver.
- Evaluation scripts are experimental and do not, by themselves, define a validated benchmark.
- Use each script's `--help` output and the matching revision's source as the current command reference.

Do not commit generated logs, datasets, checkpoints, or local configuration. For the unit-test command and dependency setup, see the [Arena guide](../README_CN.md).