# MRS_ROBOT_arena

Arena-native robot embodiments, tasks, datasets, policies, and evaluation suites for the MRS robot platform.

This repository is an **external Isaac Lab-Arena extension**. The upstream Arena source is kept unmodified under
`third_party/isaaclab-arena`; project-specific code lives in `src/mrs_robot_arena`. This repository does not contain
the ROS 2 simulator, duplicate robot USD files, or own the real-robot safety/control stack.

## Repository boundaries

| Repository | Owns |
| --- | --- |
| `MRS_ROBOT` | GUI, real-robot ROS applications, deployment, and business integration |
| `MRS_ROBOT_sim` | Isaac Sim + ROS 2 runtime, published OpenFleX USD, sensor configuration, and robot interface contracts |
| `MRS_ROBOT_arena` | Arena embodiments, scenes, tasks, policies, learning configs, and reproducible benchmarks |

The Arena project reads the canonical robot USD and interface contract from a sibling `MRS_ROBOT_sim` checkout. It
does not copy or edit those source files. Set `MRS_ROBOT_SIM_ROOT` to that checkout explicitly; machine-specific
absolute paths are intentionally not guessed.

## Runtime compatibility

The supported baseline is deliberately pinned as one tested stack:

| Component | Pin |
| --- | --- |
| Isaac Sim | `6.0.0.1` (6.0 line) |
| Isaac Lab | `3.0.0b2` (3.0 line; exact source commit is pinned below) |
| IsaacLab-Arena | `release/0.3.0` at `8737b4ceb25f99f81a81786b7fde73139b52f324` |
| Isaac Lab source submodule | `af1bab4dc173ba69b08fab779c14ead61d13fd33` |
| Python | `3.12` |

The machine-readable lock is `configs/runtime/versions.yaml`. Keep the Arena commit, its Isaac Lab submodule commit,
and the simulator version in sync; do not update only one of them. Use the Arena Docker/runtime for this exact stack.
The workstation checkout of Arena and `/home/y/IsaacLab` are separate checkouts and must not be mixed with it.

## Checkout and setup

Clone this repository, including upstream sources:

```bash
git clone --recurse-submodules <MRS_ROBOT_arena-url> MRS_ROBOT_arena
cd MRS_ROBOT_arena
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Run Isaac Lab-Arena commands in the Docker/runtime prescribed by the pinned upstream checkout. Install Arena first,
then this extension into that same Isaac Sim Python environment:

```bash
/isaac-sim/python.sh -m pip install -e third_party/isaaclab-arena
/isaac-sim/python.sh -m pip install -e '.[dev]'
export MRS_ROBOT_SIM_ROOT=/workspaces/MRS_ROBOT_sim
```

Bind-mount `MRS_ROBOT_sim` into the container at the path used by `MRS_ROBOT_SIM_ROOT`. Never use a host-only
absolute asset path from inside the container.

## Current implementation status

- The Python package and tests are intentionally simulator-independent at import time.
- `AssetResolver` reads the existing USD and contract by stable resource name.
- Arena `EmbodimentBase`, task, environment, and policy implementations are added incrementally after the pinned
  runtime is available; imports that require Isaac Sim must happen after Kit/SimulationApp initialization.
- ROS package names, topics, joint names, and the existing GUI launch contract remain unchanged by this repository.

## Development and tests

Fast unit tests do not launch Isaac Sim:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/unit -v
```

Simulation integration tests must run in the pinned Arena container. Add tests for articulation load/reset, action
mapping, gripper mimic behavior, sensor frames, task success/timeout, and multi-environment isolation before treating
an embodiment or task as benchmark-ready.

Generated outputs, datasets, checkpoints, and local runtime configuration do not belong in Git. Store small,
reviewed baseline summaries with their commit, asset version, seeds, and metric protocol under `benchmarks/`.
