# Arena Extension

Arena-native robot embodiments, tasks, datasets, policies, and evaluation suites for the MRS robot platform.

Chinese documentation: [README_CN.md](README_CN.md).

This directory is the **Arena benchmark layer inside `MRS_ROBOT_sim`**. Project code lives in
`src/mrs_arena`; the compatibility package `mrs_robot_arena` only re-exports old imports. The upstream source is
the root submodule `../third_party/IsaacLab-Arena` and is kept unmodified. Robot USD, ROS/VR transport, and
Isaac Lab articulation/action configuration are owned by sibling layers `../sim_runtime` and `../isaaclab_ext`.

## Repository boundaries

| Repository | Owns |
| --- | --- |
| `MRS_ROBOT` | GUI, real-robot ROS applications, deployment, and business integration |
| `MRS_ROBOT_sim` | Isaac Sim + ROS 2 runtime, Arena extension, published OpenFleX USD, sensors, and robot contracts |

The Arena extension reads the canonical robot USD and interface contract from its parent `MRS_ROBOT_sim` checkout.
It does not copy or edit those source files. `scripts/run_native_isaac.sh` resolves the parent automatically; set
`MRS_ROBOT_SIM_ROOT` explicitly only when running in a container or a nonstandard layout.

## Runtime compatibility

The target baseline is deliberately pinned as one compatible stack; runtime acceptance still requires a matching-host run:

| Component | Pin |
| --- | --- |
| Isaac Sim | `6.0.0.1` (6.0 line) |
| Isaac Lab | `3.0.0b2` (3.0 line; exact source commit is pinned below) |
| IsaacLab-Arena | `release/0.3.0` at `8737b4ceb25f99f81a81786b7fde73139b52f324` |
| Isaac Lab source submodule | `af1bab4dc173ba69b08fab779c14ead61d13fd33` |
| Python | `3.12` |

The machine-readable lock is `../configs/versions.yaml` (`configs/runtime/versions.yaml` is a compatibility link).
Keep the Arena commit, its Isaac Lab submodule commit,
and the simulator version in sync; do not update only one of them. Docker is optional: Arena 0.3.0 supports native
`uv` installation as well as Docker. The workstation checkout of Arena and `/home/y/IsaacLab` are separate checkouts
and must not be mixed with it.

## Checkout and setup

From the `MRS_ROBOT_sim` root, initialize the upstream Arena submodule and enter this directory:

```bash
git submodule update --init third_party/IsaacLab-Arena
cd arena_benchmark
```

If the repository was cloned without submodules:

```bash
git submodule update --init third_party/IsaacLab-Arena
```

For native Linux setup, use the pinned upstream lock file and Python 3.12. Keep the environment and cache on the
repository filesystem (adjust if it is on a small system disk):

```bash
uv python install 3.12
UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
  uv sync --project ../third_party/IsaacLab-Arena --no-default-groups --group isaaclab-from-wheel
uv pip install --python .venv/bin/python --no-deps --editable ../third_party/IsaacLab-Arena
uv pip install --python .venv/bin/python --editable ../sim_runtime/teleoperation
uv pip install --python .venv/bin/python --editable ../isaaclab_ext
uv pip install --python .venv/bin/python -e .
export MRS_ROBOT_SIM_ROOT="$(realpath ..)"
```

This selects the pinned Isaac Lab 3.0.0b2 wheel dependency group. The upstream source-based group is also available
when developing against the pinned Isaac Lab submodule, with additional checkout and install cost. For Docker, mount
the unified `MRS_ROBOT_sim` root and set `MRS_ROBOT_SIM_ROOT` to its container path; never put a host-only asset path in the repo.

Run fast checks and the no-task smoke environment:

```bash
PYTHONPATH=src:../isaaclab_ext/src:../sim_runtime/teleoperation/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit -q
bash scripts/run_native_isaac.sh scripts/smoke_openflex.py --viz none --device cuda:0
```

Run the task/metric acceptance smoke (records one successful episode in a temporary directory):

```bash
bash scripts/run_native_isaac.sh scripts/smoke_openflex_task.py --viz none --device cuda:0
```

`OPENFLEX_ARENA_SMOKE_OK` means Arena loaded the robot USD, reset, processed a low-speed base/arm action probe, and
completed zero-action steps with finite observations. It is not a task reward or benchmark result.

Run the four-camera RGB/depth observation acceptance smoke (one environment, one step):

```bash
bash scripts/run_native_isaac.sh scripts/smoke_openflex_cameras.py --enable_cameras --viz none --device cuda:0 --steps 1
```

`OPENFLEX_ARENA_CAMERA_SMOKE_OK` confirms the Arena observation manager returned RGB and
`distance_to_image_plane` tensors for all four configured cameras. Cameras remain disabled by default. This does not
publish ROS image topics or add images to VR HDF5 episodes.

Run the first physical tabletop task smoke without VR or a headset:

```bash
bash scripts/run_native_isaac.sh scripts/smoke_openflex_pick_cube.py --viz none --device cuda:0
```

`OPENFLEX_PICK_CUBE_MDP_SMOKE_OK` confirms the cube starts in contact with the physical table, idle robot actions do
not produce a false success, and the lift-success predicate/metric work when the cube state is injected at the goal.
The final state injection tests the predicate only; this smoke does not test robot reach, grasp, or lift control.

For desktop keyboard teleoperation:

```bash
bash scripts/run_native_isaac.sh scripts/teleop_openflex.py --viz kit --device cuda:0
```

The native launcher filters ROS Humble's Python 3.10 and shared-library paths from the Isaac Sim Python 3.12
process while preserving other paths such as CUDA. Use the wrapper whenever the shell has sourced ROS Humble; the
Arena control loop itself does not require ROS.

Keys: `I/K` drive forward/back, `J/L` strafe, `U/O` rotate; `Tab` selects the arm, `1`-`7` selects its joint,
hold `W/S` to jog; `R/F` lift, `A/D` head yaw, `Q/E` head pitch, `Z/X` open/close the two grippers, and `Esc`
exits. Base commands go directly to the simulated swerve joints using the canonical controller geometry and speed/
acceleration limits; no ROS messages are published.

## Current implementation status

- The package root and contract validator are simulator-independent at import time.
- Arena imports `mrs_robot_lab` to reuse the canonical USD-backed articulation, action groups, and observation config.
- OpenFlexEmbodiment is the first Arena-native mobile-manipulator embodiment. Its stable 22D action is ordered as
  base Twist (3), left arm (7), right arm (7), lift (1), head (2), and grippers (1+1); observations include joint
  position/velocity.
- Optional Arena camera observations reuse the canonical runtime mount contract for four cameras and expose RGB plus
  `distance_to_image_plane` under `camera_obs`; camera rendering is opt-in with `--enable_cameras`.
- The custom base ActionTerm maps Twist to swerve steering and wheel velocity targets from the parent simulation repo's
  controller YAML, with contract clipping and wheel acceleration limiting. It is simulator-native and does not
  publish ROS. A registered procedural table/cube lift task now checks a 25 cm lift goal; physical robot grasping,
  industrial task suites, training configuration, and a full benchmark protocol remain out of scope. The one-step
  task remains only an integration smoke. VR HDF5 capture currently records control/proprioception, not images.
- ROS package names, topics, joint names, and the existing GUI launch contract remain unchanged by this repository.

## Desktop GUI workflow

The simulation workspace is organized into nine pages. The Chinese page names are the GUI labels:

| Page | Purpose and current support |
| --- | --- |
| 仿真配置 | Configure Isaac Sim/Arena paths and Python runtime |
| 传感配置 | Existing classic Isaac Sim sensor controls; Arena RGB/depth is controlled by the camera-stream option on the data collection page |
| 资产管理 | Inspect canonical robot USD, contract, sensor, and base-controller resources |
| 场景任务 | Browse registered environment configs and task code, including the starter table/cube lift MDP; graphical task editor is not implemented |
| 数据采集 | Select classic Isaac Sim or IsaacLab-Arena; both reuse the same LeRobot collection form and recorder |
| 数据管理 | Existing LeRobot data tools; Arena HDF5 conversion and augmentation are not connected |
| 模型训练 | Existing single-job LeRobot offline training; Arena simulation training, RL, and parallel scheduling are not connected |
| 模型评测 | Existing model inference; Arena benchmarks/parallel evaluation are not connected |
| 任务中心 | Monitor simulator, collection, training, and evaluation processes and logs |

On `数据采集`, select classic Isaac Sim or IsaacLab-Arena. For Arena, choose an upstream scene, then either leave the task selector on “load selected scene only (NoTask)” or choose a compatible task. NoTask loads the selected scene and robot without an operation task; a selected task loads the complete scene/task graph from that task YAML. Enable the camera stream if RGB is wanted, then start Kit and the VR ROS bridge. The shared form updates its task label from the selection and invokes the same LeRobot recorder, schema, and dataset format used for real-robot collection. Arena preflight checks Arena's ROS streams and does not require the classic sensor-state API or pass classic scene-reset settings; episode resets remain in Kit/operator control. The Kit `OpenFlex VR Teleoperation` HDF5 panel remains a separate debugging recorder and does not save images. The two backends are mutually exclusive and neither starts robot hardware bringup. Arena augmentation and simulation training are not connected yet; current LeRobot management/offline-training workflows remain available. Inspect process status and logs in `任务中心`.

## Development and tests

Fast unit tests do not launch Isaac Sim:

```bash
PYTHONPATH=src:../isaaclab_ext/src:../sim_runtime/teleoperation/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/unit -q
```

Simulation integration tests must run in the pinned Arena runtime (native or container). Add tests for articulation load/reset, action
mapping, gripper mimic behavior, sensor frames, task success/timeout, and multi-environment isolation before treating
an embodiment or task as benchmark-ready.

Generated outputs, datasets, checkpoints, and local runtime configuration do not belong in Git. Store small,
reviewed baseline summaries with their commit, asset version, seeds, and metric protocol under `benchmarks/`.
