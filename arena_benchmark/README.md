# Arena Simulation and Evaluation Layer

> **Development status:** this component is under active development and is not a stable release. Interfaces, tasks, and runtime setup may change.
>
> The full desktop workflow requires a separately deployed, compatible real-robot GUI repository. That GUI is not included here, and a matching revision is not pinned by this repository.

This directory provides Arena-side environments, task implementations, policy adapters, and evaluation utilities. It consumes assets and interface configuration from sibling simulation components and the pinned upstream Arena submodule; it does not provide the desktop GUI or real-robot safety controls.

## Runtime target

See [`../configs/versions.yaml`](../configs/versions.yaml) for the machine-readable version lock. The current target is Linux x86_64, Python 3.12, Isaac Sim 6.0.0.1, Isaac Lab 3.0.0b2, and the pinned Arena revision. These are development targets, not a compatibility guarantee for a stable product. Other architectures, driver versions, and GPU minimums are not declared supported.

## Setup

From the repository root, initialize the upstream source:

```bash
git submodule update --init third_party/IsaacLab-Arena
```

Then, from this directory, install the pinned wheel-based environment and editable project packages:

```bash
uv python install 3.12
UV_CACHE_DIR="$PWD/.cache/uv" UV_PROJECT_ENVIRONMENT="$PWD/.venv" \
  uv sync --project ../third_party/IsaacLab-Arena --no-default-groups --group isaaclab-from-wheel
uv pip install --python .venv/bin/python --no-deps --editable ../third_party/IsaacLab-Arena
uv pip install --python .venv/bin/python --editable ../sim_runtime/teleoperation
uv pip install --python .venv/bin/python --editable ../isaaclab_ext
uv pip install --python .venv/bin/python --editable .
```

The first Kit launch may show an NVIDIA/Omniverse license prompt. Review it and decide personally whether to accept it; the repository does not accept it automatically.

## Checks and limitations

Run simulator-independent tests from this directory:

```bash
CONTRACT_DIR="$(printf '%s\n' ../sim_runtime/ros2/*_isaac_sim/*_contract)"
PYTHONPATH="src:$CONTRACT_DIR" .venv/bin/python -m unittest discover -s tests/unit -v
```

Kit/Isaac Sim integration checks require the pinned runtime and suitable GPU/driver. Unit tests do not validate rendering, physical contacts, ROS streams, GUI integration, or a complete benchmark protocol. Treat current tasks and evaluation scripts as experimental until their target-host acceptance has been recorded.

See [README_CN.md](README_CN.md) for the Chinese guide. The repository root README describes the overall deployment boundary.