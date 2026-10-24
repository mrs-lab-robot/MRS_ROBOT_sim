# Isaac Lab Extension

This layer adapts the simulation resources for Isaac Lab. It provides articulation, action and observation configuration, camera adapters, recorders, and simulation tools. The implementation and its interfaces are still under active development; they are not a stable API.

## Install

Use the Python/Isaac versions recorded in [`../configs/versions.yaml`](../configs/versions.yaml). Install the extension and the simulator-independent transport package into the same Isaac Lab environment:

```bash
python -m pip install -e ../sim_runtime/teleoperation
python -m pip install -e .
```

The full desktop workflow additionally requires the separately deployed companion GUI/robot workspace. This repository does not include that GUI or real-robot safety controls.

## Checks

Simulator-independent tests (run from the repository root; the suite lives in the single
`tests/` tree — see [`../tests/README.md`](../tests/README.md)):

```bash
pytest tests/unit/lab -q
```

Scripts that initialize Kit or create a simulation require the pinned Isaac runtime and a suitable GPU/driver. Their execution is not covered by the unit test command above. Episode persistence also requires the package dependencies declared in this project.