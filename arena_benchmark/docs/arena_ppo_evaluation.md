# Arena Policy Evaluation (Experimental)

The navigation evaluation entrypoint is experimental. It is useful for development checks, but the task set, reset distribution, metric protocol, and reproducibility process are not frozen. Results must not be presented as a validated benchmark without a separately reviewed protocol and target-runtime acceptance.

## Current workflow

- The evaluator consumes a checkpoint produced by the compatible training path.
- Run it only in the Python/Isaac Sim/Arena versions recorded in [`../../configs/versions.yaml`](../../configs/versions.yaml).
- A result file is written under the selected run directory. Keep generated results, checkpoints, and logs outside Git.
- The desktop GUI is not included here. Full integrated workflows require a compatible revision of the companion GUI repository.

The command-line entrypoint is under `../scripts/`; check its `--help` output for the current arguments. Do not assume arguments or result schemas are stable across revisions.

## Validation boundary

A unit test can validate checkpoint metadata handling and result aggregation without launching Kit. It does not validate the task dynamics, contact behavior, policy success, simulator timing, or a complete benchmark. Record the code revision, simulator versions, seed range, task configuration, and metric definition for any experimental result.