# Integration Tests

Integration tests require the pinned Isaac Sim/Arena runtime and an appropriate GPU/driver. Run them only in a dedicated environment; they are not part of the simulator-independent unit-test command.

Current integration and smoke scripts exercise limited paths such as environment loading, reset/step behavior, observations, and selected tasks. Passing them does not establish a stable API, a complete benchmark, GUI integration, or real-robot readiness. Check the selected test's `--help` and source for its current invocation.