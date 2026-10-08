# Integration tests

Place Arena/Isaac Sim tests here. Run them with the pinned runtime from either a native Python 3.12/uv installation
or the matching Docker image; Docker is optional. `scripts/smoke_openflex.py` is the first no-task load/reset/step
integration check. `scripts/smoke_openflex_task.py` then composes the registered one-step task with the same scene
and embodiment, and checks successful termination plus the recorded `success_rate` metric.
`scripts/smoke_openflex_cameras.py --enable_cameras` additionally renders all four configured RGB/depth cameras and
checks the tensors returned in Arena's `camera_obs` observation group; it does not test ROS image publishing or HDF5
image recording.
