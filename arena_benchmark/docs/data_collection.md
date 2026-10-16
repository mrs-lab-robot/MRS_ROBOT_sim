# Data Capture (Experimental)

Data capture interfaces and schemas are still under development. Do not treat the current format as a stable interchange contract.

The current teleoperation recorder stores per-step simulation time, command sequence, action, and joint-state samples in HDF5. Dimensions and field order are defined by the current machine-readable embodiment configuration and may change. Verify them against the checked-out revision before consuming recorded data.

Camera observations can be enabled in selected simulator paths, but this does not imply that camera frames are included in the HDF5 episode. Dataset export, augmentation, training, and real-robot workflow integration are not a complete end-to-end pipeline in this repository.

Generated datasets belong in local output directories and must not be committed. The full desktop workflow requires a separately deployed, compatible real-robot GUI repository, which is not included here.