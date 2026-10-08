# Scripts

Keep convenience and deployment scripts thin. The package implementation and reusable logic belong under `src/`.
# Runtime entry points

- `bash scripts/run_native_isaac.sh scripts/smoke_openflex.py --viz none --device cuda:0`: headless Arena embodiment smoke test.
- `bash scripts/run_native_isaac.sh scripts/smoke_openflex_cameras.py --enable_cameras --viz none --device cuda:0 --steps 1`: four-camera RGB/depth observation acceptance smoke.
- `bash scripts/run_native_isaac.sh scripts/smoke_openflex_task.py --viz none --device cuda:0`: task registration, success termination, and recorded success-rate smoke test.
- `bash scripts/run_native_isaac.sh scripts/smoke_openflex_dual_arm_box.py --viz none --device cuda:0`: build the dual-arm box task, inspect left/right filtered contact sensors, and verify that an idle robot is not reported successful. This does not verify physical grasp or a full success trajectory.
- `bash scripts/run_native_isaac.sh scripts/teleop_openflex.py --viz kit --device cuda:0`: Kit keyboard joint jog.
- `bash scripts/run_native_isaac.sh scripts/teleop_ros_openflex.py --viz kit --device cuda:0`: Arena Kit GUI controlled by the ROS 2 relay topics, without VR.
- `bash scripts/run_native_isaac.sh scripts/teleop_vr_openflex.py --viz kit --device cuda:0`: Arena Kit GUI + ROS VR relay + manual HDF5 episode capture. See `README_CN.md` for building and launching the ROS relay.
