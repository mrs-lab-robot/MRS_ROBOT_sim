# MRS Robot Isaac Lab extension

`isaaclab_ext` adapts the canonical robot in `../sim_runtime` for Isaac Lab. It owns `mrs_robot_lab`, including the
asset resolver, contract-derived `ArticulationCfg`, action and observation groups, camera wrappers, teleoperation
adapters, recorders, and a no-task smoke environment. It does not import or require IsaacLab-Arena.

The robot USD and robot interface contract each have one canonical copy. `AssetResolver` reads them from
`sim_runtime/assets/robots/openflex_robot.usda` and
`sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract/config/embodiment.yaml`.
Camera mount and image parameters are read from `sim_runtime/config/sensors/realsense/`.

## Runtime setup

Use the Python 3.12 / Isaac Sim / Isaac Lab versions in [`../configs/versions.yaml`](../configs/versions.yaml).
Install this package and the simulator-independent teleoperation package into the same Isaac Lab environment:

```bash
python -m pip install -e ../sim_runtime/teleoperation
python -m pip install -e .
export MRS_ROBOT_SIM_ROOT="$(realpath ..)"
```

After Isaac Lab is installed and the Kit app is initialized, run the independent spawn, joint, camera, or action
smoke checks:

```bash
python scripts/spawn_robot.py --device cuda:0
python scripts/test_joint_control.py --device cuda:0
python scripts/test_camera.py --device cuda:0
python scripts/test_actions.py --device cuda:0
python scripts/teleop_robot.py --device cuda:0 --viz kit
```

Train the contract-backed flat-ground navigation task with PPO (the command
creates a versioned run directory, a resumable RSL-RL checkpoint, and a
`run_manifest.json` containing the task/scene/embodiment hashes):

```bash
python scripts/train_navigation_ppo.py --device cuda:0 --num-envs 64 --iterations 100
```

To resume a checkpoint produced by this command, pass
`--resume-from /path/to/run/checkpoints/model_final.pt`. Resume is rejected if
the task, scene, or embodiment contract hash differs. This entrypoint has unit
coverage for the pinned RSL-RL runner/checkpoint API; a successful `--help` or
unit test does not replace a real Kit/GPU training run or Arena evaluation.

To drive the GUI simulation from the existing ROS 2/RViz command topics, start
`ros2 launch mrs_robot_arena_bridge ros_control.launch.py` in a ROS Humble terminal,
then run `python scripts/teleop_ros_robot.py --device cuda:0 --viz kit` in this
Isaac Lab environment. ROS stays in its Python 3.10 process; the Kit process uses
the localhost UDP bridge and does not import `rclpy`.

`test_camera.py` checks RGB and `distance_to_image_plane` output from each configured camera. `test_actions.py`
checks the contract-derived 22D action grouping and applies a bounded articulation target probe. These scripts require
the pinned Isaac runtime; they cannot be validated by the simulator-independent unit suite.

Run adapter and config tests without Isaac Sim:

```bash
PYTHONPATH=src:../sim_runtime/teleoperation/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest tests -q
```

HDF5 episode persistence requires the package dependencies `h5py` and `numpy`; they are declared in this package.
