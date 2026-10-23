"""采集链路冒烟：TeleopEpisodeRecorder -> HDF5（不启动 Kit）。

验证点：
  1. start / append / save 全流程能写出 HDF5；
  2. episode 里该有的 dataset 都在、形状正确；
  3. 带完整溯源字段时 metadata 会被标成可训练 schema。

用法（在 MRS_ROBOT_sim 根目录）：
    MRS_ROBOT_SIM_ROOT=$PWD arena_benchmark/.venv/bin/python tools/smoke_capture_recorder.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:  # 生成 4 张假相机帧，走通 camera_frames 分支
    from PIL import Image
    import io

    def _dummy_png(index: int) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), (index * 40 % 255, 80, 160)).save(buffer, format="PNG")
        return buffer.getvalue()

    CAMERA_FRAMES = {name: _dummy_png(i) for i, name in enumerate(
        ("base_d435", "head_d435", "left_wrist_d405", "right_wrist_d405")
    )}
except ImportError:  # 没装 Pillow 就只验数值链路
    CAMERA_FRAMES = {}

from mrs_robot_lab.assets.robot_interface import ACTION_DIMENSION, JOINT_STATE_NAMES
from mrs_robot_lab.recorders.teleop_episode import EpisodeState, TeleopEpisodeRecorder

OUT = Path.home() / ".hermes/cache/scratch/capture_smoke/episode_000000.hdf5"


def main() -> int:
    recorder = TeleopEpisodeRecorder()
    recorder.start()
    joint_dim = len(JOINT_STATE_NAMES)
    rng = np.random.default_rng(0)
    for step in range(12):
        joints = rng.normal(0.0, 0.05, joint_dim)
        recorder.append(
            sim_time_ns=1_000_000_000 + step * 33_333_333,
            step_index=step,
            joint_position=joints.tolist(),
            joint_velocity=np.zeros(joint_dim).tolist(),
            next_joint_position=joints.tolist(),
            command_seq=step,
            action=rng.normal(0.0, 0.1, ACTION_DIMENSION).tolist(),
            raw_command={"left_grip": 0.2, "right_grip": 0.0},
            camera_frames=dict(CAMERA_FRAMES) if CAMERA_FRAMES else None,
        )
    print("episode 状态:", recorder.state, "步数:", recorder.num_steps, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    metadata = {
        "episode_id": "smoke-0000",
        "config_hash": "0" * 64,
        "contract_hash": "1" * 64,
        "random_seed": 0,
        "input_source": "synthetic_vr",
        "joint_state_order": list(JOINT_STATE_NAMES),
        "data_mode": "ground_truth",
        "task_id": "smoke",
    }
    path = recorder.save(OUT, metadata=metadata)
    print("已写入:", path, path.stat().st_size, "bytes", flush=True)
    assert recorder.state is not EpisodeState.RECORDING

    import h5py

    with h5py.File(path, "r") as dataset:
        keys = sorted(dataset.keys())
        print("datasets:", keys, flush=True)
        for key in keys:
            if hasattr(dataset[key], "shape"):
                print(f"  {key:24s} {dataset[key].shape}", flush=True)
        meta = json.loads(dataset.attrs["metadata_json"])
        print("format:", dataset.attrs["format"], "| schema:", meta.get("episode_schema_version"), flush=True)
        steps = dataset["sim_time_ns"].shape[0]
        print("步数:", steps, flush=True)
        assert steps == 12
        assert dataset["action"].shape == (12, ACTION_DIMENSION), dataset["action"].shape
        assert "joint_position" in keys
        print("✓ 采集 HDF5 结构正确", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
