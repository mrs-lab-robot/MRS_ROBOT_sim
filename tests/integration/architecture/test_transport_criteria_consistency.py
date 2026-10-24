"""交叉一致性守卫：lab 与 arena 必须跑同一份「双臂搬箱」成功判据。

判据只有一份实现，在 ``mrs_robot_lab``（训练环境用的那份）；arena 的
``mrs_arena.tasks.transport_logic`` 只是它的薄封装。这份测试把同一段合成的
接触/抬升/搬运/释放序列同时喂给两个入口，逐步断言 ``success`` 与 ``phase``
完全相等——任何一侧再长出一份自己的状态机，这里就会红。
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from mrs_arena.tasks.transport_logic import (
    create_transport_progress_state,
    update_transport_progress,
)
from mrs_robot_lab.environments.learning.task_logic_torch import BatchedBoxTransportEvaluator


REPO_ROOT = Path(__file__).resolve().parents[3]
NUM_ENVS = 3
DT = 0.1
INITIAL = (0.65, 0.0, 0.80)
TARGET = (0.90, 0.0, 0.80)
CRITERIA = {
    "position_tolerance_m": 0.05,
    "min_lift_height_m": 0.10,
    "min_carry_distance_m": 0.20,
    "max_linear_speed_mps": 0.02,
    "stable_seconds": 0.5,
}

# 每一行是一步：三个 env 的 (box 位置, 左手接触, 右手接触, 标量速度)。
# env0 = 正常路径；env1 = 搬运中掉一只手（历史仍然算数）、最后也成功；
# env2 = 只单侧抓过、中途快速移动、最后在目标处又把箱子挪走（成功会回退）。
SEQUENCE = (
    # 0 全部在原位，无接触
    (((0.65, 0.0, 0.80), (0.65, 0.0, 0.80), (0.65, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.0)),
    # 1 env0 双手接触；env1 单手；env2 无接触
    (((0.65, 0.0, 0.80), (0.65, 0.0, 0.80), (0.65, 0.0, 0.80)), (True, True, False), (True, False, False), (0.0, 0.0, 0.0)),
    # 2 抬起 15cm：env0/env1 双手，env2 仍单手
    (((0.65, 0.0, 0.95), (0.65, 0.0, 0.95), (0.65, 0.0, 0.95)), (True, True, True), (True, True, False), (0.0, 0.0, 0.0)),
    # 3 搬运中：env0 走 15cm（不够 20cm），env1 走 21cm
    (((0.80, 0.0, 0.95), (0.86, 0.0, 0.95), (0.65, 0.0, 0.95)), (True, True, True), (True, True, False), (0.0, 0.0, 0.0)),
    # 4 env0 走够 22cm；env2 这时才双手接触（历史是粘性的）
    (((0.87, 0.0, 0.95), (0.87, 0.0, 0.95), (0.65, 0.0, 0.95)), (True, True, True), (True, True, True), (0.0, 0.0, 0.0)),
    # 5 放到目标：env1 只松一只手（不算释放）
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.70, 0.0, 0.95)), (True, True, True), (True, False, True), (0.0, 0.0, 0.0)),
    # 6 env0 在目标处但速度 0.5（稳定计时不累加）；env1 双手都松；env2 走到目标
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (True, False, True), (True, False, True), (0.5, 0.0, 0.0)),
    # 7 env0 松手开始计时；env2 只松一只手
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (False, False, False), (False, False, True), (0.0, 0.0, 0.0)),
    # 8 env2 松手但速度 0.5 → 不累加
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.5)),
    # 9-12 稳定 0.5s（dt=0.1，5 步）：env1 在第 10 步到点，env0 在第 11 步
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.0)),
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.0)),
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.0)),
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.90, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.0)),
    # 13 env2 把箱子挪走 → 已放置的历史还在，但不再在目标处，成功回退
    (((0.90, 0.0, 0.80), (0.90, 0.0, 0.80), (0.78, 0.0, 0.80)), (False, False, False), (False, False, False), (0.0, 0.0, 0.0)),
)


def make_lab_evaluator() -> BatchedBoxTransportEvaluator:
    return BatchedBoxTransportEvaluator(
        initial_position=INITIAL,
        target_position=TARGET,
        num_envs=NUM_ENVS,
        device="cpu",
        **CRITERIA,
    )


def make_arena_state():
    return create_transport_progress_state(NUM_ENVS, device="cpu")


def step_lab(evaluator: BatchedBoxTransportEvaluator, row):
    positions, left, right, speeds = row
    phase, success = evaluator.update(
        torch.tensor(positions, dtype=torch.float32),
        torch.tensor([[speed, 0.0, 0.0] for speed in speeds], dtype=torch.float32),
        torch.tensor(left),
        torch.tensor(right),
        dt=DT,
    )
    return success, phase


def step_arena(state, row):
    positions, left, right, speeds = row
    return update_transport_progress(
        state,
        box_position=torch.tensor(positions, dtype=torch.float32),
        box_linear_speed=torch.tensor(speeds, dtype=torch.float32),
        initial_position=torch.tensor([INITIAL] * NUM_ENVS, dtype=torch.float32),
        target_position=TARGET,
        left_contact=torch.tensor(left),
        right_contact=torch.tensor(right),
        dt=DT,
        **CRITERIA,
    )


def run_both(sequence):
    """逐步跑两个入口，返回 (lab 轨迹, arena 轨迹)，并逐步断言完全相等。"""

    evaluator = make_lab_evaluator()
    state = make_arena_state()
    lab_trace, arena_trace = [], []
    for index, row in enumerate(sequence):
        lab_success, lab_phase = step_lab(evaluator, row)
        arena_success, arena_phase = step_arena(state, row)

        assert lab_success.dtype is torch.bool and arena_success.dtype is torch.bool
        assert lab_phase.dtype is torch.long and arena_phase.dtype is torch.long
        assert lab_success.tolist() == arena_success.tolist(), (
            f"step {index}: success 分叉 lab={lab_success.tolist()} arena={arena_success.tolist()}"
        )
        assert lab_phase.tolist() == arena_phase.tolist(), (
            f"step {index}: phase 分叉 lab={lab_phase.tolist()} arena={arena_phase.tolist()}"
        )
        lab_trace.append((lab_success.tolist(), lab_phase.tolist()))
        arena_trace.append((arena_success.tolist(), arena_phase.tolist()))
    return lab_trace, arena_trace


def test_lab_and_arena_agree_on_every_step_of_the_transport_sequence():
    lab_trace, _ = run_both(SEQUENCE)

    success = [step[0] for step in lab_trace]
    phases = [step[1] for step in lab_trace]

    # 序列本身得是有信息量的：三条 env 走出不同轨迹，成功与失败都出现过。
    assert phases[0] == [0, 0, 0]
    assert phases[2] == [2, 2, 0]  # env2 只单侧接触过，抬起来也不算
    assert phases[5] == [4, 4, 2]  # env1 只松一只手 → 停在 PLACED
    assert phases[6] == [4, 5, 4]
    assert success[10] == [False, True, False]  # env1 稳定满 0.5s
    assert success[11] == [True, True, False]
    assert phases[12] == [6, 6, 5]
    assert success[13] == [True, True, False]  # env2 挪走箱子，成功回退
    assert not any(row[2] for row in success)  # env2 一次都没成功
    # 注意要铺平所有 env：只取 row[1] 只是 env1 的轨迹（env1 单手接触，永远走不到 1 档）。
    assert {phase for row in phases for phase in row} == {0, 1, 2, 3, 4, 5, 6}  # 0..6 全档都走过


def test_lab_and_arena_stay_identical_after_a_reset():
    """reset 之后两边必须重新从 0 档开始，并且仍然逐步一致。"""

    evaluator = make_lab_evaluator()
    state = make_arena_state()
    for row in SEQUENCE[:7]:
        step_lab(evaluator, row)
        step_arena(state, row)

    evaluator.reset()
    for value in state.values():
        value.zero_()

    tail = SEQUENCE[7:]
    for index, row in enumerate(tail):
        lab_success, lab_phase = step_lab(evaluator, row)
        arena_success, arena_phase = step_arena(state, row)
        assert lab_phase.tolist() == arena_phase.tolist(), f"reset 后第 {index} 步 phase 分叉"
        assert lab_success.tolist() == arena_success.tolist(), f"reset 后第 {index} 步 success 分叉"
        if index == 0:
            # 历史清空后没人再抓过箱子：释放序列不该再被判成功。
            assert lab_phase.tolist() == [0, 0, 0]
            assert lab_success.tolist() == [False, False, False]


def test_arena_transport_logic_does_not_reimplement_the_criterion():
    """静态守卫：arena 侧只允许委托，不允许再写一份状态机。"""

    source = (REPO_ROOT / "arena_benchmark/src/mrs_arena/tasks/transport_logic.py").read_text(
        encoding="utf-8"
    )

    assert "from mrs_robot_lab.environments.learning.task_logic_torch import" in source
    assert "BatchedBoxTransportEvaluator" in source
    # 旧的分叉实现特征：自己维护 seen 标记、自己算 phase 阶梯。
    assert not re.search(r'_seen"\]\s*\|=', source), "arena 不得自己维护判据历史"
    assert not re.search(r"phase\s*=\s*torch\.where", source), "arena 不得自己算 phase 阶梯"
