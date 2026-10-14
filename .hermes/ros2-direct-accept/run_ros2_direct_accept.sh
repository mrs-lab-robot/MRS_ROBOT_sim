#!/usr/bin/env bash
# ros2_direct 修复项 C 一键补测脚本（A6000 上执行）
# 前提：nvidia-smi 显示 GPU 空闲（修复项 A 的仿真已停）。
# 用法: bash run_ros2_direct_accept.sh          # 全量: kit+rosbridge+driver+record
#       VALIDATE_ONLY=1 bash run_ros2_direct_accept.sh   # 只拉 kit+rosbridge+driver,自检后退出
set -uo pipefail

HOST_ROOT="$HOME/mrs_robot/MRS_ROBOT"
ARENA_ROOT="$HOST_ROOT/MRS_ROBOT_sim/arena_benchmark"
VLA_ROOT="$HOST_ROOT/openflex_ws"
SIM_ROOT="$HOST_ROOT/MRS_ROBOT_sim"
ACCEPT_DIR="$SIM_ROOT/.hermes/ros2-direct-accept"
RUN_DIR=/tmp/mrs-accept
DATA_ROOT="$VLA_ROOT/datasets/lerobot/ros2_accept_demo"
ROS_DOMAIN_ID=49

mkdir -p "$RUN_DIR/arena_datasets"
cd "$RUN_DIR"

check_gpu_free() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used > 1000 )); then
        echo "GPU 占用 ${used} MiB —— 请确认修复项 A 的仿真已停止后再跑" >&2
        exit 3
    fi
}

preflight_cleanup() {
    echo ">>> [P0] 清理上次运行可能遗留的孤儿进程 (kit/relay/vr节点/driver)"
    pkill -f teleop_vr_openflex.py 2>/dev/null
    pkill -f "mrs_robot_arena_bridge/lib/mrs_robot_arena_bridge/arena_relay" 2>/dev/null
    pkill -f openarmx_teleop_vr_node 2>/dev/null
    pkill -f openarmx_head_teleop_vr_pico 2>/dev/null
    pkill -f vr_lift_control 2>/dev/null
    pkill -f "ros2 launch mrs_robot_arena_bridge" 2>/dev/null
    pkill -f leader_driver.py 2>/dev/null
    sleep 8
    pgrep -af "teleop_vr_openflex|arena_relay|leader_driver|openarmx_teleop_vr_node" | grep -v pgrep \
        && echo ">>> [P0][WARN] 仍有进程残留" >&2 || echo ">>> [P0] 无残留。"
}

start_kit() {
    echo ">>> [P1] 启动 Arena kit 仿真 (ros2_direct)"
    (
        cd "$ARENA_ROOT"
        MRS_ARENA_PYTHON="$ARENA_ROOT/.venv/bin/python" \
        bash scripts/run_native_isaac.sh scripts/teleop_vr_openflex.py \
            --headless --device cuda:0 \
            --record-dir "$RUN_DIR/arena_datasets" \
            --render-hz 30 --physics-hz 120 \
            --command-host 127.0.0.1 --state-host 127.0.0.1 \
            --camera-stream-host 127.0.0.1 \
            --control-host 127.0.0.1 --control-port 24105 \
            --record-image-hz 15.0 --episodes 1 \
            --enable_cameras --camera-stream-port 24104 \
            --data-mode ros2_direct \
            --seed 7 \
            > "$RUN_DIR/kit.log" 2>&1
    ) &
    echo $! > "$RUN_DIR/kit.pid"
}

start_ros() {
    echo ">>> [P2] 启动 Arena ROS bridge (colcon build + arena_vr_teleop.launch.py)"
    nohup bash -c "
        source /opt/ros/humble/setup.bash
        source '$VLA_ROOT/install/setup.bash'
        cd '$SIM_ROOT'
        colcon build --base-paths sim_runtime/ros2/mrs_robot_arena_bridge \
            --build-base sim_runtime/ros2/build --install-base sim_runtime/ros2/install
        source sim_runtime/ros2/install/setup.bash
        export PYTHONPATH='$SIM_ROOT/sim_runtime/teleoperation/src':\"\$PYTHONPATH\"
        export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=1
        exec ros2 launch mrs_robot_arena_bridge arena_vr_teleop.launch.py
    " > "$RUN_DIR/ros.log" 2>&1 &
    echo $! > "$RUN_DIR/ros.pid"
}

start_driver() {
    echo ">>> [P3] 启动模拟 VR 命令流 (leader_driver)"
    nohup bash -c "
        source /opt/ros/humble/setup.bash
        source '$SIM_ROOT/sim_runtime/ros2/install/setup.bash'
        export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=1
        exec python3 '$ACCEPT_DIR/leader_driver.py'
    " > "$RUN_DIR/driver.log" 2>&1 &
    echo $! > "$RUN_DIR/driver.pid"
}

wait_ready() {
    echo ">>> 等待链路就绪 (最长 180s)"
    local waited=0
    while (( waited < 180 )); do
        if grep -q "ARENA_CAPTURE_API_READY" "$RUN_DIR/kit.log" 2>/dev/null \
           && grep -q "Arena relay ready" "$RUN_DIR/ros.log" 2>/dev/null \
           && grep -q "leader_driver ready" "$RUN_DIR/driver.log" 2>/dev/null; then
            echo ">>> 链路就绪 (Kit API + relay + driver)。"
            return 0
        fi
        if ! kill -0 "$(cat "$RUN_DIR/kit.pid")" 2>/dev/null; then
            echo ">>> Kit 仿真进程已退出 —— 查 $RUN_DIR/kit.log" >&2
            return 1
        fi
        sleep 5; waited=$((waited + 5))
    done
    echo ">>> 180s 内未全部就绪 —— 查 $RUN_DIR/{kit,ros,driver}.log" >&2
    return 1
}

validate_topics() {
    echo ">>> [自检] ROS topic 存活性"
    bash -c "
        source /opt/ros/humble/setup.bash
        source '$SIM_ROOT/sim_runtime/ros2/install/setup.bash'
        export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=1
        ros2 topic list
    " > "$RUN_DIR/topic_list.txt" 2>&1
    for t in /joint_states /left_forward_position_controller/commands \
             /right_forward_position_controller/commands \
             /head_forward_position_controller/commands \
             /lift_manual_position_controller/action_position \
             /cam_left/color/image/compressed /cam_head/color/image/compressed; do
        if grep -qx "$t" "$RUN_DIR/topic_list.txt"; then
            echo "    [OK] $t"
        else
            echo "    [MISS] $t" >&2
            return 1
        fi
    done
    echo ">>> 全部必需 topic 已就位。"
}

stop_all() {
    echo ">>> 停止全部进程"
    for pid_file in driver.pid ros.pid kit.pid; do
        [[ -f "$RUN_DIR/$pid_file" ]] && kill "$(cat "$RUN_DIR/$pid_file")" 2>/dev/null
    done
    sleep 3
    pkill -f teleop_vr_openflex.py 2>/dev/null
    pkill -f arena_vr_teleop.launch.py 2>/dev/null
    pkill -f leader_driver.py 2>/dev/null
    pkill -f "mrs_robot_arena_bridge/lib/mrs_robot_arena_bridge/arena_relay" 2>/dev/null
    pkill -f openarmx_teleop_vr_node 2>/dev/null
    pkill -f openarmx_head_teleop_vr_pico 2>/dev/null
    pkill -f vr_lift_control 2>/dev/null
    sleep 1
}

wait_joint_states_data() {
    echo ">>> [自检] /joint_states 数据流 (kit→relay state 链路, 3次×30s)"
    local attempt=1
    while (( attempt <= 3 )); do
        if timeout 30 bash -c "
            source /opt/ros/humble/setup.bash
            source '$SIM_ROOT/sim_runtime/ros2/install/setup.bash'
            export ROS_DOMAIN_ID=$ROS_DOMAIN_ID ROS_LOCALHOST_ONLY=1
            ros2 topic echo --once /joint_states --field name 2>/dev/null | grep -q lift_joint
        "; then
            echo ">>> /joint_states 有数据流 (第 $attempt 次探测)。"
            return 0
        fi
        echo ">>> [WARN] 第 $attempt/3 次探测 30s 内未收到 /joint_states —— kit 疑似未发 state。" >&2
        echo "    kit.log 尾部: $(tail -2 "$RUN_DIR/kit.log" 2>/dev/null | tr '\n' ' ')" >&2
        attempt=$((attempt + 1))
        sleep 3
    done
    echo ">>> /joint_states 90s+ 无数据流 —— kit→relay state 链路未通, 记录现场后中止。" >&2
    curl -s -m 3 http://127.0.0.1:24105/api/v1/status > "$RUN_DIR/kit_status_gate_fail.json" 2>&1
    return 1
}

if [[ "${VALIDATE_ONLY:-0}" == 1 ]]; then
    preflight_cleanup
    check_gpu_free
    start_kit; sleep 20; start_ros; sleep 30; start_driver
    if wait_ready && validate_topics && wait_joint_states_data; then
        echo "VALIDATE_ONLY 完成。进程保留，可继续 start_capture 步骤。"
    else
        stop_all
        exit 1
    fi
    exit 0
fi

preflight_cleanup
check_gpu_free
rm -rf "$DATA_ROOT"          # start_capture.sh 要求目标数据集目录不存在
start_kit; sleep 20; start_ros; sleep 30; start_driver
if ! wait_ready; then stop_all; exit 1; fi
validate_topics || { stop_all; exit 1; }
wait_joint_states_data || { stop_all; exit 1; }

echo ">>> [P4] LeRobot ROS2 直采记录 (GUI 数据采集页'开始采集'等价)"
if [[ ! -f "$ACCEPT_DIR/capture.yaml" ]]; then echo "缺 capture.yaml" >&2; stop_all; exit 1; fi
bash "$HOST_ROOT/scripts/start_capture.sh" --config "$ACCEPT_DIR/capture.yaml" \
    > "$RUN_DIR/record.log" 2>&1
record_rc=$?
stop_all
echo ">>> record 退出码: $record_rc —— 日志 $RUN_DIR/record.log"
echo ">>> episode 产物:"
find "$DATA_ROOT" -type f | sort
echo ">>> 完成。产物目录: $DATA_ROOT"
