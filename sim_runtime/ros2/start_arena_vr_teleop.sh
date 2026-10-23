#!/usr/bin/env bash
# 一键启动「真机同款」VR 遥操作链（pose_bridge + IK + 头控 + 底盘 + 升降 + arena relay）
#
#   bash sim_runtime/ros2/start_arena_vr_teleop.sh
#   bash sim_runtime/ros2/start_arena_vr_teleop.sh arena_host:=127.0.0.1
#
# 依赖：
#   - /opt/ros/humble
#   - 完整构建过的 openflex_ws（用 OPENFLEX_WS 覆盖路径）
#   - 本仓 sim_runtime/ros2 下的 mrs_robot_arena_bridge（已 colcon build）
#   - mrs_teleoperation 通过 PYTHONPATH 注入（relay_node 依赖它）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

# 登录 shell 里可能已经导出了指向别处的 OPENFLEX_WS（如 ~/OpenFlex_ws），
# 这里只认「确实装好的」那个：显式给的优先，否则按常见位置找。
OPENFLEX_WS="${OPENFLEX_WS:-}"
if [ -z "$OPENFLEX_WS" ] || [ ! -f "$OPENFLEX_WS/install/setup.bash" ]; then
    for candidate in \
        /media/y/workdisk/project/MRS_ROBOT/openflex_ws \
        "$REPO_ROOT/../MRS_ROBOT/openflex_ws"; do
        if [ -f "$candidate/install/setup.bash" ]; then
            OPENFLEX_WS="$candidate"
            break
        fi
    done
fi

# ROS 的 setup.bash 会读未定义变量，set -u 下必须先放开
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
if [ ! -f "$OPENFLEX_WS/install/setup.bash" ]; then
    echo "✗ 找不到 $OPENFLEX_WS/install/setup.bash（先 colcon build openflex_ws）" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "$OPENFLEX_WS/install/setup.bash"
if [ ! -f "$HERE/install/setup.bash" ]; then
    echo "✗ 找不到 $HERE/install/setup.bash（先 colcon build sim_runtime/ros2）" >&2
    exit 3
fi
# shellcheck disable=SC1090
source "$HERE/install/setup.bash"
set -u

export PYTHONPATH="$REPO_ROOT/sim_runtime/teleoperation/src:${PYTHONPATH:-}"

echo "▶ VR 遥操作链启动"
echo "  repo     : $REPO_ROOT"
echo "  underlay : $OPENFLEX_WS"
echo "  入口     : Pico APK → UDP 5100（pico_pose_bridge）"
echo "  出口     : arena relay → UDP 24102（仿真 teleop）"
echo

exec ros2 launch mrs_robot_arena_bridge arena_vr_teleop.launch.py "$@"
