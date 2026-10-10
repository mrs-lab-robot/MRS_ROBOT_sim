#!/usr/bin/env bash
# Launch an Arena script with the project Python environment, excluding ROS
# Humble's Python 3.10 and native libraries from the Isaac Sim Python 3.12 process.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${MRS_ARENA_PYTHON:-${repo_root}/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_native_isaac.sh <script.py> [args...]" >&2
    exit 2
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "Project Python not found: ${python_bin}" >&2
    echo "Create the project environment first (see README_CN.md)." >&2
    exit 2
fi

filter_ros_humble_paths() {
    local value="${1:-}" output="" component
    local -a components=()
    IFS=: read -r -a components <<< "${value}"
    for component in "${components[@]}"; do
        [[ -z "${component}" ]] && continue
        if [[ "${component}" == "/opt/ros/humble" || "${component}" == "/opt/ros/humble/"* ]]; then
            continue
        fi
        [[ -n "${output}" ]] && output+=:
        output+="${component}"
    done
    printf '%s' "${output}"
}

filtered_pythonpath="$(filter_ros_humble_paths "${PYTHONPATH:-}")"
export LD_LIBRARY_PATH="$(filter_ros_humble_paths "${LD_LIBRARY_PATH:-}")"
kit_ros2_core_dir="${repo_root}/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core"
kit_ros2_humble_lib="${kit_ros2_core_dir}/humble/lib"
ros2_distro="${ROS_DISTRO:-humble}"
if [[ "${ros2_distro}" == "humble" && -d "${kit_ros2_humble_lib}" ]]; then
    export ROS_DISTRO="${ros2_distro}"
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${kit_ros2_humble_lib}"
fi

# 经典 Isaac Sim 的环境隔离：部分机器的登录 shell 会导出经典安装遗留的
# ISAAC_PATH/EXP_PATH/CARB_APP_PATH（如 /opt/isaac-sim-4.5、~/isaacsim-6.0）。
# pip 版 isaacsim 必须自己接管这些变量，否则 AppLauncher 会在错误位置解析
# isaacsim.simulation_app 扩展（SimulationApp 为 None -> TypeError）。
# IsaacLab 官方 CLI（isaaclab/cli/commands/envs.py）同样在启动前 unset 这三个。
unset ISAAC_PATH EXP_PATH CARB_APP_PATH

# Omniverse Kit 首次运行会在交互式终端询问 EULA；无 tty 的自动化启动
# （GUI / 远程 ssh）会永久卡在提问上。部署即视为已接受，这里预先置好保险。
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"

if [[ -z "${MRS_ROBOT_SIM_ROOT:-}" ]]; then
    if [[ -d "${repo_root}/../isaac_sim_core" ]]; then
        sim_root="${repo_root}/.."
    elif [[ -d "${repo_root}/../MRS_ROBOT_sim/isaac_sim_core" ]]; then
        sim_root="${repo_root}/../MRS_ROBOT_sim"
    else
        echo "MRS_ROBOT_sim root not found; set MRS_ROBOT_SIM_ROOT explicitly." >&2
        exit 2
    fi
    MRS_ROBOT_SIM_ROOT="${sim_root}"
fi
MRS_ROBOT_SIM_ROOT="$(realpath "${MRS_ROBOT_SIM_ROOT}")"
if [[ ! -d "${MRS_ROBOT_SIM_ROOT}/sim_runtime" ]]; then
    echo "MRS_ROBOT_SIM_ROOT does not contain sim_runtime: ${MRS_ROBOT_SIM_ROOT}" >&2
    exit 2
fi
export MRS_ROBOT_SIM_ROOT
export PYTHONPATH="${repo_root}/src:${MRS_ROBOT_SIM_ROOT}/isaaclab_ext/src:${MRS_ROBOT_SIM_ROOT}/sim_runtime/teleoperation/src:${MRS_ROBOT_SIM_ROOT}/sim_runtime/ros2/mrs_robot_arena_bridge:${MRS_ROBOT_SIM_ROOT}/sim_runtime/ros2/openflex_isaac_sim/openflex_isaac_contract"
if [[ -n "${filtered_pythonpath}" ]]; then
    export PYTHONPATH+=":${filtered_pythonpath}"
fi

target_script="$1"
shift
if [[ "${target_script}" != /* ]]; then
    target_script="${repo_root}/${target_script}"
fi
if [[ ! -f "${target_script}" ]]; then
    echo "Arena script not found: ${target_script}" >&2
    exit 2
fi

exec "${python_bin}" "${target_script}" "$@"
