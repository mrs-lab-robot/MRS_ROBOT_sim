"""Launch VR input/solvers and the Arena relay, without starting robot hardware."""

import os
import socket
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _require_port_free(port: int, bind_host: str, label: str) -> None:
    """启动前占位检查：同一时间只允许一套 VR 遥操作栈。

    两套同时跑时第二套的 pico_pose_bridge 会抢不到 5100（报文被内核分给其中
    一个 socket），而它的 relay 又抢不到 24102 → 表现为“VR 完全驱动不了”，
    但日志里每个节点都"启动成功"。宁可在启动时直接失败。
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind((bind_host, port))
    except OSError as exc:
        raise RuntimeError(
            f"{label} 端口 {port} 已被占用（{exc}）。\n"
            "已有一套 VR 遥操作栈在运行：不要同时使用 GUI 的「Arena ROS 桥」和 "
            "sim_runtime/ros2/start_arena_vr_teleop.sh。先停掉已有的那套：\n"
            "  pgrep -af 'pico_pose_bridge|arena_relay'   # 看进程\n"
            "  pkill -f launch mrs_robot_arena_bridge      # 或按 PID 停"
        ) from exc
    finally:
        probe.close()

# relay_node 依赖 mrs_teleoperation（只在 sim_runtime/teleoperation/src 里，
# 没装进 ROS 的 python3.10）；不注入 PYTHONPATH 时 relay 启动即报
# ModuleNotFoundError: No module named 'mrs_teleoperation'，随后 relay 进程死掉，
# /joint_states、/cmd_vel、/cam_* 全部消失（机器人不动、相机流打不开）。
# 注意：这个 launch 位于 sim_runtime/ros2/<pkg>/launch，parents[3] 是 sim_runtime
# 而不是仓库根，写死层级会在目录调整后失效；改成向上找到含
# sim_runtime/teleoperation/src 的那一层。
def _find_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "sim_runtime" / "teleoperation" / "src").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


_REPO_ROOT = _find_repo_root()
_TELEOP_SRC = _REPO_ROOT / "sim_runtime" / "teleoperation" / "src"
_RELAY_ENV = {
    "PYTHONPATH": os.pathsep.join(
        [str(_TELEOP_SRC), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
}


def generate_launch_description():
    # 单实例检查：5100 = Pico 入口；24103/24104 = relay 自己绑定。
    # 注意 24102 是「仿真端」在监听（relay 只往它发），所以绝不能把它当作占用冲突。
    _require_port_free(5100, "0.0.0.0", "Pico 入口（pico_pose_bridge）")
    _require_port_free(24103, "127.0.0.1", "Arena 状态（relay）")
    _require_port_free(24104, "127.0.0.1", "Arena 相机（relay）")
    vr_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("openarmx_integrated_bringup"),
                "launch",
                "integrated_vr_teleop.launch.py",
            ])
        ),
        launch_arguments={
            "enable_joystick_control": "true",
            "enable_button_lift": "true",
            "enable_head_teleop": "true",
            "enable_waist_control": "false",
            # 底盘速度：真机默认 0.3/0.3，仿真里按手感提到 2.5 倍
            "max_linear_speed": "0.75",
            "boost_linear_speed": "0.75",
            "max_angular_speed": "0.75",
        }.items(),
    )
    relay = Node(
        package="mrs_robot_arena_bridge",
        executable="arena_relay",
        name="mrs_robot_arena_relay",
        output="screen",
        additional_env=_RELAY_ENV,
        parameters=[
            PathJoinSubstitution([
                FindPackageShare("mrs_robot_arena_bridge"), "config", "arena_vr_teleop.yaml"
            ]),
            {
                "arena_host": LaunchConfiguration("arena_host"),
                "state_bind_host": LaunchConfiguration("state_bind_host"),
                "camera_bind_host": LaunchConfiguration("camera_bind_host"),
            },
        ],
    )
    return LaunchDescription([
        DeclareLaunchArgument("arena_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("state_bind_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("camera_bind_host", default_value="127.0.0.1"),
        vr_launch,
        relay,
    ])
