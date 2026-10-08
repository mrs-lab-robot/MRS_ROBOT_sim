#!/usr/bin/env python3
"""Start the robot and ROS 2 control path without any sensor component or visualization."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    Shutdown,
    TimerAction,
    UnsetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _fastdds_profile_action(profile_path: str):
    profile_path = profile_path.strip()
    if profile_path:
        return SetEnvironmentVariable(
            name="FASTRTPS_DEFAULT_PROFILES_FILE",
            value=profile_path,
        )
    return UnsetEnvironmentVariable(name="FASTRTPS_DEFAULT_PROFILES_FILE")


def _apply_fastdds_profile(context, *args, **kwargs):
    profile_path = LaunchConfiguration("fastdds_profiles_file").perform(context)
    return [_fastdds_profile_action(profile_path)]


def _apply_ros_localhost_only(context, *args, **kwargs):
    value = LaunchConfiguration("ros_localhost_only").perform(context)
    return [SetEnvironmentVariable(name="ROS_LOCALHOST_ONLY", value=value)]


def _resolve_isaac_python(install: Path) -> Path | None:
    """Resolve the interpreter for a workstation install or a pip environment."""
    for candidate in (install / "python.sh", install / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def _isaac_sim_process_environment(install: Path, interpreter: Path) -> dict[str, str]:
    """Keep ROS overlays from shadowing Kit libraries in a pip Isaac process."""
    if interpreter != install / "bin" / "python":
        return {}

    inherited_library_paths = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if path
    ]
    return {
        "ISAACSIM_PATH": str(install),
        "LD_LIBRARY_PATH": os.pathsep.join(
            (str(install / "lib"), *inherited_library_paths)
        ),
        # ROS Humble overlays are built for Python 3.10 and must not leak into
        # Isaac Sim 6's Python 3.12 interpreter. The runtime adds its own
        # consolidated pure-Python sensor package path after it starts.
        "PYTHONPATH": "",
        # Match the validated Isaac environment shim; CUDA is provided by the
        # Kit/PyTorch runtime, not the workstation's standalone toolkit path.
        "CUDA_HOME": "",
    }


def _default_sensor_asset_dir() -> Path:
    configured = os.environ.get("ISAACSIM_ROBOT_ROOT", "").strip()
    if not configured:
        configured = os.environ.get("ISAACSIM_SENSOR_ASSET_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "isaac_sim_core").is_dir():
            return candidate
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent, parent / "MRS_ROBOT_sim", parent / "isaacsim_robot"):
            if (candidate / "isaac_sim_core").is_dir():
                return candidate
    raise RuntimeError(
        "Isaac Sim asset root was not found. Launch from the MRS_ROBOT_sim "
        "repository root or set ISAACSIM_ROBOT_ROOT."
    )


def _default_isaac_path() -> Path:
    configured_path = os.environ.get("ISAACSIM_PATH", "").strip()
    candidates = ([Path(configured_path).expanduser()] if configured_path else []) + [
        Path.home() / "isaacsim-6.0",
        Path.home() / "isaacsim",
        Path("/opt/isaac-sim"),
        Path("/opt/isaac-sim-5.1"),
        Path("/isaac-sim"),
    ]
    for candidate in candidates:
        for install in (candidate, candidate / "runtime"):
            if _resolve_isaac_python(install) is not None:
                return install
    return Path("/isaac-sim")


def _default_generated_urdf() -> Path:
    configured = os.environ.get("ISAACSIM_ROBOT_GENERATED_DIR", "").strip()
    if configured:
        return Path(configured).expanduser() / "robot_control_only.urdf"
    repository_root = os.environ.get("ISAACSIM_ROBOT_ROOT", "").strip()
    if repository_root:
        return (
            Path(repository_root).expanduser()
            / "reports"
            / "runtime"
            / "generated"
            / "robot_control_only.urdf"
        )
    # Keep generated launch artifacts inside the workspace by default.  This
    # makes the launch usable in containers and managed runners where the
    # user's home directory may be read-only, while still allowing callers to
    # override the location with ISAACSIM_ROBOT_GENERATED_DIR.
    for parent in Path(__file__).resolve().parents:
        if (parent / "isaac_sim_core").is_dir():
            return parent / "reports" / "runtime" / "generated" / "robot_control_only.urdf"
        if (parent / "src" / "openflex_isaac_sim").is_dir():
            return parent / "reports" / "runtime" / "generated" / "robot_control_only.urdf"
        if (parent / "openflex_ws").is_dir():
            return parent / "openflex_ws" / "reports" / "runtime" / "generated" / "robot_control_only.urdf"
    return Path("/tmp/openflex_robot_control_only.urdf")


def _default_robot_usd() -> Path:
    configured_root = os.environ.get("ISAACSIM_ROBOT_ROOT", "").strip()
    if configured_root:
        return (
            Path(configured_root).expanduser()
            / "sim_runtime"
            / "assets"
            / "robots"
            / "openflex_robot.usda"
        )

    for parent in Path(__file__).resolve().parents:
        for candidate in (
            parent / "sim_runtime" / "assets" / "robots" / "openflex_robot.usda",
            parent / "assets" / "robots" / "openflex_robot.usda",
            parent / "isaac_sim_core" / "assets" / "robots" / "openflex_robot.usda",
        ):
            if candidate.is_file():
                return candidate

    raise RuntimeError(
        "Canonical OpenFlex USD asset was not found. Set ISAACSIM_ROBOT_ROOT "
        "or pass robot_usd:=/absolute/path/to/openflex_robot.usda."
    )


def _resolve_source_xacro(value: str) -> Path:
    normalized = value.strip().lower()
    if normalized in ("", "auto", "default"):
        share = Path(get_package_share_directory("openflex_isaac_description"))
        return share / "urdf" / "openflex_robot.urdf.xacro"
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _assert_api_port_available(host: str, port: int, service_name: str = "Isaac REST API") -> None:
    bind_host = host.strip() or "127.0.0.1"
    if bind_host == "localhost":
        bind_host = "127.0.0.1"
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((bind_host, port))
        except OSError as error:
            raise RuntimeError(
                f"{service_name} port {bind_host}:{port} is already in use. "
                "Stop the stale Isaac/launch process or choose another api_port before starting."
            ) from error


def _generate_sensor_free_urdf(context, output: Path) -> str:
    source = _resolve_source_xacro(LaunchConfiguration("source_xacro").perform(context))
    generator = (
        Path(get_package_prefix("openflex_isaac_description"))
        / "lib"
        / "openflex_isaac_description"
        / "generate_isaac_urdf.py"
    )
    if not source.is_file():
        raise RuntimeError(f"source_xacro does not exist: {source}")
    if not generator.is_file():
        raise RuntimeError(f"URDF generator is not installed: {generator}")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(generator),
        "--source",
        str(source),
        "--output",
        str(output),
        "--robot-name",
        LaunchConfiguration("robot_name").perform(context),
        "--command-topic",
        LaunchConfiguration("command_topic").perform(context),
        "--state-topic",
        LaunchConfiguration("state_topic").perform(context),
        "--enable-head",
        LaunchConfiguration("enable_head").perform(context),
        "--sensor-profile",
        "none",
        "--disable-sensors",
        "--preserve-visual-materials",
    ]
    if not _as_bool(LaunchConfiguration("show_head_camera").perform(context)):
        command.append("--hide-head-camera")
    if not _as_bool(LaunchConfiguration("show_lift_mast").perform(context)):
        command.append("--hide-lift-mast")
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "sensor-free URDF generation failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    urdf = output.read_text(encoding="utf-8")
    if "<sensor" in urdf or "<isaac:sensor" in urdf:
        raise RuntimeError(f"generated URDF contains sensor definitions: {output}")
    return urdf


def _controller_chain(context, controller_file: Path):
    controllers = ["joint_state_broadcaster", "swerve_drive_controller"]
    if _as_bool(LaunchConfiguration("start_upper_body").perform(context)):
        controllers.extend(
            (
                "left_forward_position_controller",
                "right_forward_position_controller",
                "head_forward_position_controller",
                "lift_position_controller",
            )
        )
    upper_body_param_files = {
        "left_forward_position_controller": controller_file.parent / "left_arm_controller.yaml",
        "right_forward_position_controller": controller_file.parent / "right_arm_controller.yaml",
        "head_forward_position_controller": controller_file.parent / "head_controller.yaml",
        "lift_position_controller": controller_file.parent / "lift_controller.yaml",
    }
    nodes = []
    for controller in controllers:
        parameter_file = upper_body_param_files.get(controller, controller_file)
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    controller,
                    "--controller-manager",
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "60",
                    "--service-call-timeout",
                    "60",
                    "--param-file",
                    str(parameter_file),
                ],
                output="both",
            )
        )

    def start_next_controller(event, _context, next_controller):
        if event.returncode != 0:
            # Log the failure but don't cascade-shutdown the launch; the OGN
            # graph publishes joint_states directly, so joint_state_broadcaster
            # failures are non-fatal. Other controllers (swerve, arms, head,
            # lift) are more critical.
            import launch.logging
            launch.logging.get_logger().warning(
                f"Controller spawner exited with code {event.returncode}; continuing launch"
            )
        return [next_controller]

    actions = []
    for previous, current in zip(nodes, nodes[1:]):
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=previous,
                    on_exit=lambda event, context, next_controller=current: start_next_controller(
                        event, context, next_controller
                    ),
                )
            )
        )
    actions.append(TimerAction(period=1.0, actions=[nodes[0]]))
    return actions


def launch_setup(context, *args, **kwargs):
    bringup_share = Path(get_package_share_directory("openflex_isaac_bringup"))
    requested_profile = LaunchConfiguration("sensor_profile").perform(context).strip().lower()
    if requested_profile != "none":
        raise RuntimeError(
            "sensor_profile must be 'none'; create and destroy sensors from the GUI after the robot is running"
        )
    if LaunchConfiguration("lidar_mount_mode").perform(context).strip().lower() != "parented":
        raise RuntimeError(
            "runtime MID360 sensors must use lidar_mount_mode:=parented"
        )
    sensor_profile = "none"
    output_urdf = _resolve_path(LaunchConfiguration("generated_urdf").perform(context))
    robot_description = _generate_sensor_free_urdf(context, output_urdf)
    robot_usd = _resolve_path(LaunchConfiguration("robot_usd").perform(context))
    if not robot_usd.is_file():
        raise RuntimeError(f"robot USD asset does not exist: {robot_usd}")
    controller_file = bringup_share / "config" / "controllers.isaac.mobile_base.yaml"
    stage = LaunchConfiguration("stage").perform(context).strip()
    if not stage or stage.lower() in ("auto", "default"):
        stage_path = bringup_share / "config" / "empty_stage.usd"
    else:
        stage_path = _resolve_path(stage)
    if not stage_path.is_file():
        raise RuntimeError(f"robot control stage does not exist: {stage_path}")
    if not controller_file.is_file():
        raise RuntimeError(f"controller configuration does not exist: {controller_file}")

    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    controller_use_sim_time = ParameterValue(
        LaunchConfiguration("controller_use_sim_time"), value_type=bool
    )
    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        name="controller_manager",
        output="both",
        parameters=[
            {
                "robot_description": robot_description,
                # Keep controller-manager timing independently selectable. A
                # wall-clock controller loop can still publish at its 90 Hz
                # target when the Isaac scene runs below RTF 1; the rest of
                # the visualization/state-publisher stack may continue using
                # simulation time.
                "use_sim_time": controller_use_sim_time,
            },
            ParameterFile(str(controller_file)),
        ],
        remappings=[("~/robot_description", "/robot_description")],
        on_exit=Shutdown(),
    )
    state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description, "use_sim_time": use_sim_time}],
    )
    description_publisher = Node(
        package="openflex_isaac_bringup",
        executable="robot_description_publisher.py",
        name="openflex_robot_description_publisher",
        output="both",
        parameters=[
            {
                "urdf_path": str(output_urdf),
                "topic": "/robot_description",
                "republish_period_sec": 0.0,
            }
        ],
    )

    scripts_share = Path(get_package_share_directory("isaac_ros2_scripts"))
    isaac_install = _resolve_path(LaunchConfiguration("isaac_path").perform(context))
    isaac_python = _resolve_isaac_python(isaac_install)
    if isaac_python is None:
        raise RuntimeError(
            "Isaac Sim interpreter was not found under "
            f"{isaac_install}; expected python.sh or bin/python"
        )
    start_script = (
        Path(get_package_prefix("openflex_isaac_bringup"))
        / "lib"
        / "openflex_isaac_bringup"
        / "start_robot_control_sim.py"
    )
    if not start_script.is_file():
        raise RuntimeError(f"control-only Isaac launcher is not installed: {start_script}")
    api_host = LaunchConfiguration("api_host").perform(context)
    api_port = LaunchConfiguration("api_port").perform(context)
    _assert_api_port_available(api_host, int(api_port))
    sensor_control_port = int(
        LaunchConfiguration("sensor_control_port").perform(context)
    )
    _assert_api_port_available(
        "127.0.0.1", sensor_control_port, service_name="sensor control API"
    )
    physics_hz = LaunchConfiguration("physics_hz").perform(context)
    render_hz = LaunchConfiguration("render_hz").perform(context)
    headless = LaunchConfiguration("headless").perform(context)
    sim_process_environment = _isaac_sim_process_environment(
        isaac_install, isaac_python
    )
    sim_process_environment.update(
        {
            "OPENFLEX_ISAAC_SENSOR_ASSET_DIR": str(
                _resolve_path(LaunchConfiguration("sensor_asset_dir").perform(context))
            ),
            "OPENFLEX_ROBOT_USD": str(robot_usd.resolve()),
            "OPENFLEX_ISAAC_MID360_ASSET_DIR": str(
                _resolve_path(LaunchConfiguration("sensor_asset_dir").perform(context))
            ),
            "OPENFLEX_ROBOT_SENSOR_PROFILE": sensor_profile,
            "OPENFLEX_MID360_LIDAR_TRANSPORT": LaunchConfiguration(
                "lidar_transport"
            ).perform(context),
            "OPENFLEX_MID360_OBJECT_ID_MAP": LaunchConfiguration(
                "lidar_object_id_map"
            ).perform(context),
            "OPENFLEX_SENSOR_CONTROL_PORT": LaunchConfiguration(
                "sensor_control_port"
            ).perform(context),
        }
    )
    sim_process = ExecuteProcess(
        cmd=[
            str(isaac_python),
            str(start_script),
            str(scripts_share),
            str(stage_path),
            render_hz,
            physics_hz,
            physics_hz,
            headless,
            api_port,
        ],
        output="both",
        name="isaac_robot_control_sim",
        additional_env=sim_process_environment,
    )

    spawn_node = Node(
        package="openflex_isaac_bringup",
        executable="spawn_robot_when_ready.py",
        name="spawn_robot_control_only",
        output="both",
        parameters=[
            {
                "urdf_path": str(output_urdf),
                "usd_path": str(robot_usd),
                "x": float(LaunchConfiguration("x").perform(context)),
                "y": float(LaunchConfiguration("y").perform(context)),
                "z": float(LaunchConfiguration("z").perform(context)),
                "roll": float(LaunchConfiguration("roll").perform(context)),
                "pitch": float(LaunchConfiguration("pitch").perform(context)),
                "yaw": float(LaunchConfiguration("yaw").perform(context)),
                "fixed": _as_bool(LaunchConfiguration("fixed").perform(context)),
                # The core robot launch always stays sensor-free. Sensor
                # resources are registered as factories and instantiated only
                # through the runtime lifecycle API after the control graph is ready.
                "enable_sensors": False,
                "apply_appearance": False,
                "api_host": api_host,
                "api_port": int(api_port),
                "wait_timeout": float(LaunchConfiguration("spawn_wait_timeout").perform(context)),
                "poll_period": 2.0,
                "auto_play": True,
                "play_timeout": 60.0,
                "post_play_settle_sec": 0.0,
            }
        ],
    )
    sim_ready = Node(
        package="openflex_isaac_bringup",
        executable="wait_for_sim_ready.py",
        name="wait_for_robot_control_ready",
        output="both",
        parameters=[
            {
                "clock_topic": "/clock",
                "joint_states_topic": LaunchConfiguration("state_topic"),
                "timeout_sec": float(LaunchConfiguration("sim_ready_timeout").perform(context)),
                "log_period_sec": 5.0,
                "require_nonzero_clock": True,
                "require_nonzero_joint_stamp": True,
            }
        ],
    )
    def start_control(event, _context):
        if event.returncode != 0:
            return [Shutdown()]
        return [controller_manager, *_controller_chain(context, controller_file)]

    def start_ready_gate(event, _context):
        if event.returncode != 0:
            return [Shutdown()]
        return [sim_ready]


    actions = [
        state_publisher,
        description_publisher,
        Node(
            package="openflex_isaac_bringup",
            executable="vla_contract_bridge.py",
            name="openflex_vla_contract_bridge",
            output="both",
        ),
        Node(
            package="openflex_isaac_bringup",
            executable="isaacsim_compat_bridge.py",
            name="openflex_sim_sensor_contract_bridge",
            output="both",
            parameters=[
                {
                    "sensor_only_mode": True,
                    "sim_lidar_topic": "/openflex/livox_frame/lidar",
                    "livox_lidar_topic": "/livox/lidar",
                    "livox_lidar_mode": "custom",
                    "pointcloud_topics": ["/livox/lidar_points"],
                    "scan_topics": ["/scan"],
                    "publish_scan_from_pointcloud": True,
                    "publish_livox_imu": False,
                    "publish_battery_state": False,
                }
            ],
        ),
        sim_process,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn_node, on_exit=start_ready_gate)
        ),
        RegisterEventHandler(
            OnProcessExit(target_action=sim_ready, on_exit=start_control)
        ),
        TimerAction(period=2.0, actions=[spawn_node]),
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    bringup_share = Path(get_package_share_directory("openflex_isaac_bringup"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_type", default_value="openflex", choices=["openflex"]),
            DeclareLaunchArgument("source_xacro", default_value="auto"),
            DeclareLaunchArgument("generated_urdf", default_value=str(_default_generated_urdf())),
            DeclareLaunchArgument(
                "robot_usd",
                default_value=str(_default_robot_usd()),
                description="Canonical articulated OpenFlex USD referenced directly by Isaac Sim.",
            ),
            DeclareLaunchArgument("robot_name", default_value="openflex"),
            DeclareLaunchArgument("command_topic", default_value="/openflex/joint_command"),
            DeclareLaunchArgument("state_topic", default_value="/openflex/joint_states"),
            DeclareLaunchArgument("enable_head", default_value="true"),
            DeclareLaunchArgument(
                "show_head_camera",
                default_value="true",
                description="Show the physical Femto camera mesh on the head; disable only for mesh debugging.",
            ),
            DeclareLaunchArgument(
                "show_lift_mast",
                default_value="true",
                description="Show the tall lift mast visual; collision and lift control remain active when hidden.",
            ),
            DeclareLaunchArgument(
                "sensor_profile",
                default_value="none",
                choices=["none"],
                description="The robot core starts without sensor prims; sensors are managed dynamically from the GUI.",
            ),
            DeclareLaunchArgument(
                "lidar_transport",
                default_value="helper",
                description="MID360 ROS transport; helper uses the historical LidarSensor render-product chain",
            ),
            DeclareLaunchArgument(
                "lidar_mount_mode",
                default_value="parented",
                choices=["parented"],
                description="Runtime MID360 is mounted below the robot hierarchy and follows its motion.",
            ),
            DeclareLaunchArgument(
                "lidar_object_id_map",
                default_value="false",
                description="Publish the MID360 object-id map topic; the native transport has no "
                "StableIdMap annotator, so object-id output must stay off there",
            ),
            DeclareLaunchArgument(
                "sensor_asset_dir",
                default_value=str(_default_sensor_asset_dir()),
                description="consolidated sensor asset root or historical sensor repository root",
            ),
            DeclareLaunchArgument("stage", default_value="auto"),
            DeclareLaunchArgument("isaac_path", default_value=str(_default_isaac_path())),
            DeclareLaunchArgument("headless", default_value="true"),
            DeclareLaunchArgument("start_upper_body", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            # The real robot controller contract is a wall-clock 90 Hz loop.
            # Isaac /clock remains available to the rest of the simulated
            # graph, but tying controller-manager scheduling to it would make
            # the externally observed /joint_states rate fall with RTF.
            DeclareLaunchArgument("controller_use_sim_time", default_value="false"),
            DeclareLaunchArgument("api_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("api_port", default_value="8085"),
            DeclareLaunchArgument("sensor_control_port", default_value="8086"),
            DeclareLaunchArgument("physics_hz", default_value="120.0"),
            # No sensor render product is required for the control-only path.
            # 30 Hz is sufficient for diagnostic capture and keeps the
            # complete robot above the real-time threshold. Use 60 Hz only
            # when an interactive Isaac viewport specifically needs it.
            DeclareLaunchArgument("render_hz", default_value="30.0"),
            DeclareLaunchArgument("spawn_wait_timeout", default_value="900.0"),
            DeclareLaunchArgument("sim_ready_timeout", default_value="180.0"),
            DeclareLaunchArgument("x", default_value="0.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.25"),
            DeclareLaunchArgument("roll", default_value="0.0"),
            DeclareLaunchArgument("pitch", default_value="0.0"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument("fixed", default_value="false"),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="49",
                description="ROS 2 domain used by Isaac Sim, controllers, and standalone RViz.",
            ),
            DeclareLaunchArgument(
                "ros_localhost_only",
                default_value="1",
                description=(
                    "Restrict the simulation ROS graph to this machine by default; "
                    "set to 0 only when remote ROS peers are required."
                ),
            ),
            DeclareLaunchArgument(
                "fastdds_profiles_file",
                default_value="",
                description=(
                    "Fast DDS XML profile file; empty selects local/default discovery "
                    "and clears any inherited remote-only profile."
                ),
            ),
            SetEnvironmentVariable(
                name="ROS_DOMAIN_ID",
                value=LaunchConfiguration("ros_domain_id"),
            ),
            OpaqueFunction(function=_apply_ros_localhost_only),
            OpaqueFunction(function=_apply_fastdds_profile),
            OpaqueFunction(function=launch_setup),
        ]
    )
