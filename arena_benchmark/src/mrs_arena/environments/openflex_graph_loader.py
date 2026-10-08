"""Adapt upstream Arena graph YAMLs to the locally registered OpenFlex asset."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any


_DROID_REGISTRIES = {"droid_abs_joint_pos"}


def scene_only_graph_data(scene_data: dict[str, Any], env_name: str) -> dict[str, Any]:
    """Convert a scene YAML mapping into an Arena graph with the registered NoTask."""
    if not isinstance(scene_data, dict):
        raise ValueError("Arena 场景 YAML 必须是字典")
    if not env_name.strip():
        raise ValueError("Arena 场景运行名称不能为空")
    graph_data = deepcopy(scene_data)
    graph_data.pop("external_yaml", None)
    graph_data["env_name"] = env_name
    graph_data["task"] = {
        "composition": "atomic",
        "description": f"Load Arena scene {env_name} without an assigned task",
        "subtasks": [{"kind": "OpenFlexNoTask", "params": {}}],
    }
    return graph_data


def adapt_graph_spec_for_openflex(graph_spec: Any) -> Any:
    """Replace a parameter-free Droid embodiment while preserving graph IDs.

    Graph relations and task parameters refer to the embodiment by its graph
    ID, so only the asset registry name changes. Droid-specific embodiment
    parameters are rejected instead of being silently passed to OpenFlex.
    """
    embodiment = getattr(graph_spec, "embodiment", None)
    if embodiment is None:
        raise ValueError("Arena graph 缺少 embodiment")
    params = getattr(embodiment, "params", {}) or {}
    if not isinstance(params, dict):
        raise ValueError("Arena embodiment.params 必须是字典")
    if params:
        raise ValueError(
            "当前 OpenFlex 适配器不支持 Droid 专用参数："
            + ", ".join(sorted(params))
        )

    source_registry = str(getattr(embodiment, "registry_name", ""))
    if source_registry not in _DROID_REGISTRIES | {"openflex"}:
        raise ValueError(f"当前 OpenFlex 适配器不支持来源 embodiment：{source_registry or '未声明'}")
    embodiment.registry_name = "openflex"
    return graph_spec


def apply_simulation_rate_overrides(
    environment: Any, *, physics_hz: int, render_hz: int
) -> Any:
    """Wrap the YAML callback so shared GUI rates are applied to Isaac Lab."""
    physics_hz = int(physics_hz)
    render_hz = int(render_hz)
    if not 1 <= physics_hz <= 500:
        raise ValueError("physics_hz must be between 1 and 500")
    if not 1 <= render_hz <= min(120, physics_hz):
        raise ValueError("render_hz must be between 1 and min(120, physics_hz)")
    previous_callback = getattr(environment, "env_cfg_callback", None)

    def apply_to_config(env_cfg: Any) -> Any:
        if previous_callback is not None:
            env_cfg = previous_callback(env_cfg)
        env_cfg.sim.dt = 1.0 / physics_hz
        env_cfg.sim.render_interval = max(1, int(round(physics_hz / render_hz)))
        return env_cfg

    environment.env_cfg_callback = apply_to_config
    return environment


def apply_reset_randomization_overrides(
    environment: Any, randomization: dict[str, Any] | None
) -> Any:
    """Attach only the user-enabled, YAML-declared randomization terms to reset."""
    randomization = _validate_reset_randomization(randomization or {})
    if not randomization:
        return environment
    previous_callback = getattr(environment, "env_cfg_callback", None)

    def apply_to_config(env_cfg: Any) -> Any:
        if previous_callback is not None:
            env_cfg = previous_callback(env_cfg)
        from isaaclab.managers import EventTermCfg
        from isaaclab_arena.utils.configclass import (
            combine_configclass_instances,
            make_configclass,
        )

        event_fields = []
        for index, (asset_name, options) in enumerate(randomization.get("objects", {}).items()):
            event_fields.append((
                f"gui_reset_object_pose_{index}",
                EventTermCfg,
                EventTermCfg(
                    func=randomize_openflex_object_pose,
                    mode="reset",
                    params={
                        "asset_name": asset_name,
                        "position_xy_m": options.get("position_xy_m"),
                        "yaw_deg": options.get("yaw_deg"),
                    },
                ),
            ))
        light = randomization.get("light")
        if light:
            event_fields.append((
                "gui_reset_light_intensity",
                EventTermCfg,
                EventTermCfg(
                    func=randomize_usd_light_intensity,
                    mode="reset",
                    params={
                        "prim_path": light["prim_path"],
                        "intensity": light["intensity"],
                    },
                ),
            ))
        extra_events = make_configclass(
            "GuiResetRandomizationEventsCfg", event_fields
        )()
        env_cfg.events = combine_configclass_instances(
            "EventsCfg", env_cfg.events, extra_events
        )
        return env_cfg

    environment.env_cfg_callback = apply_to_config
    return environment


def _validate_reset_randomization(randomization: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(randomization, dict) or set(randomization) - {"objects", "light"}:
        raise ValueError("reset randomization must contain only objects and light")
    result: dict[str, Any] = {}
    objects = randomization.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError("reset randomization objects must be a mapping")
    if objects:
        result["objects"] = {}
    for asset_name, options in objects.items():
        if not isinstance(asset_name, str) or not asset_name.strip() or not isinstance(options, dict):
            raise ValueError("reset randomization contains an invalid object entry")
        if not options or set(options) - {"position_xy_m", "yaw_deg"}:
            raise ValueError(f"unsupported reset randomization option for {asset_name}")
        normalized = {}
        for option, bounds in options.items():
            low, high = _numeric_range(bounds, f"{asset_name}.{option}")
            if option == "position_xy_m" and (low < -1 or high > 1):
                raise ValueError("position_xy_m must be within [-1, 1] meters")
            if option == "yaw_deg" and (low < -180 or high > 180):
                raise ValueError("yaw_deg must be within [-180, 180]")
            normalized[option] = [low, high]
        result["objects"][asset_name] = normalized
    light = randomization.get("light")
    if light is not None:
        if not isinstance(light, dict) or set(light) != {"prim_path", "intensity"}:
            raise ValueError("light reset randomization requires prim_path and intensity")
        prim_path = light["prim_path"]
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise ValueError("light prim_path must be an absolute USD path")
        low, high = _numeric_range(light["intensity"], "light.intensity")
        if low < 0 or high > 100000:
            raise ValueError("light intensity must be within [0, 100000]")
        result["light"] = {"prim_path": prim_path, "intensity": [low, high]}
    return result


def _numeric_range(bounds: Any, label: str) -> tuple[float, float]:
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError(f"{label} must be a [min, max] range")
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} range must contain numbers") from error
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{label} range must be finite and ordered")
    return low, high


def randomize_openflex_object_pose(
    env: Any,
    env_ids: Any,
    *,
    asset_name: str,
    position_xy_m: list[float] | None,
    yaw_deg: list[float] | None,
) -> None:
    """Apply independent XY and yaw offsets around an object's default reset pose."""
    import torch
    from isaaclab.utils import math as math_utils

    asset = env.scene[asset_name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long).reshape(-1)
    state = asset.data.default_root_state[env_ids].clone()
    state[:, :3] += env.scene.env_origins[env_ids]
    count = len(env_ids)
    if position_xy_m is not None:
        bounds = torch.tensor(position_xy_m, device=asset.device, dtype=state.dtype)
        offsets = torch.rand((count, 2), device=asset.device, dtype=state.dtype)
        state[:, :2] += bounds[0] + offsets * (bounds[1] - bounds[0])
    if yaw_deg is not None:
        low, high = math.radians(yaw_deg[0]), math.radians(yaw_deg[1])
        yaw = torch.empty(count, device=asset.device, dtype=state.dtype).uniform_(low, high)
        zeros = torch.zeros_like(yaw)
        yaw_quaternion = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        state[:, 3:7] = math_utils.quat_mul(state[:, 3:7], yaw_quaternion)
    asset.write_root_pose_to_sim(state[:, :7], env_ids)
    asset.write_root_velocity_to_sim(state[:, 7:], env_ids)


def randomize_usd_light_intensity(
    env: Any, env_ids: Any, *, prim_path: str, intensity: list[float]
) -> None:
    """Resample one global USD light's intensity whenever the environment resets."""
    import omni.usd
    import torch

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"reset randomization light prim not found: {prim_path}")
    intensity_attribute = prim.GetAttribute("inputs:intensity")
    if not intensity_attribute or not intensity_attribute.IsValid():
        raise RuntimeError(f"USD light has no intensity attribute: {prim_path}")
    low, high = intensity
    sample = torch.rand((), device=getattr(env, "device", "cpu")).item()
    intensity_attribute.Set(float(low + sample * (high - low)))


def make_openflex_environment_from_yaml(
    task_yaml_path: str | Path,
    *,
    enable_cameras: bool = False,
):
    """Load the Arena graph and convert it after AppLauncher initialized Kit."""
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    # Importing this module registers the OpenFlex asset with Arena. Keep this
    # inside the factory: callers must initialize AppLauncher before reaching it.
    from mrs_arena.embodiments.openflex import OpenFlexEmbodiment as _OpenFlexEmbodiment

    path = Path(task_yaml_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Arena task YAML 不是文件：{path}")
    graph_spec = adapt_graph_spec_for_openflex(ArenaEnvGraphSpec.from_yaml(path))
    return graph_spec.to_arena_env(enable_cameras=enable_cameras)


def make_openflex_scene_environment_from_yaml(
    scene_yaml_path: str | Path,
    *,
    enable_cameras: bool = False,
):
    """Load the selected scene graph with Arena's NoTask leaf after Kit starts."""
    import re
    import yaml

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from mrs_arena.embodiments.openflex import OpenFlexEmbodiment as _OpenFlexEmbodiment
    from mrs_arena.tasks.openflex_no_task import OpenFlexNoTask as _OpenFlexNoTask

    path = Path(scene_yaml_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Arena scene YAML 不是文件：{path}")
    scene_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    env_name = re.sub(r"[^a-zA-Z0-9_]+", "_", path.stem).strip("_").lower()
    graph_data = scene_only_graph_data(scene_data, env_name or "openflex_arena_scene")
    graph_spec = ArenaEnvGraphSpec.from_dict(graph_data)
    graph_spec = adapt_graph_spec_for_openflex(graph_spec)
    return graph_spec.to_arena_env(enable_cameras=enable_cameras)
