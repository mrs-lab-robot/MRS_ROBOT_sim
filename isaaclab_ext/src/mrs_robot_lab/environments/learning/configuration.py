"""Load one versioned task and its referenced scene/assets for runtime use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openflex_isaac_contract.scene_spec import SceneSpec, load_scene_spec
from openflex_isaac_contract.task_spec import TaskSpec, load_task_spec


@dataclass(frozen=True)
class TaskRuntimeConfiguration:
    task: TaskSpec
    scene: SceneSpec
    task_spec_path: Path
    scene_spec_path: Path
    base_stage_path: Path
    object_asset_paths: dict[str, Path]


def load_task_configuration(
    task_spec_path: str | Path,
    *,
    scene_spec_path: str | Path | None = None,
    backend: str = "isaac_lab",
) -> TaskRuntimeConfiguration:
    """Load a task, infer its sibling scene YAML, and resolve all USD references.

    Relative scene asset references are interpreted relative to the scene YAML,
    never the process working directory. This keeps local, remote, GUI and
    training invocations independent of where they were launched from.
    """

    task_path = Path(task_spec_path).expanduser().resolve()
    task = load_task_spec(task_path)
    if task.supported_backends and backend not in task.supported_backends:
        raise ValueError(f"task {task.task_id!r} does not support backend {backend!r}")

    if scene_spec_path is None:
        candidate = task_path.parent.parent / "scenes" / f"{task.scene_id}.yaml"
        scene_path = candidate.resolve()
    else:
        scene_path = Path(scene_spec_path).expanduser().resolve()
    scene = load_scene_spec(scene_path)
    if scene.scene_id != task.scene_id:
        raise ValueError(
            f"task {task.task_id!r} references scene {task.scene_id!r}, "
            f"but {scene_path} defines {scene.scene_id!r}"
        )

    base_stage_path = _resolve_file(scene_path.parent, scene.base_stage_usd, "base_stage_usd")
    object_asset_paths: dict[str, Path] = {}
    for placement in scene.objects:
        if placement.usd_path:
            object_asset_paths[placement.object_id] = _resolve_file(
                scene_path.parent, placement.usd_path, f"objects.{placement.object_id}.usd_path"
            )
    return TaskRuntimeConfiguration(
        task=task,
        scene=scene,
        task_spec_path=task_path,
        scene_spec_path=scene_path,
        base_stage_path=base_stage_path,
        object_asset_paths=object_asset_paths,
    )


def _resolve_file(base_dir: Path, reference: str, field_name: str) -> Path:
    path = Path(reference).expanduser()
    resolved = (path if path.is_absolute() else base_dir / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} asset does not exist: {resolved}")
    return resolved
