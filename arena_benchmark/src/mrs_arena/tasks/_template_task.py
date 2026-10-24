"""新任务模板（复制本文件为 <task_id>.py 再改）。

规矩见 `docs/task-authoring.md`：
  1. 物体位姿只从 task/scene YAML 读（用 `spec_placement` 里的工具）；
  2. 判据只吃数据、不读 YAML；
  3. 任务类只做装配，不写死数值。

这个文件本身不进 registry（文件名以下划线开头，registry 是显式列表）。
`get_scene_cfg()` 留成 NotImplementedError 是有意的：场景组装每个任务都不一样，
照着 `openflex_navigation_task.py` / `openflex_pick_cube_task.py` 写那一段即可，
但**取数值**的部分已经写好，直接复用。
"""

from __future__ import annotations

from pathlib import Path
import os

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from mrs_arena.tasks.spec_placement import (
    ObjectPlacement,
    PlacementRandomization,
    object_placement_from_spec,
    randomization_from_spec,
    target_position_from_spec,
)


def default_template_task_path() -> Path:
    configured_root = os.environ.get("MRS_ROBOT_SIM_ROOT")
    sim_root = (
        Path(configured_root).expanduser().resolve()
        if configured_root
        else Path(__file__).resolve().parents[4]
    )
    return sim_root / "sim_runtime/config/tasks/_template_task.yaml"


class TemplateTask(TaskBase):
    """骨架：把 YAML 里的数值读出来，交给场景组装和判据。"""

    DEFAULT_OBJECT_ID = "replace_me_object"

    def __init__(
        self,
        task_spec_path: str | Path | None = None,
        *,
        object_id: str | None = None,
        episode_length_s: float | None = None,
    ) -> None:
        from openflex_isaac_contract.task_spec import load_task_spec

        self.task_spec_path = Path(task_spec_path or default_template_task_path())
        self.spec = load_task_spec(self.task_spec_path)
        self.object_id = object_id or self.DEFAULT_OBJECT_ID
        super().__init__(
            episode_length_s=episode_length_s or self.spec.episode_length_s,
            task_description=self.spec.description,
        )

    # ---- 数值全部从这里取，不要在别处再写一份 --------------------------------

    def object_placement(self) -> ObjectPlacement:
        """物体初始位姿（来自 initial_state.<object_id>_pose）。"""

        return object_placement_from_spec(self.spec, self.object_id)

    def target_position(self) -> tuple[float, float, float]:
        """目标位置（来自 success_criteria.params.target_position）。"""

        return target_position_from_spec(self.spec, self.object_id)

    def randomization(self) -> PlacementRandomization:
        """随机化幅度（来自 randomization_config）。"""

        return randomization_from_spec(self.spec)

    # ---- 场景组装：每个任务都不一样，照现有任务写 -----------------------------

    def get_scene_cfg(self):  # noqa: D401 - 模板占位
        raise NotImplementedError(
            "把这里换成你的场景组装：参考 openflex_navigation_task.get_scene_cfg() "
            "或 openflex_pick_cube_task；物体位姿请用 self.object_placement()，"
            "不要写坐标常量。"
        )

    def get_termination_cfg(self):
        """终止条件：照现有任务写（超时/物体掉落/成功）。模板里先给空配置。"""

        from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: F401

        return {}  # 换成 {"time_out": DoneTerm(func=...), "success": DoneTerm(func=...)}

    def get_events_cfg(self):
        """reset 事件：物体按 YAML 位姿重置就在这里做（用 self.object_placement()）。"""

        return {}

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Mimic 数据生成：没接线就明确返回 None，不要假装支持。"""

        return None

    def get_metrics(self):
        return [SuccessRateMetric()]


__all__ = ["TemplateTask", "default_template_task_path"]
