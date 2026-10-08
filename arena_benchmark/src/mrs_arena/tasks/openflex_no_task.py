"""Arena NoTask shim that accepts graph-level task descriptions."""

from __future__ import annotations

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.tasks.no_task import NoTask


@register_task
class OpenFlexNoTask(NoTask):
    """Preserve Arena's no-task behavior while matching the graph task contract."""

    def __init__(self, task_description: str | None = None):
        super().__init__()
        self.task_description = task_description
