"""Load versioned TaskSpec content and manage active task configuration.

Provides runtime task switching for Isaac Sim stage-1 workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openflex_isaac_contract.task_spec import load_task_spec


class TaskLoader:
    """Manage runtime task configuration from versioned TaskSpec YAML files.

    Loads TaskSpec definitions and maintains the active task configuration.
    Task switching replaces the active configuration without requiring
    simulator restart.
    """

    def __init__(self):
        self._active_task: dict[str, Any] | None = None

    def load_task(self, spec_path: str | Path) -> dict[str, Any]:
        """Load a TaskSpec from YAML and set it as the active task.

        Args:
            spec_path: Path to the TaskSpec YAML file

        Returns:
            Result dictionary with keys:
            - success (bool): Whether the load succeeded
            - task_id (str): Task identifier (on success)
            - scene_id (str): Required scene identifier (on success)
            - episode_length_s (float): Episode duration (on success)
            - success_criteria (list): Success criteria definitions (on success)
            - action_spec (dict | None): Action specification (on success)
            - initial_state (dict): Initial state configuration (on success)
            - randomization_config (dict): Randomization settings (on success)
            - reward_config (dict | None): Reward configuration (on success)
            - termination_config (dict | None): Termination configuration (on success)
            - metadata (dict): Task metadata (on success)
            - supported_backends (list): Supported backend list (on success)
            - capabilities (list): Capability list (on success)
            - error (str): Error message (on failure)
        """
        try:
            spec = load_task_spec(spec_path)

            # 将 TaskSpec 转换为字典格式
            result = {
                "success": True,
                "task_id": spec.task_id,
                "description": spec.description,
                "scene_id": spec.scene_id,
                "episode_length_s": spec.episode_length_s,
                "success_criteria": [
                    {
                        "criterion_type": crit.criterion_type,
                        "params": crit.params,
                        "tolerance": crit.tolerance,
                        "required": crit.required,
                    }
                    for crit in spec.success_criteria
                ],
                "action_spec": spec.action_spec,
                "initial_state": spec.initial_state,
                "randomization_config": spec.randomization_config,
                "reward_config": spec.reward_config,
                "termination_config": spec.termination_config,
                "metadata": spec.metadata,
                "supported_backends": list(spec.supported_backends),
                "capabilities": list(spec.capabilities),
            }

            # 设置为活动任务
            self._active_task = result

            return result

        except FileNotFoundError as exc:
            return {
                "success": False,
                "error": f"TaskSpec文件不存在: {exc}",
            }
        except ValueError as exc:
            return {
                "success": False,
                "error": f"TaskSpec验证失败: {exc}",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"任务加载异常: {exc}",
            }

    def get_active_task(self) -> dict[str, Any] | None:
        """Get the currently active task configuration.

        Returns:
            Active task configuration dictionary, or None if no task is loaded.
        """
        return self._active_task

    def clear_active_task(self) -> None:
        """Clear the active task configuration."""
        self._active_task = None
