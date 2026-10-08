from __future__ import annotations

from pathlib import Path
import unittest

from openflex_isaac_contract.task_spec import load_task_spec


SIM_ROOT = Path(__file__).resolve().parents[3]


class NavigationTaskConfigTest(unittest.TestCase):
    def test_shared_navigation_task_declares_arena_as_a_supported_backend(self) -> None:
        task = load_task_spec(SIM_ROOT / "sim_runtime/config/tasks/navigation_to_goal.yaml")

        self.assertIn("isaac_lab", task.supported_backends)
        self.assertIn("arena", task.supported_backends)
        self.assertEqual(task.task_id, "navigation_to_goal")


if __name__ == "__main__":
    unittest.main()
