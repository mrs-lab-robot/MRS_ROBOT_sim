from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mrs_robot_lab.contracts import ContractError, load_embodiment_contract


class EmbodimentContractTest(unittest.TestCase):
    def _write_contract(self, content: str) -> Path:
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        contract_path = Path(temporary_dir.name) / "embodiment.yaml"
        contract_path.write_text(content, encoding="utf-8")
        return contract_path

    def test_loads_dimensions_and_only_counted_joint_actions(self) -> None:
        path = self._write_contract(
            "schema_version: 1\n"
            "robot:\n  id: openflex\n"
            "actions:\n"
            "  - name: base_twist\n    dimension: 3\n    fields: [linear.x, linear.y, angular.z]\n"
            "  - name: lift_position\n    dimension: 1\n    joints: [lift_joint]\n    fields: [lift_joint]\n"
            "  - name: lift_velocity\n    dimension: 1\n    joints: [lift_joint]\n    fields: [lift_joint]\n"
            "    counts_toward_action: false\n"
            "observation_layout:\n"
            "  - observation: joint_state\n    dimension: 2\n"
            "    fields: [position.lift_joint, velocity.lift_joint]\n"
        )

        contract = load_embodiment_contract(path)

        self.assertEqual(contract.robot_id, "openflex")
        self.assertEqual(contract.action_dimensions, {"base_twist": 3, "lift_position": 1, "lift_velocity": 1})
        self.assertEqual(contract.action_fields["base_twist"], ("linear.x", "linear.y", "angular.z"))
        self.assertEqual(contract.joint_action_names, ("lift_joint",))
        self.assertEqual(contract.observation_dimensions, {"joint_state": 2})
        self.assertEqual(
            contract.observation_fields,
            {"joint_state": ("position.lift_joint", "velocity.lift_joint")},
        )

    def test_rejects_action_dimension_that_disagrees_with_declared_fields(self) -> None:
        path = self._write_contract(
            "schema_version: 1\nrobot:\n  id: openflex\n"
            "actions:\n"
            "  - name: lift_position\n    dimension: 2\n    joints: [lift_joint]\n    fields: [lift_joint]\n"
            "observation_layout: []\n"
        )

        with self.assertRaisesRegex(ContractError, "lift_position.*dimension 2.*1 field"):
            load_embodiment_contract(path)

    def test_exposes_joint_groups_and_limits_from_the_single_source_contract(self) -> None:
        path = self._write_contract(
            "schema_version: 1\nrobot:\n  id: openflex\n"
            "actions:\n"
            "  - name: left_arm_position\n    dimension: 2\n"
            "    joints: [left_joint1, left_joint2]\n"
            "    fields: [left_joint1, left_joint2]\n"
            "    limits: [{minimum: -1.0, maximum: 1.0}, {minimum: -2.0, maximum: 2.0}]\n"
            "observation_layout:\n"
            "  - observation: joint_state\n    dimension: 4\n"
            "    fields: [position.left_joint1, position.left_joint2, velocity.left_joint1, velocity.left_joint2]\n"
        )

        contract = load_embodiment_contract(path)

        self.assertEqual(contract.action_joints["left_arm_position"], ("left_joint1", "left_joint2"))
        self.assertEqual(
            contract.action_limits["left_arm_position"],
            ((-1.0, 1.0), (-2.0, 2.0)),
        )


if __name__ == "__main__":
    unittest.main()
