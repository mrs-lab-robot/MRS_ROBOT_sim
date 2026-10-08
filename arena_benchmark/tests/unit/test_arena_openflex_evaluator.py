from mrs_arena.learning.arena_openflex_evaluator import make_episode_result


def test_generic_arena_episode_result_keeps_seed_and_task_provenance():
    result = make_episode_result(
        task_id="openflex_dual_arm_box_transport",
        index=2,
        seed=44,
        steps=17,
        success=True,
        extras={"success_rate": 1.0},
    )

    assert result == {
        "task_id": "openflex_dual_arm_box_transport",
        "index": 2,
        "seed": 44,
        "steps": 17,
        "success": True,
        "extras": {"success_rate": 1.0},
    }
