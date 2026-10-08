from mrs_arena.environments.openflex_task_registry import (
    get_openflex_task_profile,
    list_openflex_task_profiles,
)


def test_first_party_openflex_task_registry_exposes_box_and_cube_tasks():
    profiles = {profile.task_id: profile for profile in list_openflex_task_profiles()}

    assert "openflex_dual_arm_box_transport" in profiles
    assert "openflex_pick_cube" in profiles
    assert "openflex_navigation_to_goal" in profiles
    assert get_openflex_task_profile("openflex_dual_arm_box_transport").scene_id == (
        "openflex_box_transport"
    )


def test_first_party_openflex_task_registry_rejects_unknown_task():
    import pytest

    with pytest.raises(KeyError, match="unknown OpenFlex task"):
        get_openflex_task_profile("missing")
