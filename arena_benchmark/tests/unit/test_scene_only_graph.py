from mrs_arena.environments.openflex_graph_loader import scene_only_graph_data


def test_scene_only_graph_data_adds_registered_no_task_and_keeps_scene_assets():
    source = {
        "embodiment": {"id": "robot", "registry_name": "droid_abs_joint_pos", "params": {}},
        "background": {"id": "table", "registry_name": "table", "params": {}},
        "objects": [{"id": "bowl", "registry_name": "bowl", "params": {}}],
        "relations": [{"kind": "on", "subject": "bowl", "reference": "table", "params": {}}],
    }

    result = scene_only_graph_data(source, "robolab_scenes_banana_bowl")

    assert result["env_name"] == "robolab_scenes_banana_bowl"
    assert result["task"] == {
        "composition": "atomic",
        "description": "Load Arena scene robolab_scenes_banana_bowl without an assigned task",
        "subtasks": [{"kind": "OpenFlexNoTask", "params": {}}],
    }
    assert result["objects"] == source["objects"]
    assert result["relations"] == source["relations"]


def test_scene_only_graph_data_overrides_a_bundled_task_and_external_include():
    source = {
        "env_name": "task_with_scene",
        "external_yaml": "../scenes/scene.yaml",
        "embodiment": {"id": "robot"},
        "background": {"id": "table"},
        "task": {"composition": "atomic", "description": "old", "subtasks": []},
    }

    result = scene_only_graph_data(source, "selected_scene")

    assert "external_yaml" not in result
    assert result["task"]["subtasks"] == [{"kind": "OpenFlexNoTask", "params": {}}]
    assert result["env_name"] == "selected_scene"
