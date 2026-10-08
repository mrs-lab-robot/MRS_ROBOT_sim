from math import isclose, pi, sin, cos
from types import SimpleNamespace

from mrs_arena.compat.scene_manifest import (
    build_openflex_scene_task_manifest,
    quaternion_xyzw_to_euler_xyz,
)


def test_quaternion_xyzw_is_converted_to_xyz_euler_angles():
    euler = quaternion_xyzw_to_euler_xyz((0.0, 0.0, sin(pi / 4), cos(pi / 4)))

    assert isclose(euler[0], 0.0, abs_tol=1e-9)
    assert isclose(euler[1], 0.0, abs_tol=1e-9)
    assert isclose(euler[2], pi / 2, abs_tol=1e-9)


def test_arena_layout_manifest_preserves_scale_and_offsets_scene_from_robot():
    table = SimpleNamespace(
        name="maple_table",
        usd_path="https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/table.usd",
        scale=(1.0, 1.0, 0.8),
        get_initial_pose=lambda: SimpleNamespace(
            position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)
        ),
    )
    banana = SimpleNamespace(
        name="banana",
        usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/banana.usd",
        scale=(0.5, 0.5, 0.5),
        get_initial_pose=lambda: SimpleNamespace(
            position_xyz=(0.2, 0.1, 0.75), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)
        ),
    )
    arena_env = SimpleNamespace(
        name="banana_in_bowl",
        scene=SimpleNamespace(assets={"maple_table": table, "banana": banana}),
    )
    graph_spec = SimpleNamespace(
        env_name="banana_in_bowl",
        background=SimpleNamespace(id="table_node", registry_name="maple_table"),
        objects=[SimpleNamespace(id="banana", registry_name="banana_asset")],
        task=SimpleNamespace(
            description="pick up the banana and place it in the bowl",
            subtasks=[
                SimpleNamespace(
                    kind="PickAndPlaceTask",
                    params={"pick_up_object": "banana", "destination_location": "bowl"},
                )
            ],
        ),
    )

    manifest = build_openflex_scene_task_manifest(
        arena_env,
        graph_spec,
        scene_id="arena_robolab_banana_bowl",
        task_id="arena_robolab_banana_in_bowl",
        source_scene_yaml="robolab/scenes/banana_bowl.yaml",
        source_task_yaml="robolab/tasks/banana_in_bowl.yaml",
        x_offset_m=1.05,
    )

    assets = {asset["role"]: asset for asset in manifest["scene"]["assets"]}
    assert assets["background"]["prim_name"] == "MRS_Arena_table_node"
    assert assets["background"]["pose"]["x"] == 1.05
    assert assets["background"]["pose"]["z"] == 0.0
    assert assets["background"]["scale"] == [1.0, 1.0, 0.8]
    assert assets["task_object"]["pose"]["x"] == 1.25
    assert assets["task_object"]["pose"]["y"] == 0.1
    assert assets["task_object"]["scale"] == [0.5, 0.5, 0.5]
    assert manifest["task"]["target_component"] == "MRS_Arena_banana"
    assert manifest["task"]["source"]["scene_yaml"] == "robolab/scenes/banana_bowl.yaml"
