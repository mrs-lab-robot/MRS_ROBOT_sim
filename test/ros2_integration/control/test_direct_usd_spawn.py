#!/usr/bin/env python3
"""Regression tests that the Isaac robot is referenced from the canonical USD."""

from pathlib import Path
import ast
import os
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
BRINGUP = ROOT / "sim_runtime" / "ros2" / "openflex_isaac_sim" / "openflex_isaac_bringup"
START_SCRIPT = BRINGUP / "scripts" / "start_robot_control_sim.py"
SPAWN_SCRIPT = BRINGUP / "scripts" / "spawn_robot_when_ready.py"
SIM_LAUNCH = BRINGUP / "launch" / "sim.launch.py"


class FakeAttribute:
    def __init__(self) -> None:
        self.value = None

    def Set(self, value) -> None:
        self.value = value


class FakeReferences:
    def __init__(self) -> None:
        self.asset_path = None

    def AddReference(self, asset_path: str) -> None:
        self.asset_path = asset_path


class FakePrim:
    def __init__(self) -> None:
        self.references = FakeReferences()
        self.authored_translation = (0.0, 0.0, 0.175)
        self.translate = FakeAttribute()
        self.rotate_xyz = FakeAttribute()

    def GetReferences(self) -> FakeReferences:
        return self.references


class FakeXformable:
    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def GetOrderedXformOps(self):
        return [FakeTranslateOp(self.prim.authored_translation)]


class FakeResetTransformOp:
    def __init__(self) -> None:
        self.value = None

    def Set(self, value) -> None:
        self.value = value


class FakeResetXformable:
    def __init__(self) -> None:
        self.calls = []
        self.transform_op = FakeResetTransformOp()
        self.reset_xform_stack = False

    def ClearXformOpOrder(self) -> None:
        self.calls.append(("clear",))

    def AddTransformOp(self, precision, suffix):
        self.calls.append(("add_transform", precision, suffix))
        return self.transform_op

    def SetResetXformStack(self, value: bool) -> None:
        self.calls.append(("reset", value))
        self.reset_xform_stack = value


class FakeTranslateOp:
    def __init__(self, value) -> None:
        self.value = value

    def GetOpType(self):
        return "translate"

    def IsInverseOp(self) -> bool:
        return False

    def Get(self):
        return self.value


class FakeXformCommonAPI:
    RotationOrderXYZ = "XYZ"

    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def SetTranslate(self, value) -> None:
        self.prim.translate.Set(value)

    def SetRotate(self, value, _rotation_order) -> None:
        self.prim.rotate_xyz.Set(value)


class FakeStage:
    def __init__(self) -> None:
        self.prim_path = None
        self.prim_type = None
        self.prim = FakePrim()

    def DefinePrim(self, path: str, prim_type: str) -> FakePrim:
        self.prim_path = path
        self.prim_type = prim_type
        return self.prim


class FakeValidPrim:
    def IsValid(self) -> bool:
        return True


class FakeUsdComponentStage:
    def __init__(self) -> None:
        self.prims = {}
        self.removed_paths = []

    def GetPrimAtPath(self, path: str):
        return self.prims.get(path)

    def RemovePrim(self, path: str) -> None:
        self.removed_paths.append(path)
        self.prims.pop(path, None)


class FakePrimPath:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class FakeEnvironmentPrim:
    def __init__(self, path: str) -> None:
        self.path = path

    def IsValid(self) -> bool:
        return True

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self) -> FakePrimPath:
        return FakePrimPath(self.path)


class FakeWorldPrim:
    def __init__(self, stage) -> None:
        self.stage = stage

    def GetChildren(self):
        return [
            prim for path, prim in self.stage.prims.items()
            if path.startswith("/World/") and path.count("/") == 2
        ]

    def IsValid(self) -> bool:
        return True


class FakeEnvironmentStage:
    def __init__(self, paths) -> None:
        self.prims = {path: FakeEnvironmentPrim(path) for path in paths}
        self.removed_paths = []

    def GetPrimAtPath(self, path: str):
        if path == "/World":
            return FakeWorldPrim(self)
        return self.prims.get(path)

    def RemovePrim(self, path) -> None:
        path = str(path)
        self.removed_paths.append(path)
        self.prims.pop(path, None)


class DirectUsdSpawnTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = runpy.run_path(str(START_SCRIPT))

    def test_interactive_viewport_uses_the_isaac_51_stable_camera_helper(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        viewport_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_set_robot_viewport"
        )
        imported_modules = {
            node.module
            for node in ast.walk(viewport_function)
            if isinstance(node, ast.ImportFrom)
        }

        self.assertIn("isaacsim.core.utils.viewports", imported_modules)
        self.assertNotIn("isaacsim.core.rendering_manager", imported_modules)

    def test_ros_clock_graph_is_created_after_robot_stage_load(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        spawn_function = functions["spawn_robot_control_only"]
        server_wrapper = functions["create_server_with_clock"]
        empty_scene_guard = next(
            node
            for node in ast.walk(server_wrapper)
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Constant)
                and child.value == "OPENFLEX_EMPTY_SCENE"
                for child in ast.walk(node.test)
            )
        )
        spawn_calls_clock_setup = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_ensure_clock_graph"
            for node in ast.walk(spawn_function)
        )
        server_clock_calls = [
            node
            for node in ast.walk(server_wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_ensure_clock_graph"
        ]

        self.assertTrue(spawn_calls_clock_setup)
        self.assertTrue(server_clock_calls)
        self.assertTrue(
            all(
                empty_scene_guard.lineno <= node.lineno <= empty_scene_guard.end_lineno
                for node in server_clock_calls
            ),
            "creating the ROS clock graph before USD stage replacement leaves stale physics callbacks",
        )

    def test_ros_clock_graph_uses_playback_tick_in_simulation_pipeline(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        clock_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_ensure_clock_graph"
        )
        pipeline_stages = {
            node.attr
            for node in ast.walk(clock_function)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("GRAPH_PIPELINE_STAGE_")
        }

        self.assertIn("GRAPH_PIPELINE_STAGE_SIMULATION", pipeline_stages)
        self.assertNotIn("GRAPH_PIPELINE_STAGE_ONDEMAND", pipeline_stages)

    def test_ros_clock_graph_uses_isaac_sim_6_playback_tick_event(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        clock_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_ensure_clock_graph"
        )
        source = ast.unparse(clock_function)

        self.assertIn("omni.graph.action.OnPlaybackTick", source)
        self.assertIn("OnPlaybackTick.outputs:tick", source)
        self.assertNotIn("isaacsim.core.nodes.OnPhysicsStep", source)
        self.assertNotIn("omni.graph.action.OnTick", source)

    def test_isaac_sim_6_control_graph_uses_migrated_joint_state_nodes(self) -> None:
        build_graph = self.runtime.get("_isaac6_joint_control_graph_blueprint")
        self.assertTrue(callable(build_graph))

        create_nodes, connections, values = build_graph(
            "/openflex/Geometry/base_link",
            "/openflex/joint_states",
            "/openflex/joint_command",
        )
        node_types = dict(create_nodes)
        self.assertEqual(node_types["OnPlaybackTick"], "omni.graph.action.OnPlaybackTick")
        self.assertEqual(
            node_types["ReadJointState"],
            "isaacsim.sensors.physics.IsaacReadJointState",
        )
        self.assertEqual(
            node_types["PublishJointState"],
            "isaacsim.ros2.bridge.ROS2PublishJointState",
        )
        self.assertEqual(
            node_types["SubscribeJointState"],
            "isaacsim.ros2.bridge.ROS2SubscribeJointState",
        )
        self.assertNotIn("isaacsim.core.nodes.IsaacArticulationState", node_types.values())
        self.assertNotIn("isaacsim.ros2.bridge.ROS2Publisher", node_types.values())
        self.assertNotIn("isaacsim.ros2.bridge.ROS2Subscriber", node_types.values())
        self.assertIn(
            ("OnPlaybackTick.outputs:tick", "ReadJointState.inputs:execIn"),
            connections,
        )
        self.assertIn(
            ("ReadJointState.outputs:jointPositions", "PublishJointState.inputs:jointPositions"),
            connections,
        )
        self.assertIn(
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            connections,
        )
        configured_values = dict(values)
        self.assertEqual(configured_values["PublishJointState.inputs:topicName"], "/openflex/joint_states")
        self.assertEqual(configured_values["SubscribeJointState.inputs:topicName"], "/openflex/joint_command")
        self.assertEqual(configured_values["ArticulationController.inputs:robotPath"], "/openflex/Geometry/base_link")

    def test_isaac_sim_6_control_graph_replaces_legacy_graph_after_drive_setup(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        spawn_function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "spawn_robot_control_only"
        )
        calls = [
            node.func.attr if isinstance(node.func, ast.Attribute)
            else node.func.id if isinstance(node.func, ast.Name)
            else ""
            for node in ast.walk(spawn_function)
            if isinstance(node, ast.Call)
        ]
        legacy_setup_index = calls.index("main")
        migrated_setup_index = calls.index("_install_isaac6_joint_control_graph")
        self.assertLess(legacy_setup_index, migrated_setup_index)

    def test_startup_does_not_rewrite_the_legacy_robot_control_graph(self) -> None:
        startup = START_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("def _patch_robot_control_graph_pipeline", startup)
        self.assertNotIn("_patch_robot_control_graph_pipeline()", startup)

    def test_robot_usd_is_referenced_at_ros_robot_root_with_spawn_pose(self) -> None:
        reference_robot = self.runtime.get("_reference_robot_usd")
        self.assertTrue(
            callable(reference_robot),
            "Isaac startup must reference the canonical robot USD directly",
        )
        stage = FakeStage()
        class FakeUsdGeom:
            XformOp = types.SimpleNamespace(TypeTranslate="translate")
            Xformable = FakeXformable
            XformCommonAPI = FakeXformCommonAPI

        pxr = types.ModuleType("pxr")
        pxr.Gf = types.SimpleNamespace(
            Vec3d=lambda *values: tuple(values),
            Vec3f=lambda *values: tuple(values),
        )
        pxr.UsdGeom = FakeUsdGeom

        with tempfile.TemporaryDirectory() as temporary_directory:
            usd_path = Path(temporary_directory) / "openflex_robot.usda"
            usd_path.write_text("#usda 1.0\n", encoding="utf-8")
            with patch.dict(sys.modules, {"pxr": pxr}):
                prim_path = reference_robot(
                    stage,
                    str(usd_path),
                    x=1.0,
                    y=2.0,
                    z=3.0,
                    roll=0.0,
                    pitch=0.5,
                    yaw=-1.0,
                )

        self.assertEqual(prim_path, "/openflex")
        self.assertEqual(stage.prim_path, "/openflex")
        self.assertEqual(stage.prim_type, "Xform")
        self.assertEqual(stage.prim.references.asset_path, str(usd_path.resolve()))
        self.assertEqual(stage.prim.translate.value, (1.0, 2.0, 3.175))
        self.assertAlmostEqual(stage.prim.rotate_xyz.value[0], 0.0)
        self.assertAlmostEqual(stage.prim.rotate_xyz.value[1], 28.6478897565)
        self.assertAlmostEqual(stage.prim.rotate_xyz.value[2], -57.2957795131)

    def test_direct_usd_restores_lift_drive_profile_from_control_urdf(self) -> None:
        """Direct USD must retain the lift hold behavior of the old importer path."""
        apply_drive_profiles = self.runtime.get("_apply_direct_usd_drive_profiles")
        self.assertTrue(
            callable(apply_drive_profiles),
            "direct USD startup must apply isaac_drive_api profiles before physics playback",
        )

        from pxr import Sdf, Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        lift_joint = stage.DefinePrim("/openflex/Physics/lift_joint", "PhysicsPrismaticJoint")
        lift_drive = UsdPhysics.DriveAPI.Apply(lift_joint, "linear")
        lift_drive.CreateStiffnessAttr().Set(2500.0)
        lift_drive.CreateDampingAttr().Set(300.0)
        lift_drive.CreateMaxForceAttr().Set(500.0)
        lift_joint.CreateAttribute(
            "physxJoint:jointFriction", Sdf.ValueTypeNames.Float
        ).Set(1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            urdf_path = Path(temporary_directory) / "openflex_isaac.urdf"
            urdf_path.write_text(
                """<robot name=\"openflex\">
  <joint name=\"lift_joint\" type=\"prismatic\">
    <isaac_drive_api stiffness=\"100000000.0\" damping=\"500000.0\" max_force=\"3000.0\" joint_friction=\"50.0\"/>
  </joint>
</robot>
""",
                encoding="utf-8",
            )
            applied = apply_drive_profiles(stage, str(urdf_path), robot_root_path="/openflex")

        self.assertEqual(applied, ["lift_joint"])
        self.assertEqual(lift_drive.GetStiffnessAttr().Get(), 100000000.0)
        self.assertEqual(lift_drive.GetDampingAttr().Get(), 500000.0)
        self.assertEqual(lift_drive.GetMaxForceAttr().Get(), 3000.0)
        self.assertEqual(lift_joint.GetAttribute("physxJoint:jointFriction").Get(), 50.0)

    def test_nested_rigid_body_reset_plan_excludes_root_and_deduplicates(self) -> None:
        plan_resets = self.runtime.get("_nested_rigid_body_reset_paths")
        self.assertTrue(
            callable(plan_resets),
            "nested rigid bodies need a runtime XformStack reset to be valid PhysX links",
        )

        paths = plan_resets(
            (
                "/openflex/Geometry/base_link",
                "/openflex/Geometry/base_link/left_link1",
                "/openflex/Geometry/base_link/left_link1",
                "/openflex/Geometry/base_link/right_link1",
            ),
            articulation_root_path="/openflex/Geometry/base_link",
        )

        self.assertEqual(
            paths,
            (
                "/openflex/Geometry/base_link/left_link1",
                "/openflex/Geometry/base_link/right_link1",
            ),
        )

    def test_rigid_body_xform_reset_preserves_cached_world_transform(self) -> None:
        reset_xform = self.runtime.get("_reset_xform_stack_preserving_world_pose")
        self.assertTrue(
            callable(reset_xform),
            "the reset operation must retain the composed link pose from the source USD",
        )
        xformable = FakeResetXformable()
        world_transform = ((1.0, 0.0), (0.0, 1.0))

        reset_xform(xformable, world_transform, precision="double")

        self.assertEqual(
            xformable.calls,
            [
                ("clear",),
                ("add_transform", "double", "openflexWorldPose"),
                ("reset", True),
            ],
        )
        self.assertEqual(xformable.transform_op.value, world_transform)
        self.assertTrue(xformable.reset_xform_stack)

    def test_prim_resync_diagnostics_are_opt_in_and_keep_the_callback_alive(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        install_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_install_control_only_spawn"
        )
        watch_function = next(
            node
            for node in ast.walk(install_function)
            if isinstance(node, ast.FunctionDef)
            and node.name == "install_prim_resync_watch"
        )
        watch_setting = next(
            node
            for node in ast.walk(watch_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "OPENFLEX_PRIM_WATCH"
        )
        retained_callback = next(
            node
            for node in ast.walk(watch_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_PRIM_WATCH_LISTENERS"
        )

        self.assertEqual(watch_setting.args[1].value, "0")
        self.assertIsInstance(retained_callback.args[0], ast.Tuple)
        retained_names = {
            item.id for item in retained_callback.args[0].elts if isinstance(item, ast.Name)
        }
        self.assertIn("listener", retained_names)
        self.assertIn("_on_objects_changed", retained_names)

    def test_reapplying_gui_scene_components_updates_the_existing_usd_prim(self) -> None:
        reapply_component = self.runtime.get("_reapply_usd_component")
        self.assertTrue(
            callable(reapply_component),
            "episode resets must update GUI-managed scene prims instead of stacking references",
        )
        stage = FakeUsdComponentStage()
        additions = []
        updates = []

        def add_usd(params):
            additions.append(dict(params))
            prim = FakeValidPrim()
            stage.prims[f"/World/{params['prim_name']}"] = prim
            return {"success": True, "data": dict(params)}

        def update_usd(prim, params):
            updates.append((prim, dict(params)))
            return {"success": True, "data": dict(params)}

        reapply_component(
            stage, {"prim_name": "MRS_TargetCube", "x": 0.1}, add_usd, update_usd
        )
        first_prim = stage.prims["/World/MRS_TargetCube"]
        reapply_component(
            stage, {"prim_name": "MRS_TargetCube", "x": 0.4}, add_usd, update_usd
        )

        self.assertEqual(stage.removed_paths, [])
        self.assertEqual(len(additions), 1)
        self.assertEqual(len(updates), 1)
        self.assertIs(updates[0][0], first_prim)
        self.assertEqual(updates[0][1]["x"], 0.4)

    def test_reapplying_unmanaged_usd_component_preserves_existing_reference(self) -> None:
        reapply_component = self.runtime.get("_reapply_usd_component")
        self.assertTrue(callable(reapply_component))
        stage = FakeUsdComponentStage()
        stage.prims["/World/UserObject"] = FakeValidPrim()
        additions = []

        updates = []
        reapply_component(
            stage,
            {"prim_name": "UserObject"},
            additions.append,
            lambda prim, params: updates.append((prim, params)),
        )

        self.assertEqual(stage.removed_paths, [])
        self.assertEqual(len(additions), 1)
        self.assertEqual(updates, [])

    def test_existing_scene_prim_pose_uses_xyz_translation_and_radian_input(self) -> None:
        set_pose = self.runtime.get("_set_usd_component_pose")
        self.assertTrue(callable(set_pose))
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest("USD Python bindings are available only inside Isaac Sim")

        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/World/MRS_TargetCube").GetPrim()

        result = set_pose(
            prim,
            {"x": 0.25, "y": -0.1, "z": 0.78, "yaw": 1.5707963267948966},
        )

        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        translate = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        rotate = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ)
        self.assertEqual(result["data"]["prim_path"], "/World/MRS_TargetCube")
        self.assertEqual(tuple(translate.Get()), (0.25, -0.1, 0.78))
        self.assertAlmostEqual(tuple(rotate.Get())[2], 90.0)

    def test_scene_replacement_removes_old_environment_but_preserves_robot_and_base_stage(self) -> None:
        replace_environment = self.runtime.get("_replace_scene_environment")
        self.assertTrue(callable(replace_environment))
        stage = FakeEnvironmentStage(("/World/MRS_Waypoint", "/World/Ground", "/openflex"))

        def add_usd(params):
            path = f"/World/{params['prim_name']}"
            stage.prims[path] = FakeEnvironmentPrim(path)
            return {"success": True, "data": {"prim_path": path}}

        result = replace_environment(
            stage,
            [
                {"prim_name": "MRS_Tabletop", "usd_path": "/sim/table.usda"},
                {"prim_name": "MRS_TargetCube", "usd_path": "/sim/cube.usda"},
            ],
            add_usd,
            lambda _prim, params: {"success": True, "data": params},
        )

        self.assertEqual(stage.removed_paths, ["/World/MRS_Waypoint"])
        self.assertIn("/World/Ground", stage.prims)
        self.assertIn("/openflex", stage.prims)
        self.assertIn("/World/MRS_Tabletop", stage.prims)
        self.assertIn("/World/MRS_TargetCube", stage.prims)
        self.assertEqual(result["data"]["removed"], ["MRS_Waypoint"])

    def test_scene_replacement_keeps_current_environment_when_import_fails(self) -> None:
        replace_environment = self.runtime.get("_replace_scene_environment")
        self.assertTrue(callable(replace_environment))
        stage = FakeEnvironmentStage(("/World/MRS_Waypoint", "/World/Ground"))

        with self.assertRaisesRegex(RuntimeError, "加载 USD 失败"):
            replace_environment(
                stage,
                [{"prim_name": "MRS_Tabletop", "usd_path": "/sim/missing.usda"}],
                lambda _params: {"success": False, "error": "missing asset"},
                lambda _prim, params: {"success": True, "data": params},
            )

        self.assertEqual(stage.removed_paths, [])
        self.assertIn("/World/MRS_Waypoint", stage.prims)

    def test_control_launch_passes_usd_for_sim_and_urdf_for_ros_metadata(self) -> None:
        launch = SIM_LAUNCH.read_text(encoding="utf-8")
        spawn = SPAWN_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"usd_path": str(robot_usd)', launch)
        self.assertIn('"OPENFLEX_ROBOT_USD": str(robot_usd.resolve())', launch)
        self.assertIn('"robot_usd"', launch)
        self.assertIn("_default_robot_usd()", launch)
        self.assertIn("self.usd_path", spawn)
        self.assertIn('"urdf_path": str(output_urdf)', launch)

    def test_isaac_python_resolver_accepts_workstation_and_pip_layouts(self) -> None:
        launch_module = runpy.run_path(str(SIM_LAUNCH))
        resolve_isaac_python = launch_module.get("_resolve_isaac_python")
        self.assertTrue(
            callable(resolve_isaac_python),
            "Isaac launch must resolve both workstation python.sh and pip env bin/python",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            install = Path(temporary_directory)
            workstation_python = install / "python.sh"
            workstation_python.touch()
            self.assertEqual(resolve_isaac_python(install), workstation_python)

            workstation_python.unlink()
            pip_python = install / "bin" / "python"
            pip_python.parent.mkdir()
            pip_python.touch()
            self.assertEqual(resolve_isaac_python(install), pip_python)

    def test_pip_isaac_launch_isolates_ros_python_and_library_environment(self) -> None:
        launch_module = runpy.run_path(str(SIM_LAUNCH))
        process_environment = launch_module.get("_isaac_sim_process_environment")
        self.assertTrue(
            callable(process_environment),
            "pip Isaac launches must isolate Kit's native libraries from sourced ROS overlays",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            install = Path(temporary_directory)
            python = install / "bin" / "python"
            with patch.dict(
                os.environ,
                {
                    "LD_LIBRARY_PATH": "/opt/ros/humble/lib:/openflex_ws/lib",
                    "PYTHONPATH": "/opt/ros/humble/python3.10",
                    "CUDA_HOME": "/usr/local/cuda-13.0",
                },
                clear=False,
            ):
                environment = process_environment(install, python)

        self.assertEqual(
            environment["LD_LIBRARY_PATH"],
            f"{install / 'lib'}:/opt/ros/humble/lib:/openflex_ws/lib",
        )
        self.assertEqual(environment["PYTHONPATH"], "")
        self.assertEqual(environment["CUDA_HOME"], "")

    def test_isaac_spawn_path_does_not_invoke_the_urdf_importer(self) -> None:
        startup = START_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("_reference_robot_usd(", startup)
        self.assertNotIn("import spawn", startup)
        self.assertNotIn("spawn.main(", startup)
        self.assertNotIn("_repair_self_root_joints", startup)
        self.assertNotIn("_apply_urdf_drive_profiles", startup)
        self.assertIn('os.environ.get("OPENFLEX_ROBOT_USD", "")', startup)
        self.assertIn("robot_controller.main(urdf_path=params[\"urdf_path\"])", startup)

    def test_control_graph_uses_the_next_runner_tick_without_nested_kit_updates(self) -> None:
        tree = ast.parse(START_SCRIPT.read_text(encoding="utf-8"))
        install_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_install_control_only_spawn"
        )
        spawn_function = next(
            node
            for node in install_function.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "spawn_robot_control_only"
        )
        controller_call_index = next(
            index
            for index, statement in enumerate(spawn_function.body)
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "main"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "robot_controller"
                for node in ast.walk(statement)
            )
        )
        nested_updates = [
            node
            for statement in spawn_function.body[controller_call_index + 1 :]
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "app"
        ]

        self.assertEqual(
            nested_updates,
            [],
            "Kit must resume its normal runner loop after the ROS control graph is authored",
        )

    def test_spawn_http_timeout_exceeds_the_rest_server_operation_deadline(self) -> None:
        source = SPAWN_SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        request_timeout = next(
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SPAWN_REQUEST_TIMEOUT_SEC"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
        )
        spawn_method = next(
            node
            for top_level in tree.body
            if isinstance(top_level, ast.ClassDef)
            and top_level.name == "SpawnRobotWhenReady"
            for node in top_level.body
            if isinstance(node, ast.FunctionDef) and node.name == "spawn_robot"
        )
        request_timeout_name = next(
            keyword.value.id
            for node in ast.walk(spawn_method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_request_json"
            and any(
                isinstance(argument, ast.Constant)
                and argument.value == "/spawn_robot"
                for argument in node.args
            )
            for keyword in node.keywords
            if keyword.arg == "timeout" and isinstance(keyword.value, ast.Name)
        )

        self.assertEqual(request_timeout_name, "SPAWN_REQUEST_TIMEOUT_SEC")
        self.assertGreaterEqual(
            request_timeout,
            180.0,
            "the installed REST server waits up to 180 seconds for spawn_robot",
        )


if __name__ == "__main__":
    unittest.main()
