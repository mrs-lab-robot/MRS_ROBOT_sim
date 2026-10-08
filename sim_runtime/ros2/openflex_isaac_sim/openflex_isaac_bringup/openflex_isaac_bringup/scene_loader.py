"""Load versioned SceneSpec content into an owned, replaceable USD namespace."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from openflex_isaac_contract.scene_spec import load_scene_spec


class SceneLoader:
    """Stage scenes under a dedicated namespace and swap only after a full load.

    The active scene lives below ``/World/MRSRobotScene``. New content is built
    under a separate staging namespace, so invalid references or transforms do
    not remove the currently active scene or unrelated world prims.
    """

    WORLD_PATH = "/World"
    ACTIVE_ROOT = "/World/MRSRobotScene"
    STAGING_ROOT = "/World/__MRSRobotSceneStaging"
    BACKUP_ROOT = "/World/__MRSRobotSceneBackup"
    OBJECTS_ROOT_NAME = "objects"

    def __init__(self, stage: Any):
        self.stage = stage

    def _usd(self):
        """Import USD bindings lazily so the contract package stays standalone."""
        from pxr import Gf, Sdf, UsdGeom

        return Gf, Sdf, UsdGeom

    def _is_valid(self, path: str) -> bool:
        prim = self.stage.GetPrimAtPath(path)
        return bool(prim and prim.IsValid())

    def _remove_owned_root(self, path: str) -> bool:
        if not self._is_valid(path):
            return False
        self.stage.RemovePrim(path)
        return True

    def clear_scene_objects(self, preserve_prims: list[str] | None = None) -> list[str]:
        """Remove this loader's reserved roots, preserving all external prims.

        ``preserve_prims`` remains accepted for compatibility with early callers;
        ownership is determined by the reserved namespace, never by deleting
        arbitrary children of ``/World``.
        """
        del preserve_prims
        removed = []
        for path in (self.ACTIVE_ROOT, self.STAGING_ROOT, self.BACKUP_ROOT):
            if self._remove_owned_root(path):
                removed.append(path)
        return removed

    @staticmethod
    def _validate_object_id(object_id: str) -> None:
        # Each placement must be exactly one USD child name, not a path.
        if not isinstance(object_id, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", object_id):
            raise ValueError(f"非法object_id（必须是单个USD标识符）: {object_id!r}")

    def instantiate_object(self, obj_placement, root_path: str | None = None) -> dict[str, Any]:
        """Create one placement while preserving its xyzw quaternion."""
        root_path = root_path or self.ACTIVE_ROOT
        try:
            self._validate_object_id(obj_placement.object_id)
            objects_path = f"{root_path}/{self.OBJECTS_ROOT_NAME}"
            self.stage.DefinePrim(objects_path, "Xform")
            prim_path = f"{objects_path}/{obj_placement.object_id}"
            prim = self.stage.DefinePrim(prim_path, "Xform")
            if not prim or not prim.IsValid():
                return {"success": False, "error": f"无法创建prim: {prim_path}"}

            if obj_placement.usd_path:
                if not prim.GetReferences().AddReference(obj_placement.usd_path):
                    self.stage.RemovePrim(prim_path)
                    return {
                        "success": False,
                        "error": f"无法添加USD引用: {obj_placement.usd_path}",
                    }

            Gf, _, UsdGeom = self._usd()
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(*obj_placement.position))

            qx, qy, qz, qw = obj_placement.rotation
            xformable.AddOrientOp().Set(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
            xformable.AddScaleOp().Set(Gf.Vec3d(*obj_placement.scale))
            return {"success": True, "prim_path": prim_path}
        except Exception as exc:
            if "prim_path" in locals() and self._is_valid(prim_path):
                self.stage.RemovePrim(prim_path)
            return {"success": False, "error": str(exc)}

    def _rename_prim(self, source_path: str, new_name: str) -> None:
        """Rename a loader-authored root in its current USD edit layer."""
        _, Sdf, _ = self._usd()
        edits = Sdf.BatchNamespaceEdit()
        edits.Add(Sdf.NamespaceEdit.Rename(Sdf.Path(source_path), new_name))
        layer = self.stage.GetEditTarget().GetLayer()
        if not layer.Apply(edits):
            raise RuntimeError(f"USD namespace rename失败: {source_path} -> {new_name}")

    def _commit_staging_scene(self) -> bool:
        """Promote the completed staging root, restoring the old root on failure."""
        had_active = self._is_valid(self.ACTIVE_ROOT)
        self._remove_owned_root(self.BACKUP_ROOT)
        if had_active:
            self._rename_prim(self.ACTIVE_ROOT, self.BACKUP_ROOT.rsplit("/", 1)[-1])

        try:
            self._rename_prim(self.STAGING_ROOT, self.ACTIVE_ROOT.rsplit("/", 1)[-1])
        except Exception:
            if had_active and self._is_valid(self.BACKUP_ROOT) and not self._is_valid(self.ACTIVE_ROOT):
                self._rename_prim(self.BACKUP_ROOT, self.ACTIVE_ROOT.rsplit("/", 1)[-1])
            raise

        if had_active:
            self._remove_owned_root(self.BACKUP_ROOT)
        return had_active

    def load_scene(self, spec_path: str | Path) -> dict[str, Any]:
        """Build a SceneSpec in staging and replace only the prior owned scene."""
        try:
            spec = load_scene_spec(spec_path)
            self._remove_owned_root(self.STAGING_ROOT)
            self._remove_owned_root(self.BACKUP_ROOT)

            staging = self.stage.DefinePrim(self.STAGING_ROOT, "Xform")
            if not staging or not staging.IsValid():
                raise RuntimeError(f"无法创建临时场景prim: {self.STAGING_ROOT}")

            environment_path = f"{self.STAGING_ROOT}/Environment"
            environment = self.stage.DefinePrim(environment_path, "Xform")
            if not environment or not environment.IsValid():
                raise RuntimeError(f"无法创建环境引用prim: {environment_path}")
            if not environment.GetReferences().AddReference(spec.base_stage_usd):
                raise RuntimeError(f"无法添加基础场景USD引用: {spec.base_stage_usd}")

            loaded = []
            for obj in spec.objects:
                result = self.instantiate_object(obj, root_path=self.STAGING_ROOT)
                if not result["success"]:
                    raise _SceneObjectLoadError(obj.object_id, result["error"], loaded)
                loaded.append(obj.object_id)

            replaced = self._commit_staging_scene()
            return {
                "success": True,
                "scene_id": spec.scene_id,
                "loaded": loaded,
                "removed": [self.ACTIVE_ROOT] if replaced else [],
                "scene_root": self.ACTIVE_ROOT,
                "robot_spawn_pose": spec.robot_spawn_pose,
            }
        except _SceneObjectLoadError as exc:
            self._remove_owned_root(self.STAGING_ROOT)
            return {
                "success": False,
                "error": str(exc),
                "failed_object": exc.object_id,
                "rolled_back": exc.loaded,
            }
        except FileNotFoundError as exc:
            self._remove_owned_root(self.STAGING_ROOT)
            return {"success": False, "error": f"SceneSpec文件不存在: {exc}"}
        except ValueError as exc:
            self._remove_owned_root(self.STAGING_ROOT)
            return {"success": False, "error": f"SceneSpec验证失败: {exc}"}
        except Exception as exc:
            self._remove_owned_root(self.STAGING_ROOT)
            return {"success": False, "error": f"场景加载异常: {exc}"}


class _SceneObjectLoadError(RuntimeError):
    def __init__(self, object_id: str, message: str, loaded: list[str]):
        super().__init__(message)
        self.object_id = object_id
        self.loaded = list(loaded)
