"""Scene specification data structures and YAML loaders.

Framework-independent scene description for cross-layer use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ObjectPlacement:
    """物体摆放定义"""

    object_id: str
    object_type: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    usd_path: str | None = None
    physics: dict[str, float] = field(default_factory=dict)
    semantic_label: str | None = None


@dataclass
class SceneSpec:
    """场景描述规范（版本化）"""

    spec_version: str
    scene_id: str
    description: str
    base_stage_usd: str
    robot_spawn_pose: tuple[tuple[float, float, float], tuple[float, float, float, float]]
    objects: list[ObjectPlacement]
    lighting: dict[str, Any] | None = None
    randomization_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典"""
        return {
            "spec_version": self.spec_version,
            "scene_id": self.scene_id,
            "description": self.description,
            "base_stage_usd": self.base_stage_usd,
            "robot_spawn_pose": {
                "position": list(self.robot_spawn_pose[0]),
                "rotation": list(self.robot_spawn_pose[1]),
            },
            "objects": [
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "position": list(obj.position),
                    "rotation": list(obj.rotation),
                    "scale": list(obj.scale),
                    "usd_path": obj.usd_path,
                    "physics": obj.physics,
                    "semantic_label": obj.semantic_label,
                }
                for obj in self.objects
            ],
            "lighting": self.lighting,
            "randomization_config": self.randomization_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSpec:
        """从字典反序列化"""
        # 验证scene_id非空
        scene_id = data["scene_id"]
        if not scene_id or not scene_id.strip():
            raise ValueError("scene_id不能为空")

        # 验证robot_spawn_pose维度
        robot_pose_data = data["robot_spawn_pose"]
        position = robot_pose_data["position"]
        rotation = robot_pose_data["rotation"]

        if len(position) != 3:
            raise ValueError(f"robot_spawn_pose.position必须是3D坐标，实际{len(position)}维")
        if len(rotation) != 4:
            raise ValueError(f"robot_spawn_pose.rotation必须是四元数(x,y,z,w)，实际{len(rotation)}维")

        robot_spawn_pose = (tuple(position), tuple(rotation))

        # 解析objects
        objects = []
        seen_ids = set()

        for obj in data.get("objects", []):
            obj_id = obj["object_id"]

            # 验证object_id非空
            if not obj_id or not obj_id.strip():
                raise ValueError("object_id不能为空")

            # 验证object_id唯一
            if obj_id in seen_ids:
                raise ValueError(f"发现重复的object_id: {obj_id}")
            seen_ids.add(obj_id)

            objects.append(
                ObjectPlacement(
                    object_id=obj_id,
                    object_type=obj["object_type"],
                    position=tuple(obj["position"]),
                    rotation=tuple(obj["rotation"]),
                    scale=tuple(obj.get("scale", [1.0, 1.0, 1.0])),
                    usd_path=obj.get("usd_path"),
                    physics=obj.get("physics", {}),
                    semantic_label=obj.get("semantic_label"),
                )
            )

        return cls(
            spec_version=data["spec_version"],
            scene_id=scene_id,
            description=data["description"],
            base_stage_usd=data["base_stage_usd"],
            robot_spawn_pose=robot_spawn_pose,
            objects=objects,
            lighting=data.get("lighting"),
            randomization_config=data.get("randomization_config"),
            metadata=data.get("metadata", {}),
        )

    def save_yaml(self, path: Path | str) -> None:
        """保存为YAML文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)


def load_scene_spec(path: Path | str) -> SceneSpec:
    """从YAML文件加载SceneSpec"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SceneSpec.from_dict(data)
