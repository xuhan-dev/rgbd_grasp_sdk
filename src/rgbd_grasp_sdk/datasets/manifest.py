from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import CameraIntrinsics


@dataclass(frozen=True)
class GraspSample:
    id: str | None
    rgb: Any
    depth: Any
    intrinsics: Any
    target: str


def load_samples(path: str | Path) -> list[GraspSample]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise InputValidationError(f"manifest 文件不存在: {manifest_path}")

    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        raw = json.loads(text)
    else:
        raw = yaml.safe_load(text)

    if not isinstance(raw, list):
        raise InputValidationError("manifest 根节点必须是样本列表")

    return normalize_samples(raw, base_dir=manifest_path.parent)


def normalize_samples(
    source: Any = None,
    *,
    rgb: Any = None,
    depth: Any = None,
    intrinsics: Any = None,
    target: str | None = None,
    base_dir: str | Path | None = None,
) -> list[GraspSample]:
    keyword_values = (rgb, depth, intrinsics, target)
    has_keyword_input = any(value is not None for value in keyword_values)
    if source is not None and has_keyword_input:
        raise InputValidationError("source 与 rgb/depth/intrinsics/target 不能同时使用")

    if source is None:
        if not has_keyword_input:
            raise InputValidationError("必须提供 source 或单帧 rgb/depth/intrinsics/target")
        raw_samples = [
            {
                "rgb": rgb,
                "depth": depth,
                "intrinsics": intrinsics,
                "target": target,
            }
        ]
    elif isinstance(source, dict):
        raw_samples = [source]
    elif isinstance(source, list):
        raw_samples = source
    else:
        raise InputValidationError("source 必须是样本 dict 或样本 list")

    if not isinstance(raw_samples, list):
        raise InputValidationError("manifest 根节点必须是样本列表")

    root = Path(base_dir) if base_dir is not None else None
    return [_sample_from_mapping(index, item, root) for index, item in enumerate(raw_samples)]


def _sample_from_mapping(
    index: int,
    item: Any,
    base_dir: Path | None,
) -> GraspSample:
    if not isinstance(item, dict):
        raise InputValidationError(f"sample[{index}] 必须是对象")

    rgb = _required(item, index, "rgb")
    depth = _required(item, index, "depth")
    intrinsics = _required(item, index, "intrinsics")
    target = _required(item, index, "target")
    if not isinstance(target, str) or not target:
        raise InputValidationError(f"sample[{index}].target 必须是非空字符串")

    sample_id = item.get("id")
    if sample_id is not None and not isinstance(sample_id, str):
        raise InputValidationError(f"sample[{index}].id 必须是字符串")

    return GraspSample(
        id=sample_id,
        rgb=_resolve_path_like(rgb, base_dir),
        depth=_resolve_path_like(depth, base_dir),
        intrinsics=_resolve_path_like(intrinsics, base_dir),
        target=target,
    )


def _required(item: dict[str, Any], index: int, key: str) -> Any:
    value = item.get(key)
    if value is None:
        raise InputValidationError(f"sample[{index}].{key} 缺失")
    return value


def _resolve_path_like(value: Any, base_dir: Path | None) -> Any:
    if base_dir is None:
        return value
    if isinstance(value, str):
        path = Path(value)
        return path if path.is_absolute() else base_dir / path
    if isinstance(value, Path):
        return value if value.is_absolute() else base_dir / value
    return value
