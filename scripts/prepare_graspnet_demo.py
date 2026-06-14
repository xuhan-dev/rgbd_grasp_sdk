#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从本地 GraspNet-1Billion 数据抽取演示帧")
    parser.add_argument("--root", required=True, help="GraspNet 数据根目录")
    parser.add_argument("--output-dir", default="data/demo/graspnet_sample", help="输出目录")
    parser.add_argument("--scene-id", type=int, default=0, help="scene 编号")
    parser.add_argument("--frame-id", type=int, default=0, help="frame 编号")
    parser.add_argument("--camera", default="realsense", help="相机目录名，例如 realsense 或 kinect")
    parser.add_argument("--target", default="", help="演示目标类别或文本描述")
    return parser.parse_args(argv)


def prepare_sample(
    *,
    root: str | Path,
    output_dir: str | Path,
    scene_id: int,
    frame_id: int,
    camera: str,
    target: str,
) -> None:
    root_path = Path(root)
    output_path = Path(output_dir)
    scene_path = root_path / "scenes" / f"scene_{scene_id:04d}" / camera
    rgb_path = scene_path / "rgb" / f"{frame_id:04d}.png"
    depth_path = scene_path / "depth" / f"{frame_id:04d}.png"
    meta_path = scene_path / "meta" / f"{frame_id:04d}.npz"
    camk_path = scene_path / "camK.npy"

    _require_file(rgb_path)
    _require_file(depth_path)

    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rgb_path, output_path / "rgb.png")
    shutil.copy2(depth_path, output_path / "depth.png")
    np.savez(
        output_path / "camera_intrinsics.npz",
        K=_load_intrinsic_matrix(meta_path, camk_path),
    )
    metadata = {
        "dataset": "GraspNet-1Billion",
        "source_root": str(root_path),
        "scene_id": scene_id,
        "frame_id": frame_id,
        "camera": camera,
        "target": target,
        "rgb": "rgb.png",
        "depth": "depth.png",
        "intrinsics": "camera_intrinsics.npz",
    }
    (output_path / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _load_intrinsic_matrix(meta_path: Path, camk_path: Path) -> np.ndarray:
    if not meta_path.exists():
        return _load_camk_matrix(camk_path)
    data = np.load(meta_path)
    for key in ("intrinsic_matrix", "K", "camera_intrinsic"):
        if key in data:
            matrix = np.asarray(data[key], dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError(f"内参矩阵必须是 3x3: {meta_path}:{key}")
            return matrix
    if camk_path.exists():
        return _load_camk_matrix(camk_path)
    raise ValueError(f"GraspNet meta 文件缺少 intrinsic_matrix/K/camera_intrinsic: {meta_path}")


def _load_camk_matrix(camk_path: Path) -> np.ndarray:
    _require_file(camk_path)
    matrix = np.asarray(np.load(camk_path), dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"camK.npy 必须是 3x3 矩阵: {camk_path}")
    return matrix


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")


def main() -> None:
    args = parse_args()
    prepare_sample(
        root=args.root,
        output_dir=args.output_dir,
        scene_id=args.scene_id,
        frame_id=args.frame_id,
        camera=args.camera,
        target=args.target,
    )
    print(f"prepared demo sample: {args.output_dir}")


if __name__ == "__main__":
    main()
