import importlib.util
from pathlib import Path

import cv2
import numpy as np
import yaml


def _load_prepare_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_graspnet_demo.py"
    spec = importlib.util.spec_from_file_location("prepare_graspnet_demo", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_graspnet_demo_sample_copies_frame_and_intrinsics(tmp_path):
    module = _load_prepare_module()
    root = tmp_path / "graspnet"
    scene = root / "scenes" / "scene_0000" / "realsense"
    rgb_dir = scene / "rgb"
    depth_dir = scene / "depth"
    meta_dir = scene / "meta"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()
    meta_dir.mkdir()
    cv2.imwrite(str(rgb_dir / "0000.png"), np.zeros((4, 5, 3), dtype=np.uint8))
    cv2.imwrite(str(depth_dir / "0000.png"), np.ones((4, 5), dtype=np.uint16))
    intrinsic_matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 601.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savez(meta_dir / "0000.npz", intrinsic_matrix=intrinsic_matrix)

    output_dir = tmp_path / "demo"

    module.prepare_sample(
        root=root,
        output_dir=output_dir,
        scene_id=0,
        frame_id=0,
        camera="realsense",
        target="banana",
    )

    assert (output_dir / "rgb.png").exists()
    assert (output_dir / "depth.png").exists()
    assert np.allclose(np.load(output_dir / "camera_intrinsics.npz")["K"], intrinsic_matrix)
    metadata = yaml.safe_load((output_dir / "metadata.yaml").read_text(encoding="utf-8"))
    assert metadata["dataset"] == "GraspNet-1Billion"
    assert metadata["target"] == "banana"
    assert metadata["scene_id"] == 0
    assert metadata["frame_id"] == 0


def test_prepare_graspnet_demo_sample_supports_official_camk_npy(tmp_path):
    module = _load_prepare_module()
    root = tmp_path / "graspnet"
    scene = root / "scenes" / "scene_0075" / "realsense"
    rgb_dir = scene / "rgb"
    depth_dir = scene / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()
    cv2.imwrite(str(rgb_dir / "0003.png"), np.zeros((4, 5, 3), dtype=np.uint8))
    cv2.imwrite(str(depth_dir / "0003.png"), np.ones((4, 5), dtype=np.uint16))
    intrinsic_matrix = np.array(
        [[927.17, 0.0, 651.32], [0.0, 927.37, 349.62], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.save(scene / "camK.npy", intrinsic_matrix)

    output_dir = tmp_path / "demo"

    module.prepare_sample(
        root=root,
        output_dir=output_dir,
        scene_id=75,
        frame_id=3,
        camera="realsense",
        target="apple",
    )

    assert np.allclose(np.load(output_dir / "camera_intrinsics.npz")["K"], intrinsic_matrix)
    metadata = yaml.safe_load((output_dir / "metadata.yaml").read_text(encoding="utf-8"))
    assert metadata["scene_id"] == 75
    assert metadata["frame_id"] == 3
