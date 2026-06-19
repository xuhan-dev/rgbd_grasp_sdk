from __future__ import annotations

import json

import numpy as np
import pytest

from rgbd_grasp_sdk.datasets import GraspSample, load_samples, normalize_samples
from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.types import CameraIntrinsics


def test_normalize_single_keyword_sample_preserves_arrays():
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    depth = np.ones((2, 3), dtype=np.uint16)
    intrinsics = CameraIntrinsics(fx=1.0, fy=2.0, cx=3.0, cy=4.0)

    samples = normalize_samples(
        source=None,
        rgb=rgb,
        depth=depth,
        intrinsics=intrinsics,
        target="apple",
    )

    assert samples == [
        GraspSample(
            id=None,
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            target="apple",
        )
    ]


def test_normalize_source_list_keeps_sample_ids():
    samples = normalize_samples(
        source=[
            {
                "id": "a",
                "rgb": "a/rgb.png",
                "depth": "a/depth.png",
                "intrinsics": "a/K.npz",
                "target": "apple",
            }
        ]
    )

    assert len(samples) == 1
    assert samples[0].id == "a"
    assert str(samples[0].rgb) == "a/rgb.png"
    assert str(samples[0].depth) == "a/depth.png"
    assert str(samples[0].intrinsics) == "a/K.npz"
    assert samples[0].target == "apple"


def test_load_yaml_manifest_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
- id: sample-a
  rgb: images/rgb.png
  depth: images/depth.png
  intrinsics: camera/K.npz
  target: apple
""",
        encoding="utf-8",
    )

    samples = load_samples(manifest)

    assert samples[0].id == "sample-a"
    assert samples[0].rgb == tmp_path / "images/rgb.png"
    assert samples[0].depth == tmp_path / "images/depth.png"
    assert samples[0].intrinsics == tmp_path / "camera/K.npz"
    assert samples[0].target == "apple"


def test_load_json_manifest_resolves_relative_paths(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "rgb": "rgb.png",
                    "depth": "depth.png",
                    "intrinsics": "K.npz",
                    "target": "banana",
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = load_samples(manifest)

    assert samples[0].rgb == tmp_path / "rgb.png"
    assert samples[0].depth == tmp_path / "depth.png"
    assert samples[0].intrinsics == tmp_path / "K.npz"
    assert samples[0].target == "banana"


def test_missing_required_sample_field_raises_clear_error():
    with pytest.raises(InputValidationError, match="sample\\[0\\].depth"):
        normalize_samples(
            source=[
                {
                    "rgb": "rgb.png",
                    "intrinsics": "K.npz",
                    "target": "apple",
                }
            ]
        )


def test_source_and_keyword_inputs_are_mutually_exclusive():
    with pytest.raises(InputValidationError, match="source.*不能同时"):
        normalize_samples(
            source=[],
            rgb="rgb.png",
            depth="depth.png",
            intrinsics="K.npz",
            target="apple",
        )
