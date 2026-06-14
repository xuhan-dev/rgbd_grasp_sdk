from pathlib import Path

import pytest

from rgbd_grasp_sdk.config.loader import load_config
from rgbd_grasp_sdk.errors import ConfigError


def test_load_config_reads_yaml(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
segmentation:
  backend: yolo
grasping:
  backend: rng
mask:
  merge_instances: true
ranking:
  backend: default
outputs:
  return_point_cloud: false
  visualize_3d: true
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.segmentation.backend == "yolo"
    assert config.grasping.backend == "rng"
    assert config.outputs.return_point_cloud is False
    assert config.outputs.visualize_3d is True


def test_load_mask_aware_ranking_options(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
segmentation:
  backend: yolo
  model_path: weights.pt
grasping:
  backend: rng
  checkpoint_path: checkpoints/rng
mask:
  merge_instances: true
ranking:
  backend: mask_aware
  top_k: 5
  min_target_score: 0.25
  require_center_in_mask: true
  weights:
    rng_score: 0.7
    target_score: 0.3
outputs:
  visualize_3d: false
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.ranking.backend == "mask_aware"
    assert config.ranking.top_k == 5
    assert config.ranking.min_target_score == 0.25
    assert config.ranking.require_center_in_mask is True
    assert config.ranking.weights == {"rng_score": 0.7, "target_score": 0.3}


def test_load_config_rejects_missing_required_section(tmp_path: Path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("segmentation:\n  backend: yolo\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="缺少配置段"):
        load_config(config_file)
