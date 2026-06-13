from __future__ import annotations

import sys

import torch

from rgbd_grasp_sdk.compat.pytorch3d_ops import (
    install_mpl_toolkits_namespace_fix,
    install_rng_compat_fallbacks,
)


def test_install_rng_compat_fallbacks_registers_pytorch3d_modules(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "pytorch3d", raising=False)
    monkeypatch.delitem(sys.modules, "pytorch3d.ops", raising=False)
    monkeypatch.delitem(sys.modules, "pytorch3d.ops.utils", raising=False)
    monkeypatch.delitem(sys.modules, "pytorch3d.transforms", raising=False)

    install_rng_compat_fallbacks(project_root=tmp_path, allow_noop_grasp_nms=True)

    from pytorch3d.ops import ball_query, knn_points, sample_farthest_points
    from pytorch3d.ops.utils import masked_gather
    from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion

    points = torch.rand(1, 8, 3)
    sampled, fps_idxs = sample_farthest_points(points, K=4)
    dists, knn_idxs, _ = knn_points(points[:, :2], points, K=3)
    _, ball_idxs, _ = ball_query(points[:, :2], points, K=3, radius=2.0)
    gathered = masked_gather(points, ball_idxs)
    quat = matrix_to_quaternion(
        euler_angles_to_matrix(torch.zeros((1, 3)), convention="XYZ")
    )

    assert sampled.shape == (1, 4, 3)
    assert fps_idxs.shape == (1, 4)
    assert dists.shape == (1, 2, 3)
    assert knn_idxs.shape == (1, 2, 3)
    assert gathered.shape == (1, 2, 3, 3)
    assert quat.shape == (1, 4)


def test_install_rng_compat_fallbacks_loads_third_party_grasp_nms(
    monkeypatch, tmp_path
):
    monkeypatch.delitem(sys.modules, "grasp_nms", raising=False)
    grasp_nms_dir = tmp_path / "third_party" / "grasp_nms"
    grasp_nms_dir.mkdir(parents=True)
    (grasp_nms_dir / "grasp_nms.py").write_text(
        "def nms_grasp(grasp_group_array, translation_thresh, rotation_thresh):\n"
        "    return grasp_group_array[:1]\n",
        encoding="utf-8",
    )

    install_rng_compat_fallbacks(project_root=tmp_path, allow_noop_grasp_nms=False)

    from grasp_nms import nms_grasp

    result = nms_grasp([1, 2, 3], 0.03, 0.5)
    assert result == [1]


def test_install_mpl_toolkits_namespace_fix_prefers_user_site(monkeypatch, tmp_path):
    user_site = tmp_path / "user_site"
    mplot3d_dir = user_site / "mpl_toolkits" / "mplot3d"
    mplot3d_dir.mkdir(parents=True)
    monkeypatch.delitem(sys.modules, "mpl_toolkits", raising=False)

    install_mpl_toolkits_namespace_fix(user_site=user_site)

    import mpl_toolkits

    assert list(mpl_toolkits.__path__) == [str(user_site / "mpl_toolkits")]
