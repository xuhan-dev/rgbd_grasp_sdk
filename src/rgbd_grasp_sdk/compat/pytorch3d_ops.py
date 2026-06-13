from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import site
import sys
import types
from pathlib import Path
from typing import Any

import torch

from rgbd_grasp_sdk.errors import BackendUnavailableError


def install_rng_compat_fallbacks(
    project_root: Path | None = None,
    *,
    allow_noop_grasp_nms: bool = True,
) -> None:
    install_mpl_toolkits_namespace_fix()
    install_pytorch3d_fallbacks()
    install_grasp_nms_fallback(
        project_root=project_root or _default_project_root(),
        allow_noop=allow_noop_grasp_nms,
    )


def install_mpl_toolkits_namespace_fix(user_site: Path | None = None) -> None:
    user_mpl_toolkits = (user_site or Path(site.getusersitepackages())) / "mpl_toolkits"
    if not (user_mpl_toolkits / "mplot3d").exists():
        return

    module = types.ModuleType("mpl_toolkits")
    module.__path__ = [str(user_mpl_toolkits)]
    module.__package__ = "mpl_toolkits"
    module.__spec__ = ModuleSpec("mpl_toolkits", loader=None, is_package=True)
    sys.modules["mpl_toolkits"] = module


def install_pytorch3d_fallbacks() -> None:
    if _module_spec_exists("pytorch3d"):
        return

    pytorch3d_module = types.ModuleType("pytorch3d")
    ops_module = types.ModuleType("pytorch3d.ops")
    ops_utils_module = types.ModuleType("pytorch3d.ops.utils")
    transforms_module = types.ModuleType("pytorch3d.transforms")
    pytorch3d_module.__spec__ = ModuleSpec("pytorch3d", loader=None, is_package=True)
    ops_module.__spec__ = ModuleSpec("pytorch3d.ops", loader=None, is_package=True)
    ops_utils_module.__spec__ = ModuleSpec("pytorch3d.ops.utils", loader=None)
    transforms_module.__spec__ = ModuleSpec("pytorch3d.transforms", loader=None)

    ops_module.sample_farthest_points = sample_farthest_points
    ops_module.ball_query = ball_query
    ops_module.knn_points = knn_points
    ops_utils_module.masked_gather = masked_gather
    transforms_module.euler_angles_to_matrix = euler_angles_to_matrix
    transforms_module.matrix_to_quaternion = matrix_to_quaternion

    pytorch3d_module.ops = ops_module
    pytorch3d_module.transforms = transforms_module
    ops_module.utils = ops_utils_module

    sys.modules["pytorch3d"] = pytorch3d_module
    sys.modules["pytorch3d.ops"] = ops_module
    sys.modules["pytorch3d.ops.utils"] = ops_utils_module
    sys.modules["pytorch3d.transforms"] = transforms_module


def install_grasp_nms_fallback(project_root: Path, *, allow_noop: bool = True) -> None:
    if _module_spec_exists("grasp_nms"):
        return

    grasp_nms_path = project_root / "third_party" / "grasp_nms" / "grasp_nms.py"
    if not grasp_nms_path.exists():
        if allow_noop:
            _install_noop_grasp_nms()
            return
        raise BackendUnavailableError(f"无法找到 grasp_nms fallback: {grasp_nms_path}")

    spec = importlib.util.spec_from_file_location("grasp_nms", grasp_nms_path)
    if spec is None or spec.loader is None:
        raise BackendUnavailableError(f"无法加载 grasp_nms fallback: {grasp_nms_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["grasp_nms"] = module
    spec.loader.exec_module(module)


def sample_farthest_points(
    points: torch.Tensor,
    lengths: torch.Tensor | None = None,
    K: int = 50,
    random_start_point: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, point_count, channels = points.shape
    device = points.device
    idxs = torch.zeros((batch_size, K), dtype=torch.long, device=device)
    sampled = torch.zeros((batch_size, K, channels), dtype=points.dtype, device=device)

    for batch in range(batch_size):
        valid_count = int(lengths[batch].item()) if lengths is not None else point_count
        valid_count = max(1, min(valid_count, point_count))
        current = (
            int(torch.randint(valid_count, (1,), device=device).item())
            if random_start_point
            else 0
        )
        distances = torch.full((valid_count,), float("inf"), dtype=points.dtype, device=device)
        for index in range(K):
            idxs[batch, index] = current
            sampled[batch, index] = points[batch, current]
            distance = torch.sum(
                (points[batch, :valid_count] - points[batch, current]) ** 2,
                dim=-1,
            )
            distances = torch.minimum(distances, distance)
            current = int(torch.argmax(distances).item())

    return sampled, idxs


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int = 50,
    radius: float = 0.2,
    return_nn: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    distances = torch.cdist(p1, p2)
    masked = distances.masked_fill(distances > radius, float("inf"))
    _, idxs = torch.topk(masked, k=K, dim=-1, largest=False)
    idxs = idxs.masked_fill(torch.isinf(torch.gather(masked, -1, idxs)), -1)
    dists = torch.gather(distances, -1, idxs.clamp_min(0))
    if return_nn:
        return dists, idxs, _gather_neighbors(p2, idxs)
    return dists, idxs, None


def knn_points(
    p1: torch.Tensor,
    p2: torch.Tensor,
    K: int = 1,
    return_nn: bool = False,
    **_kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    distances = torch.cdist(p1, p2)
    dists, idxs = torch.topk(distances, k=K, dim=-1, largest=False)
    if return_nn:
        return dists, idxs, _gather_neighbors(p2, idxs)
    return dists, idxs, None


def masked_gather(points: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    gathered = _gather_neighbors(points, idxs)
    return gathered.masked_fill(idxs[..., None] < 0, 0)


def euler_angles_to_matrix(
    euler_angles: torch.Tensor,
    convention: str = "XYZ",
) -> torch.Tensor:
    if convention != "XYZ":
        raise NotImplementedError("pytorch3d fallback only supports XYZ convention")
    x, y, z = euler_angles.unbind(-1)
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
        dim=-1,
    ).reshape(euler_angles.shape[:-1] + (3, 3))
    ry = torch.stack(
        [cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
        dim=-1,
    ).reshape(euler_angles.shape[:-1] + (3, 3))
    rz = torch.stack(
        [cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
        dim=-1,
    ).reshape(euler_angles.shape[:-1] + (3, 3))
    return rx @ ry @ rz


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    qw = torch.sqrt(
        torch.clamp(1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2], min=0)
    ) / 2
    qx = torch.sqrt(
        torch.clamp(1 + matrix[..., 0, 0] - matrix[..., 1, 1] - matrix[..., 2, 2], min=0)
    ) / 2
    qy = torch.sqrt(
        torch.clamp(1 - matrix[..., 0, 0] + matrix[..., 1, 1] - matrix[..., 2, 2], min=0)
    ) / 2
    qz = torch.sqrt(
        torch.clamp(1 - matrix[..., 0, 0] - matrix[..., 1, 1] + matrix[..., 2, 2], min=0)
    ) / 2
    qx = torch.copysign(qx, matrix[..., 2, 1] - matrix[..., 1, 2])
    qy = torch.copysign(qy, matrix[..., 0, 2] - matrix[..., 2, 0])
    qz = torch.copysign(qz, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack([qw, qx, qy, qz], dim=-1)


def _gather_neighbors(points: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
    safe_idxs = idxs.clamp_min(0)
    return torch.gather(
        points[:, None].expand(-1, safe_idxs.shape[1], -1, -1),
        2,
        safe_idxs[..., None].expand(-1, -1, -1, points.shape[-1]),
    )


def _install_noop_grasp_nms() -> None:
    module = types.ModuleType("grasp_nms")
    module.__spec__ = ModuleSpec("grasp_nms", loader=None)
    module.nms_grasp = lambda grasp_group_array, translation_thresh, rotation_thresh: grasp_group_array
    sys.modules["grasp_nms"] = module


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _module_spec_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return name in sys.modules
