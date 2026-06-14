#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class Check:
    name: str
    run: Callable[[], str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 RGB-D Grasp SDK 运行时依赖")
    parser.add_argument(
        "--profile",
        default="base",
        choices=("base", "dev", "yolo", "fastsam", "rng-cu121"),
        help="依赖检查 profile",
    )
    parser.add_argument("--require-cuda", action="store_true", help="要求 torch CUDA 可用")
    parser.add_argument("--yolo-weights", help="可选 YOLO 权重路径")
    parser.add_argument("--rng-checkpoint", help="可选 RNG checkpoint 路径")
    parser.add_argument("--intrinsics", help="可选 camera_intrinsics.npz 路径")
    return parser.parse_args(argv)


def build_checks(profile: str, *, require_cuda: bool = False) -> list[Check]:
    checks = [
        python_check(),
        import_check("numpy"),
        import_check("cv2"),
        import_check("yaml"),
        import_check("rgbd_grasp_sdk"),
    ]
    if profile == "dev":
        checks.append(import_check("pytest"))
    if profile == "yolo":
        checks.append(import_check("ultralytics"))
    if profile == "fastsam":
        checks.extend([import_check("torch"), import_check("torchvision")])
    if profile == "rng-cu121":
        checks.extend(
            [
                import_check("torch"),
                torch_cuda_check(required=True),
                import_check("torchvision"),
                import_check("pytorch3d"),
                import_check("grasp_nms"),
                import_check("open3d"),
                import_check("transforms3d"),
                import_check("trimesh"),
            ]
        )
    elif require_cuda:
        checks.append(torch_cuda_check(required=True))
    return checks


def python_check() -> Check:
    def run() -> str:
        if sys.version_info < (3, 10):
            raise RuntimeError("Python >= 3.10 is required")
        return sys.version.split()[0]

    return Check("python", run)


def import_check(module_name: str) -> Check:
    def run() -> str:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "imported"))

    return Check(module_name, run)


def torch_cuda_check(*, required: bool) -> Check:
    def run() -> str:
        torch = importlib.import_module("torch")
        available = bool(torch.cuda.is_available())
        if required and not available:
            raise RuntimeError("CUDA is not available")
        return str(available)

    return Check("torch_cuda", run)


def path_exists_check(name: str, path: str | Path) -> Check:
    check_path = Path(path)

    def run() -> str:
        if not check_path.exists():
            raise FileNotFoundError(f"missing: {check_path}")
        return str(check_path)

    return Check(name, run)


def run_checks(checks: Sequence[Check]) -> list[CheckResult]:
    results = []
    for check in checks:
        try:
            detail = check.run()
        except Exception as exc:
            results.append(CheckResult(check.name, False, str(exc)))
        else:
            results.append(CheckResult(check.name, True, detail))
    return results


def main() -> None:
    args = parse_args()
    checks = build_checks(args.profile, require_cuda=args.require_cuda)
    if args.yolo_weights:
        checks.append(path_exists_check("yolo_weights", args.yolo_weights))
    if args.rng_checkpoint:
        checks.append(path_exists_check("rng_checkpoint", args.rng_checkpoint))
    if args.intrinsics:
        checks.append(path_exists_check("intrinsics", args.intrinsics))

    results = run_checks(checks)
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"{status:4} {result.name}: {result.detail}")

    if not all(result.ok for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
