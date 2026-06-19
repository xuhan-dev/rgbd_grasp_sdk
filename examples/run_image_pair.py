from __future__ import annotations

import argparse
from typing import Sequence

from rgbd_grasp_sdk import RGBDGrasp
from rgbd_grasp_sdk.types import PipelineResult


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单帧 RGB-D 抓取预测")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--rgb", required=True, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, help="depth 图像路径")
    parser.add_argument("--intrinsics", required=True, help="包含 K 的相机内参 npz")
    parser.add_argument("--target", required=True, help="目标类别或文本描述")
    parser.add_argument("--output-json", help="可选 JSON 结果输出路径")
    parser.add_argument("--output-transform-json", help="可选抓取 TF JSON 输出路径")
    parser.add_argument(
        "--visualize-3d",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否显示抓取3D可视化，默认使用配置文件 outputs.visualize_3d",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model = RGBDGrasp(args.config)
    result = model.predict_one(
        rgb=args.rgb,
        depth=args.depth,
        intrinsics=args.intrinsics,
        target=args.target,
        visualize_3d=args.visualize_3d,
        output_json=args.output_json,
        output_transform_json=args.output_transform_json,
    )
    _print_result(result)


def _print_result(result: PipelineResult) -> None:
    print(f"status: {result.status.value}")
    if result.best_grasp is not None:
        print(f"best_score: {result.best_grasp.score:.4f}")
        print(f"best_center_px: {result.best_grasp.center_px}")
        print(f"best_pose: {result.best_grasp.pose}")
    if result.error is not None:
        print(f"error: {result.error.code} - {result.error.message}")
    print(f"timings: {result.timings}")


if __name__ == "__main__":
    main()
