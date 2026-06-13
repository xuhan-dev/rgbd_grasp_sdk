from __future__ import annotations

import argparse

from rgbd_grasp_sdk.config.loader import load_config
from rgbd_grasp_sdk.grasping.factory import create_grasp_predictor
from rgbd_grasp_sdk.io import read_depth, read_intrinsics_npz, read_rgb
from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
from rgbd_grasp_sdk.segmentation.factory import create_segmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行单帧 RGB-D 抓取预测")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--rgb", required=True, help="RGB 图像路径")
    parser.add_argument("--depth", required=True, help="depth 图像路径")
    parser.add_argument("--intrinsics", required=True, help="包含 K 的相机内参 npz")
    parser.add_argument("--target", required=True, help="目标类别或文本描述")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    segmenter = create_segmenter(config.segmentation.backend, config.segmentation.options)
    grasp_predictor = create_grasp_predictor(config.grasping.backend, config.grasping.options)
    pipeline = GraspPipeline(segmenter=segmenter, grasp_predictor=grasp_predictor)

    result = pipeline.run(
        rgb=read_rgb(args.rgb),
        depth=read_depth(args.depth),
        intrinsics=read_intrinsics_npz(args.intrinsics),
        target=args.target,
    )

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
