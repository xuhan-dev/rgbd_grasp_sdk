from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_grasp_sdk.config.loader import load_config
from rgbd_grasp_sdk.config.schema import SdkConfig
from rgbd_grasp_sdk.datasets import GraspSample, normalize_samples
from rgbd_grasp_sdk.errors import InputValidationError
from rgbd_grasp_sdk.grasping.factory import create_grasp_predictor
from rgbd_grasp_sdk.io import read_depth, read_intrinsics_npz, read_rgb
from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
from rgbd_grasp_sdk.publishers import JsonFilePublisher, TransformJsonFilePublisher
from rgbd_grasp_sdk.ranking.factory import create_ranker
from rgbd_grasp_sdk.segmentation.factory import create_segmenter
from rgbd_grasp_sdk.types import CameraIntrinsics, PipelineError, PipelineResult, PipelineStatus


PipelineBuilder = Callable[[SdkConfig, bool | None], Any]


class RGBDGrasp:
    def __init__(
        self,
        config: str | Path | SdkConfig,
        *,
        pipeline_builder: PipelineBuilder | None = None,
    ) -> None:
        self.config_path = Path(config) if isinstance(config, (str, Path)) else None
        self.config = load_config(config) if isinstance(config, (str, Path)) else config
        self._pipeline_builder = pipeline_builder or self._default_pipeline_builder
        self._pipeline = self._pipeline_builder(self.config, None)

    def predict_one(
        self,
        *,
        rgb: Any,
        depth: Any,
        intrinsics: Any,
        target: str,
        strict: bool = True,
        visualize_3d: bool | None = None,
        output_json: str | Path | None = None,
        output_transform_json: str | Path | None = None,
    ) -> PipelineResult:
        try:
            sample = normalize_samples(
                source=None,
                rgb=rgb,
                depth=depth,
                intrinsics=intrinsics,
                target=target,
            )[0]
        except Exception as exc:
            if strict:
                raise
            result = self._failed_result(exc, target=target)
            self._publish_outputs(result, output_json, output_transform_json)
            return result
        result = self._run_sample(sample, strict=strict, visualize_3d=visualize_3d)
        self._publish_outputs(result, output_json, output_transform_json)
        return result

    def predict(
        self,
        source: Any = None,
        *,
        rgb: Any = None,
        depth: Any = None,
        intrinsics: Any = None,
        target: str | None = None,
        strict: bool = False,
        visualize_3d: bool | None = None,
    ) -> list[PipelineResult]:
        try:
            samples = normalize_samples(
                source,
                rgb=rgb,
                depth=depth,
                intrinsics=intrinsics,
                target=target,
            )
        except Exception as exc:
            if strict:
                raise
            has_keyword_input = any(
                value is not None for value in (rgb, depth, intrinsics, target)
            )
            if isinstance(source, list) and not has_keyword_input:
                return self._run_source_list_items(
                    source,
                    strict=strict,
                    visualize_3d=visualize_3d,
                )
            source_target, source_id = _source_metadata(source)
            return [
                self._failed_result(
                    exc,
                    target=target if has_keyword_input else source_target,
                    sample_id=source_id,
                )
            ]
        return [
            self._run_sample(sample, strict=strict, visualize_3d=visualize_3d)
            for sample in samples
        ]

    def _run_source_list_items(
        self,
        source: list[Any],
        *,
        strict: bool,
        visualize_3d: bool | None,
    ) -> list[PipelineResult]:
        results = []
        for item in source:
            try:
                sample = normalize_samples(item)[0]
            except Exception as exc:
                if strict:
                    raise
                source_target, source_id = _source_metadata(item)
                results.append(
                    self._failed_result(exc, target=source_target, sample_id=source_id)
                )
                continue
            results.append(
                self._run_sample(sample, strict=strict, visualize_3d=visualize_3d)
            )
        return results

    def _run_sample(
        self,
        sample: GraspSample,
        *,
        strict: bool,
        visualize_3d: bool | None,
    ) -> PipelineResult:
        try:
            if not sample.target:
                raise InputValidationError("target 必须是非空字符串")
            pipeline = self._pipeline
            if visualize_3d is not None:
                pipeline = self._pipeline_builder(self.config, visualize_3d)
            return pipeline.run(
                rgb=self._load_rgb(sample.rgb),
                depth=self._load_depth(sample.depth),
                intrinsics=self._load_intrinsics(sample.intrinsics),
                target=sample.target,
            )
        except Exception as exc:
            if strict:
                raise
            return self._failed_result(exc, target=sample.target, sample_id=sample.id)

    def _failed_result(
        self,
        exc: Exception,
        *,
        target: str | None,
        sample_id: str | None = None,
    ) -> PipelineResult:
        return PipelineResult(
            status=PipelineStatus.FAILED,
            error=PipelineError(
                code=_exception_code(exc),
                message=str(exc),
                detail={"exception_type": type(exc).__name__},
            ),
            metadata={"target": target, "sample_id": sample_id},
        )

    def _load_rgb(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            return read_rgb(value)
        raise InputValidationError("rgb 必须是 np.ndarray 或图片路径")

    def _load_depth(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (str, Path)):
            return read_depth(value)
        raise InputValidationError("depth 必须是 np.ndarray 或图片路径")

    def _load_intrinsics(self, value: Any) -> CameraIntrinsics:
        if isinstance(value, CameraIntrinsics):
            return value
        if isinstance(value, (str, Path)):
            return read_intrinsics_npz(value)
        raise InputValidationError("intrinsics 必须是 CameraIntrinsics 或 npz 路径")

    def _publish_outputs(
        self,
        result: PipelineResult,
        output_json: str | Path | None,
        output_transform_json: str | Path | None,
    ) -> None:
        if output_json is not None:
            JsonFilePublisher(output_json).publish(result)
        if output_transform_json is not None:
            TransformJsonFilePublisher(output_transform_json).publish(result)

    def _default_pipeline_builder(
        self,
        config: SdkConfig,
        visualize_3d: bool | None,
    ) -> GraspPipeline:
        segmenter = create_segmenter(config.segmentation.backend, config.segmentation.options)
        grasp_predictor = create_grasp_predictor(config.grasping.backend, config.grasping.options)
        return GraspPipeline(
            segmenter=segmenter,
            grasp_predictor=grasp_predictor,
            ranker=create_ranker(config.ranking),
            mask_config=config.mask,
            visualize_3d=config.outputs.visualize_3d if visualize_3d is None else visualize_3d,
        )


def _exception_code(exc: Exception) -> str:
    name = type(exc).__name__
    snake = []
    for index, char in enumerate(name):
        if char.isupper() and index > 0:
            snake.append("_")
        snake.append(char.lower())
    return "".join(snake)


def _source_metadata(source: Any) -> tuple[str | None, str | None]:
    if not isinstance(source, dict):
        return None, None
    target = source.get("target")
    sample_id = source.get("id")
    return (
        target if isinstance(target, str) else None,
        sample_id if isinstance(sample_id, str) else None,
    )
