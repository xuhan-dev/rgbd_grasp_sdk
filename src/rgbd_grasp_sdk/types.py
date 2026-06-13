from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class Transform:
    parent_frame: str
    child_frame: str
    pose: Pose6D


@dataclass(frozen=True)
class SegmentationRequest:
    rgb: np.ndarray
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MaskResult:
    mask: np.ndarray
    score: float | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationResult:
    masks: list[MaskResult]
    preview: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspRequest:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    target_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraspCandidate:
    pose: Pose6D
    score: float
    center_px: tuple[int, int]
    width: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraspPredictionResult:
    candidates: list[GraspCandidate]
    point_cloud: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineError:
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    status: PipelineStatus
    best_grasp: GraspCandidate | None = None
    candidate_grasps: list[GraspCandidate] = field(default_factory=list)
    target_mask: np.ndarray | None = None
    point_cloud: Any | None = None
    segmentation_preview: np.ndarray | None = None
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: PipelineError | None = None
