from __future__ import annotations

import json
from pathlib import Path

from rgbd_grasp_sdk.serialization import pipeline_result_to_dict
from rgbd_grasp_sdk.types import PipelineResult


class JsonFilePublisher:
    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def publish(self, result: PipelineResult) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(pipeline_result_to_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
