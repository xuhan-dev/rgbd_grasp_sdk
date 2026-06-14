from __future__ import annotations

import json

from rgbd_grasp_sdk.publishers.transform_message import best_grasp_to_transform_message
from rgbd_grasp_sdk.types import PipelineResult


class StdoutPublisher:
    def publish(self, result: PipelineResult) -> None:
        print(json.dumps(best_grasp_to_transform_message(result), ensure_ascii=False))
