import numpy as np

from rgbd_grasp_sdk.segmentation.fastsam_segmenter import FastSamSegmenter
from rgbd_grasp_sdk.types import SegmentationRequest


class FakeFastSamBackend:
    def segment_text(self, image, text):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[1, 1] = True
        return mask, image.copy()


def test_fastsam_segmenter_returns_text_prompt_mask():
    segmenter = FastSamSegmenter({"model_path": "unused.pt"}, backend=FakeFastSamBackend())

    result = segmenter.segment(
        SegmentationRequest(rgb=np.zeros((3, 3, 3), dtype=np.uint8), target="red apple")
    )

    assert len(result.masks) == 1
    assert result.masks[0].label == "red apple"
    assert result.masks[0].mask[1, 1]
    assert result.preview is not None
