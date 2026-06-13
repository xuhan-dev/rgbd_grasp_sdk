import numpy as np

from rgbd_grasp_sdk.segmentation.yolo_segmenter import YoloSegmenter
from rgbd_grasp_sdk.types import SegmentationRequest


class FakeMask:
    def __init__(self):
        self.data = np.array([[[0, 1], [0, 1]]], dtype=np.float32)


class FakeBoxes:
    cls = np.array([0])
    conf = np.array([0.75])


class FakeResult:
    names = {0: "apple"}
    masks = FakeMask()
    boxes = FakeBoxes()


class FakeModel:
    def __call__(self, image, **kwargs):
        return [FakeResult()]


def test_yolo_segmenter_converts_matching_class_to_mask():
    segmenter = YoloSegmenter({"model_path": "unused.pt"}, model=FakeModel())

    result = segmenter.segment(
        SegmentationRequest(rgb=np.zeros((2, 2, 3), dtype=np.uint8), target="apple")
    )

    assert len(result.masks) == 1
    assert result.masks[0].label == "apple"
    assert result.masks[0].score == 0.75
    assert result.masks[0].mask.dtype == bool
    assert result.masks[0].mask[:, 1].all()
