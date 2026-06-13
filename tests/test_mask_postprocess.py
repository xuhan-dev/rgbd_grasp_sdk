import numpy as np

from rgbd_grasp_sdk.config.schema import MaskConfig
from rgbd_grasp_sdk.masks.postprocess import postprocess_masks
from rgbd_grasp_sdk.types import MaskResult


def test_postprocess_masks_merges_and_dilates():
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True

    result = postprocess_masks(
        [MaskResult(mask=mask)],
        MaskConfig(merge_instances=True, dilate_kernel=3, dilate_iterations=1),
    )

    assert result.shape == (7, 7)
    assert result.sum() == 9


def test_postprocess_masks_removes_small_components():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1] = True
    mask[4:7, 4:7] = True

    result = postprocess_masks(
        [MaskResult(mask=mask)],
        MaskConfig(merge_instances=True, min_area=4),
    )

    assert not result[1, 1]
    assert result[5, 5]
