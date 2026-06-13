from rgbd_grasp_sdk.serialization.result_json import pipeline_result_to_dict
from rgbd_grasp_sdk.transforms.pose import grasp_to_transform
from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D


def test_grasp_to_transform_uses_frame_names():
    grasp = GraspCandidate(
        pose=Pose6D(0.1, 0.2, 0.3, 0.0, 1.57, 3.14),
        score=0.8,
        center_px=(10, 20),
    )

    transform = grasp_to_transform(grasp, parent_frame="camera", child_frame="grasp")

    assert transform.parent_frame == "camera"
    assert transform.child_frame == "grasp"
    assert transform.pose is grasp.pose


def test_pipeline_result_to_dict_is_json_friendly():
    grasp = GraspCandidate(
        pose=Pose6D(0.1, 0.2, 0.3, 0.0, 1.57, 3.14),
        score=0.8,
        center_px=(10, 20),
    )
    result = PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=grasp,
        candidate_grasps=[grasp],
        timings={"total": 0.01},
    )

    payload = pipeline_result_to_dict(result)

    assert payload["status"] == "success"
    assert payload["best_grasp"]["score"] == 0.8
    assert payload["best_grasp"]["pose"]["z"] == 0.3
