import json

from rgbd_grasp_sdk.publishers.json_file import JsonFilePublisher
from rgbd_grasp_sdk.publishers.stdout import StdoutPublisher
from rgbd_grasp_sdk.publishers.transform_message import best_grasp_to_transform_message
from rgbd_grasp_sdk.types import GraspCandidate, PipelineResult, PipelineStatus, Pose6D


def _result():
    return PipelineResult(
        status=PipelineStatus.SUCCESS,
        best_grasp=GraspCandidate(
            pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.0, pitch=0.1, yaw=0.2),
            score=0.9,
            center_px=(320, 240),
            width=0.06,
        ),
    )


def test_best_grasp_to_transform_message():
    message = best_grasp_to_transform_message(_result())

    assert message["parent_frame"] == "camera_color_optical_frame"
    assert message["child_frame"] == "grasp_tcp"
    assert message["translation"] == {"x": 0.1, "y": 0.2, "z": 0.3}
    assert message["rotation_rpy"] == {"roll": 0.0, "pitch": 0.1, "yaw": 0.2}
    assert message["score"] == 0.9
    assert message["center_px"] == [320, 240]
    assert message["width"] == 0.06


def test_json_file_publisher_writes_result(tmp_path):
    output_path = tmp_path / "result.json"
    JsonFilePublisher(output_path).publish(_result())

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["status"] == "success"
    assert data["best_grasp"]["score"] == 0.9


def test_stdout_publisher_prints_transform_message(capsys):
    StdoutPublisher().publish(_result())

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["child_frame"] == "grasp_tcp"
