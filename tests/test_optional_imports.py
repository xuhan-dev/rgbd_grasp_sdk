def test_base_sdk_imports_without_constructing_model_backends():
    import rgbd_grasp_sdk
    from rgbd_grasp_sdk.pipeline.grasp_pipeline import GraspPipeline
    from rgbd_grasp_sdk.types import PipelineResult

    assert rgbd_grasp_sdk.PipelineResult is PipelineResult
    assert GraspPipeline.__name__ == "GraspPipeline"
