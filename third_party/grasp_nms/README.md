# grasp_nms fallback

该目录用于承载 RegionNormalizedGrasp 运行时需要的 `grasp_nms` 模块。

当前 Python fallback 直接返回输入候选，不执行真正的 6DoF NMS，目标是让真实权重 smoke test 可以在缺少编译扩展时跑通。生产环境建议替换为编译版 `grasp_nms` 实现。
