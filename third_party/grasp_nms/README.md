# grasp_nms

RegionNormalizedGrasp 运行时需要真实的 `grasp_nms` C 扩展。

当前项目默认不使用 no-op NMS fallback。请在运行真实 RNG 前安装编译版：

```bash
python3 -m pip install grasp_nms
```

安装成功后，Python 应能导入类似 `grasp_nms.cpython-310-x86_64-linux-gnu.so` 的扩展模块。
