# Transform Contract

本项目输出的抓取位姿用于外部系统消费。除非适配器另有说明，所有长度单位均为米，角度单位均为弧度。

## Frames

- `parent_frame`: 默认 `camera_color_optical_frame`。
- `child_frame`: 默认 `grasp_tcp`。

## Pose

`Pose6D` 字段：

- `x`, `y`, `z`: 抓取 TCP 在 `parent_frame` 下的位置，单位米。
- `roll`, `pitch`, `yaw`: 抓取 TCP 姿态，XYZ 欧拉角，单位弧度。

## Gripper Axes

- `approach_axis`: 夹爪接近目标的方向。
- `closing_axis`: 两指闭合方向。
- `tcp`: 夹爪工具中心点，用于外部系统执行抓取。

## Message

TF message JSON 格式：

```json
{
  "parent_frame": "camera_color_optical_frame",
  "child_frame": "grasp_tcp",
  "translation": {"x": 0.0, "y": 0.0, "z": 0.5},
  "rotation_rpy": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
  "score": 0.8,
  "center_px": [320, 240],
  "width": 0.06
}
```

## Scope

该合约只定义 SDK 的本地输出格式。HTTP、ZeroMQ、ROS2、DDS 等外部通信协议由后续 publisher 适配层独立实现，不进入核心 pipeline。
