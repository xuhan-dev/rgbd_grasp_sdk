class RgbdGraspError(Exception):
    """SDK 基础异常。"""


class ConfigError(RgbdGraspError):
    """配置缺失或配置值非法。"""


class BackendUnavailableError(RgbdGraspError):
    """请求的可选 backend 不可用。"""


class InputValidationError(RgbdGraspError):
    """输入 RGB-D、内参或目标描述非法。"""
