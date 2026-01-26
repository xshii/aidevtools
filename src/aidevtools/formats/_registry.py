"""格式注册表 (内部模块)

将注册表独立出来避免循环导入：
- base.py 定义 FormatBase，从这里导入注册函数
- numpy_fmt.py, raw.py 导入 FormatBase，自动注册
"""
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from aidevtools.formats.base import FormatBase

_registry: Dict[str, "FormatBase"] = {}


def register(name: str, fmt: "FormatBase"):
    """注册格式"""
    _registry[name] = fmt


def get(name: str) -> "FormatBase":
    """获取格式"""
    if name not in _registry:
        raise ValueError(f"未知格式: {name}")
    return _registry[name]


def list_formats() -> list:
    """列出所有已注册格式"""
    return list(_registry.keys())
