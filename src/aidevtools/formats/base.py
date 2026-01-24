"""格式基类和注册机制"""
import numpy as np
from typing import Dict

_registry: Dict[str, "FormatBase"] = {}

class FormatBase:
    """格式基类"""
    name: str = ""

    def load(self, path: str, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str, data: np.ndarray, **kwargs):
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _registry[cls.name] = cls()

def register(name: str, fmt: FormatBase):
    """手动注册格式"""
    _registry[name] = fmt

def get(name: str) -> FormatBase:
    """获取格式"""
    if name not in _registry:
        raise ValueError(f"未知格式: {name}")
    return _registry[name]

def load(path: str, format: str = "raw", **kwargs) -> np.ndarray:
    """加载数据"""
    return get(format).load(path, **kwargs)

def save(path: str, data: np.ndarray, format: str = "raw", **kwargs):
    """保存数据"""
    get(format).save(path, data, **kwargs)

# 导入内置格式以触发注册
from aidevtools.formats import raw  # noqa: F401
from aidevtools.formats import numpy_fmt  # noqa: F401
