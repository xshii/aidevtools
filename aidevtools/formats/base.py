"""格式基类"""
import numpy as np

from aidevtools.formats._registry import get, register


class FormatBase:
    """格式基类"""

    name: str = ""

    def load(self, path: str, **kwargs) -> np.ndarray:
        """加载数据文件"""
        raise NotImplementedError

    def save(self, path: str, data: np.ndarray, **kwargs):
        """保存数据到文件"""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            register(cls.name, cls())


def load(path: str, fmt: str = "raw", **kwargs) -> np.ndarray:
    """加载数据"""
    return get(fmt).load(path, **kwargs)


def save(path: str, data: np.ndarray, fmt: str = "raw", **kwargs):
    """保存数据"""
    get(fmt).save(path, data, **kwargs)


# 注意: 内置格式的注册已移至 formats/__init__.py
# 这样可以避免循环导入 (base <-> numpy_fmt/raw)
