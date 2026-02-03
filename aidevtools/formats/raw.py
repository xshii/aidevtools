"""Raw 二进制格式"""
import numpy as np

from .base import FormatBase


class RawFormat(FormatBase):
    """Raw 二进制格式处理器"""
    name = "raw"

    def load(self, path: str, dtype=np.float32, shape=None, **kwargs) -> np.ndarray:
        data = np.fromfile(path, dtype=dtype)
        if shape:
            data = data.reshape(shape)
        return data

    def save(self, path: str, data: np.ndarray, **kwargs):
        data.tofile(path)
