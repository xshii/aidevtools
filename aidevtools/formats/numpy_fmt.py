"""Numpy 格式 (npy/npz)"""
import numpy as np

from .base import FormatBase


class NumpyFormat(FormatBase):
    """Numpy npy/npz 格式处理器"""
    name = "numpy"

    def load(self, path: str, **kwargs) -> np.ndarray:
        if path.endswith(".npz"):
            data = np.load(path)
            key = kwargs.get("key", list(data.keys())[0])
            return data[key]
        return np.load(path)

    def save(self, path: str, data: np.ndarray, **kwargs):
        if path.endswith(".npz"):
            np.savez(path, data=data)
        else:
            np.save(path, data)
