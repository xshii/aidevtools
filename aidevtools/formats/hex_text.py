"""Hex-text 格式 (每行一个字节的十六进制文本)"""
import numpy as np

from .base import FormatBase


class HexTextFormat(FormatBase):
    """Hex-text 格式处理器

    文件格式: 每行 2 个 hex 字符表示 1 个字节，例如::

        3F
        80
        00
        00
    """

    name = "hex_text"

    def load(self, path: str, dtype=np.uint8, shape=None, **kwargs) -> np.ndarray:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        raw = np.array([int(line, 16) for line in lines], dtype=np.uint8)
        if dtype != np.uint8:
            raw = np.frombuffer(raw.tobytes(), dtype=dtype)
        if shape is not None:
            raw = raw.reshape(shape)
        return raw

    def save(self, path: str, data: np.ndarray, **kwargs):
        raw = np.frombuffer(data.tobytes(), dtype=np.uint8)
        with open(path, "w") as f:
            for b in raw:
                f.write(f"{b:02X}\n")
