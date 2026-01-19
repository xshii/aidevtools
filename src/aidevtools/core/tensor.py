"""统一的张量格式"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class Tensor:
    """
    统一的张量格式

    每个张量同时包含:
    - fp32: 最高精度数据 (始终存在)
    - quantized: 量化后数据 (qtype != float32 时存在)
    - meta: 量化元信息
    """
    fp32: np.ndarray                           # 最高精度 (始终有)
    quantized: Optional[np.ndarray] = None     # 量化后 (可选)
    meta: Dict[str, Any] = field(default_factory=dict)
    qtype: str = "float32"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.fp32.shape

    @property
    def dtype(self) -> np.dtype:
        return self.fp32.dtype

    @classmethod
    def from_fp32(cls, data: np.ndarray, qtype: str = "float32") -> "Tensor":
        """
        从 fp32 创建张量，自动量化

        Args:
            data: fp32 数据
            qtype: 目标量化类型

        Returns:
            Tensor 对象
        """
        fp32 = data.astype(np.float32)

        if qtype == "float32":
            return cls(fp32=fp32, quantized=None, meta={}, qtype=qtype)

        from aidevtools.formats.quantize import quantize
        quantized, meta = quantize(fp32, qtype)
        return cls(fp32=fp32, quantized=quantized, meta=meta, qtype=qtype)

    @classmethod
    def from_quantized(cls, quantized: np.ndarray, qtype: str, meta: Dict[str, Any] = None) -> "Tensor":
        """
        从量化数据创建张量，反量化得到 fp32

        Args:
            quantized: 量化后的数据
            qtype: 量化类型
            meta: 量化元信息

        Returns:
            Tensor 对象
        """
        from aidevtools.formats.quantize import dequantize
        fp32 = dequantize(quantized, qtype, meta or {})
        return cls(fp32=fp32, quantized=quantized, meta=meta or {}, qtype=qtype)

    def to_qtype(self, qtype: str) -> "Tensor":
        """
        转换为新的量化类型

        Args:
            qtype: 目标量化类型

        Returns:
            新的 Tensor 对象
        """
        return Tensor.from_fp32(self.fp32, qtype)

    def quantize_dequantize(self) -> "Tensor":
        """
        量化再反量化，模拟量化精度损失

        Returns:
            新的 Tensor，fp32 包含量化误差
        """
        if self.qtype == "float32" or self.quantized is None:
            return self

        from aidevtools.formats.quantize import dequantize
        fp32_with_loss = dequantize(self.quantized, self.qtype, self.meta)

        return Tensor(
            fp32=fp32_with_loss,
            quantized=self.quantized,
            meta=self.meta,
            qtype=self.qtype,
        )

    def __repr__(self):
        q_info = f", quantized={self.quantized.dtype}" if self.quantized is not None else ""
        return f"Tensor(shape={self.shape}, qtype={self.qtype}{q_info})"


def generate_random(shape: Tuple[int, ...], qtype: str = "float32",
                    seed: int = None, scale: float = 1.0) -> Tensor:
    """
    生成随机张量

    Args:
        shape: 张量形状
        qtype: 量化类型
        seed: 随机种子
        scale: 缩放因子

    Returns:
        Tensor 对象
    """
    rng = np.random.default_rng(seed)
    fp32 = (rng.standard_normal(shape) * scale).astype(np.float32)
    return Tensor.from_fp32(fp32, qtype)


def generate_weight(shape: Tuple[int, ...], qtype: str = "float32",
                    seed: int = None, init: str = "xavier") -> Tensor:
    """
    生成权重张量

    Args:
        shape: 张量形状
        qtype: 量化类型
        seed: 随机种子
        init: 初始化方式 ("xavier", "kaiming", "normal")

    Returns:
        Tensor 对象
    """
    rng = np.random.default_rng(seed)

    if init == "xavier":
        fan_in = shape[0] if len(shape) >= 1 else 1
        fan_out = shape[1] if len(shape) >= 2 else 1
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif init == "kaiming":
        fan_in = shape[0] if len(shape) >= 1 else 1
        std = np.sqrt(2.0 / fan_in)
    else:  # normal
        std = 0.02

    fp32 = (rng.standard_normal(shape) * std).astype(np.float32)
    return Tensor.from_fp32(fp32, qtype)
