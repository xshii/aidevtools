"""Block Format 通用注册框架

一次 register_block_format() 调用，自动接入 load / dequantize / compare / bit analysis 全链路。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# 函数原型 — 下游工具 (compare / bit analysis) 据此调用
QuantizeFn = Callable[..., Tuple[np.ndarray, dict]]
"""(data: ndarray, **kwargs) -> (packed, meta)"""

DequantizeFn = Callable[[np.ndarray, dict], "DecodeResult"]
"""(data: ndarray, meta: dict) -> DecodeResult"""

from aidevtools.formats._quantize_registry import register_quantize
from aidevtools.formats.quantize import register_dequantize


@dataclass
class DecodeResult:
    """dequantize_fn 的结构化返回值

    values:   fp32 重建值,      shape (N,)
    mantissa: block 级整数位,   shape (N,)
    exponent: block 共享指数,   shape (num_blocks,)
    sign:     自动推导 — np.signbit(values)
    """
    values: np.ndarray
    exponent: np.ndarray
    mantissa: np.ndarray

    @property
    def sign(self) -> np.ndarray:
        """从 fp32 values 自动推导符号位: 负数=1, 正数/零=0"""
        return np.signbit(self.values).astype(np.uint8)


@dataclass
class FormatInfo:
    """对外稳定接口 — 格式规格，定下来后不改"""
    name: str
    storage_dtype: type = np.int8
    bytes_per_block: int = 0
    bit_layout: str = ""
    description: str = ""


@dataclass
class BlockFormatSpec:
    """完整注册 = FormatInfo + 内部实现"""
    info: FormatInfo
    block_size: int = 1
    quantize_fn: Optional[QuantizeFn] = None
    dequantize_fn: Optional[DequantizeFn] = None

    def __post_init__(self):
        if self.info.bytes_per_block == 0:
            self.info.bytes_per_block = (
                self.block_size * np.dtype(self.info.storage_dtype).itemsize
            )

    # 代理属性 — spec.name / spec.storage_dtype 等现有代码不用改
    @property
    def name(self): return self.info.name
    @property
    def storage_dtype(self): return self.info.storage_dtype
    @property
    def bytes_per_block(self): return self.info.bytes_per_block
    @property
    def bit_layout(self): return self.info.bit_layout
    @property
    def description(self): return self.info.description


_registry: Dict[str, BlockFormatSpec] = {}


def register_block_format(spec: BlockFormatSpec):
    """一次注册，自动接入全链路"""
    _registry[spec.name] = spec
    # 自动注册到 quantize registry
    register_quantize(spec.name)(spec.quantize_fn)
    # 自动注册到 dequantize registry — 提取 .values + reshape 保持外部接口不变
    def _dequantize_wrapper(data, meta, _fn=spec.dequantize_fn):
        result = _fn(data, meta)
        if isinstance(result, DecodeResult):
            values = result.values
            original_shape = meta.get("original_shape")
            if original_shape is not None:
                values = values.reshape(original_shape)
            return values
        return result
    register_dequantize(spec.name)(_dequantize_wrapper)


def decode(data: np.ndarray, qtype: str, meta: dict) -> DecodeResult:
    """完整解码，扩充共享指数

    调用 dequantize_fn 获取 DecodeResult，如果 exponent 长度
    小于 values（block 共享），自动 repeat 到 per-element。
    """
    spec = _registry[qtype]
    result = spec.dequantize_fn(data, meta)
    if len(result.exponent) < len(result.values):
        result.exponent = np.repeat(
            result.exponent, spec.block_size
        )[:len(result.values)]
    return result


def get_block_format(name: str) -> Optional[BlockFormatSpec]:
    """获取 block format spec，不存在返回 None"""
    return _registry.get(name)


def get_format_info(name: str) -> Optional[FormatInfo]:
    """获取对外稳定的格式信息，不存在返回 None"""
    spec = _registry.get(name)
    return spec.info if spec else None


def is_block_format(name: str) -> bool:
    """判断是否为已注册的 block format"""
    return name in _registry


def list_block_formats() -> List[str]:
    """列出所有已注册的 block format 名称"""
    return list(_registry.keys())


