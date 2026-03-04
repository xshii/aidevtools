"""Block Format 通用注册框架

一次 register_block_format() 调用，自动接入 load / dequantize / compare / bit analysis 全链路。
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from aidevtools.formats._quantize_registry import register_quantize
from aidevtools.formats.quantize import register_dequantize


@dataclass
class BlockFormatSpec:
    """Block 量化格式描述符"""
    name: str                # "bfp8", "my_custom_bfp"
    block_size: int          # 每块元素数
    mantissa_bits: int       # 有效位数（含符号位）
    quantize_fn: Callable    # (data: ndarray, **kwargs) -> (packed, meta)
    dequantize_fn: Callable  # (data: ndarray, meta: dict) -> fp32
    storage_dtype: type = np.int8   # 存储类型
    description: str = ""


_registry: Dict[str, BlockFormatSpec] = {}


def register_block_format(spec: BlockFormatSpec):
    """一次注册，自动接入全链路"""
    _registry[spec.name] = spec
    # 自动注册到 quantize registry
    register_quantize(spec.name)(spec.quantize_fn)
    # 自动注册到 dequantize registry
    register_dequantize(spec.name)(lambda data, meta, _fn=spec.dequantize_fn: _fn(data, meta))


def get_block_format(name: str) -> Optional[BlockFormatSpec]:
    """获取 block format spec，不存在返回 None"""
    return _registry.get(name)


def is_block_format(name: str) -> bool:
    """判断是否为已注册的 block format"""
    return name in _registry


def list_block_formats() -> List[str]:
    """列出所有已注册的 block format 名称"""
    return list(_registry.keys())


def get_bit_layout(name: str):
    """从 spec 自动生成 BitLayout"""
    from aidevtools.compare.strategy.bit_analysis import BitLayout

    spec = _registry[name]
    storage_bits = np.dtype(spec.storage_dtype).itemsize * 8
    return BitLayout(1, 0, storage_bits - 1, spec.name,
                     precision_bits=spec.mantissa_bits)
