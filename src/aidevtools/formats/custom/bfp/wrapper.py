"""BFP Golden API Python 包装器

必须加载 C++ 实现，失败时报错。
"""

from pathlib import Path
from typing import Tuple

import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.custom.cpp_loader import CppExtensionLoader

# 加载 C++ 扩展
_loader = CppExtensionLoader(
    name="BFP",
    module_path=Path(__file__).parent,
    import_path="aidevtools.formats.custom.bfp",
    module_name="bfp_golden",
)


def is_cpp_available() -> bool:
    """检查 C++ 实现是否可用"""
    return _loader.is_available()


# ==================== 统一接口 ====================


def fp32_to_bfp(
    data: np.ndarray, block_size: int = 16, mantissa_bits: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    fp32 -> BFP

    Args:
        data: 输入数据 (float32)
        block_size: 块大小
        mantissa_bits: 尾数位数

    Returns:
        (mantissas, shared_exps)
    """
    _loader.check()
    data = np.ascontiguousarray(data.flatten(), dtype=np.float32)
    return _loader.module.fp32_to_bfp(data, block_size, mantissa_bits)


def bfp_to_fp32(
    mantissas: np.ndarray, shared_exps: np.ndarray, block_size: int = 16, mantissa_bits: int = 8
) -> np.ndarray:
    """
    BFP -> fp32

    Args:
        mantissas: 尾数数组
        shared_exps: 共享指数数组
        block_size: 块大小
        mantissa_bits: 尾数位数

    Returns:
        还原的 float32 数据
    """
    _loader.check()
    mantissas = np.ascontiguousarray(mantissas, dtype=np.int8)
    shared_exps = np.ascontiguousarray(shared_exps, dtype=np.int8)
    return _loader.module.bfp_to_fp32(mantissas, shared_exps, block_size, mantissa_bits)


# ==================== Golden 注册 ====================


def register_bfp_golden():
    """
    注册 BFP Golden 实现

    必须先编译 C++ 扩展，否则报错。
    """
    _loader.check()

    from aidevtools.formats.quantize import register_quantize

    @register_quantize("bfp16_golden")
    def golden_bfp16(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Golden BFP16 量化 (C++ 实现)"""
        block_size = kwargs.get("block_size", 16)
        mantissas, shared_exps = fp32_to_bfp(data, block_size=block_size, mantissa_bits=8)

        # 打包
        packed = np.concatenate([shared_exps, mantissas])

        return packed, {
            "format": "bfp16_golden",
            "block_size": block_size,
            "mantissa_bits": 8,
            "cpp": True,
        }

    @register_quantize("bfp8_golden")
    def golden_bfp8(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Golden BFP8 量化 (C++ 实现)"""
        block_size = kwargs.get("block_size", 32)
        mantissas, shared_exps = fp32_to_bfp(data, block_size=block_size, mantissa_bits=4)

        # 打包
        packed = np.concatenate([shared_exps, mantissas])

        return packed, {
            "format": "bfp8_golden",
            "block_size": block_size,
            "mantissa_bits": 4,
            "cpp": True,
        }

    logger.info("BFP Golden 已注册 (C++ 实现)")
