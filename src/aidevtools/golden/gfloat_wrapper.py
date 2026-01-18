"""GFloat Golden API Python 包装器

尝试加载 C++ 实现，失败时使用 Python 后备实现。
"""
import numpy as np
from typing import Tuple

from aidevtools.core.log import logger

# 尝试加载 C++ 扩展
_cpp_available = False
try:
    from aidevtools.golden import gfloat_golden as _cpp
    _cpp_available = True
    logger.debug("GFloat C++ Golden API 加载成功")
except ImportError:
    logger.debug("GFloat C++ Golden API 未编译，使用 Python 后备实现")


# ==================== Python 后备实现 ====================

def _py_fp32_to_gfloat16(data: np.ndarray) -> np.ndarray:
    """Python 实现: fp32 -> gfloat16"""
    fp32_bits = data.view(np.uint32)
    return (fp32_bits >> 16).astype(np.uint16)


def _py_gfloat16_to_fp32(data: np.ndarray) -> np.ndarray:
    """Python 实现: gfloat16 -> fp32"""
    bits = data.astype(np.uint32) << 16
    return bits.view(np.float32)


def _py_fp32_to_gfloat8(data: np.ndarray) -> np.ndarray:
    """Python 实现: fp32 -> gfloat8"""
    fp32_bits = data.view(np.uint32)
    return (fp32_bits >> 24).astype(np.uint8)


def _py_gfloat8_to_fp32(data: np.ndarray) -> np.ndarray:
    """Python 实现: gfloat8 -> fp32"""
    bits = data.astype(np.uint32) << 24
    return bits.view(np.float32)


# ==================== 统一接口 ====================

def fp32_to_gfloat16(data: np.ndarray) -> np.ndarray:
    """
    fp32 -> gfloat16

    优先使用 C++ 实现，失败时使用 Python 后备
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if _cpp_available:
        return _cpp.fp32_to_gfloat16(data)
    return _py_fp32_to_gfloat16(data)


def gfloat16_to_fp32(data: np.ndarray) -> np.ndarray:
    """
    gfloat16 -> fp32

    优先使用 C++ 实现，失败时使用 Python 后备
    """
    data = np.ascontiguousarray(data, dtype=np.uint16)
    if _cpp_available:
        return _cpp.gfloat16_to_fp32(data)
    return _py_gfloat16_to_fp32(data)


def fp32_to_gfloat8(data: np.ndarray) -> np.ndarray:
    """
    fp32 -> gfloat8

    优先使用 C++ 实现，失败时使用 Python 后备
    """
    data = np.ascontiguousarray(data, dtype=np.float32)
    if _cpp_available:
        return _cpp.fp32_to_gfloat8(data)
    return _py_fp32_to_gfloat8(data)


def gfloat8_to_fp32(data: np.ndarray) -> np.ndarray:
    """
    gfloat8 -> fp32

    优先使用 C++ 实现，失败时使用 Python 后备
    """
    data = np.ascontiguousarray(data, dtype=np.uint8)
    if _cpp_available:
        return _cpp.gfloat8_to_fp32(data)
    return _py_gfloat8_to_fp32(data)


def is_cpp_available() -> bool:
    """检查 C++ 实现是否可用"""
    return _cpp_available


# ==================== Golden 注册 ====================

def register_gfloat_golden():
    """
    注册 GFloat Golden 实现

    将 C++ 实现（或 Python 后备）注册为 golden，
    用于精确比对。
    """
    from aidevtools.formats.quantize import register_quantize

    @register_quantize("gfloat16_golden")
    def golden_gfloat16(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Golden gfloat16 量化"""
        result = fp32_to_gfloat16(data)
        return result, {
            "format": "gfloat16_golden",
            "cpp": _cpp_available,
        }

    @register_quantize("gfloat8_golden")
    def golden_gfloat8(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Golden gfloat8 量化"""
        result = fp32_to_gfloat8(data)
        return result, {
            "format": "gfloat8_golden",
            "cpp": _cpp_available,
        }

    impl = "C++" if _cpp_available else "Python"
    logger.info(f"GFloat Golden 已注册 ({impl} 实现)")
