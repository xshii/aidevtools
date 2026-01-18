"""GFloat Golden API Python 包装器

必须加载 C++ 实现，失败时报错。
"""
import numpy as np
from typing import Tuple

from aidevtools.core.log import logger

# 加载 C++ 扩展（失败时报错）
_cpp = None
_import_error = None

try:
    from aidevtools.golden import gfloat_golden as _cpp
    logger.debug("GFloat C++ Golden API 加载成功")
except ImportError as e:
    _import_error = e
    logger.warn(f"GFloat C++ Golden API 加载失败: {e}")


def _check_cpp():
    """检查 C++ 扩展是否可用，不可用时报错"""
    if _cpp is None:
        raise ImportError(
            f"GFloat C++ Golden API 未编译或加载失败。\n"
            f"原因: {_import_error}\n"
            f"请先编译: cd src/aidevtools/golden/cpp && bash build.sh"
        )


# ==================== 统一接口 ====================

def fp32_to_gfloat16(data: np.ndarray) -> np.ndarray:
    """
    fp32 -> gfloat16

    使用 C++ 实现，未编译时报错
    """
    _check_cpp()
    data = np.ascontiguousarray(data, dtype=np.float32)
    return _cpp.fp32_to_gfloat16(data)


def gfloat16_to_fp32(data: np.ndarray) -> np.ndarray:
    """
    gfloat16 -> fp32

    使用 C++ 实现，未编译时报错
    """
    _check_cpp()
    data = np.ascontiguousarray(data, dtype=np.uint16)
    return _cpp.gfloat16_to_fp32(data)


def fp32_to_gfloat8(data: np.ndarray) -> np.ndarray:
    """
    fp32 -> gfloat8

    使用 C++ 实现，未编译时报错
    """
    _check_cpp()
    data = np.ascontiguousarray(data, dtype=np.float32)
    return _cpp.fp32_to_gfloat8(data)


def gfloat8_to_fp32(data: np.ndarray) -> np.ndarray:
    """
    gfloat8 -> fp32

    使用 C++ 实现，未编译时报错
    """
    _check_cpp()
    data = np.ascontiguousarray(data, dtype=np.uint8)
    return _cpp.gfloat8_to_fp32(data)


def is_cpp_available() -> bool:
    """检查 C++ 实现是否可用"""
    return _cpp is not None


# ==================== Golden 注册 ====================

def register_gfloat_golden():
    """
    注册 GFloat Golden 实现

    必须先编译 C++ 扩展，否则报错。
    """
    _check_cpp()

    from aidevtools.formats.quantize import register_quantize

    @register_quantize("gfloat16_golden")
    def golden_gfloat16(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Golden gfloat16 量化"""
        result = fp32_to_gfloat16(data)
        return result, {"format": "gfloat16_golden", "cpp": True}

    @register_quantize("gfloat8_golden")
    def golden_gfloat8(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Golden gfloat8 量化"""
        result = fp32_to_gfloat8(data)
        return result, {"format": "gfloat8_golden", "cpp": True}

    logger.info("GFloat Golden 已注册 (C++ 实现)")
