"""GFloat Golden API Python 包装器

必须加载 C++ 实现，失败时报错。
"""
from pathlib import Path
from typing import Tuple

import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.custom.cpp_loader import CppExtensionLoader

# 加载 C++ 扩展
_loader = CppExtensionLoader(
    name="GFloat",
    module_path=Path(__file__).parent,
    import_path="aidevtools.formats.custom.gfloat",
    module_name="gfloat_golden",
)


def is_cpp_available() -> bool:
    """检查 C++ 实现是否可用"""
    return _loader.is_available()


# ==================== 统一接口 ====================

def fp32_to_gfloat16(data: np.ndarray) -> np.ndarray:
    """fp32 -> gfloat16"""
    _loader.check()
    data = np.ascontiguousarray(data, dtype=np.float32)
    return _loader.module.fp32_to_gfloat16(data)


def gfloat16_to_fp32(data: np.ndarray) -> np.ndarray:
    """gfloat16 -> fp32"""
    _loader.check()
    data = np.ascontiguousarray(data, dtype=np.uint16)
    return _loader.module.gfloat16_to_fp32(data)


def fp32_to_gfloat8(data: np.ndarray) -> np.ndarray:
    """fp32 -> gfloat8"""
    _loader.check()
    data = np.ascontiguousarray(data, dtype=np.float32)
    return _loader.module.fp32_to_gfloat8(data)


def gfloat8_to_fp32(data: np.ndarray) -> np.ndarray:
    """gfloat8 -> fp32"""
    _loader.check()
    data = np.ascontiguousarray(data, dtype=np.uint8)
    return _loader.module.gfloat8_to_fp32(data)


# ==================== Golden 注册 ====================

def register_gfloat_golden():
    """
    注册 GFloat Golden 实现

    必须先编译 C++ 扩展，否则报错。
    """
    _loader.check()

    from aidevtools.formats.quantize import register_quantize

    @register_quantize("gfloat16_golden")
    def golden_gfloat16(data: np.ndarray, **_kwargs) -> Tuple[np.ndarray, dict]:
        """Golden gfloat16 量化"""
        result = fp32_to_gfloat16(data)
        return result, {"format": "gfloat16_golden", "cpp": True}

    @register_quantize("gfloat8_golden")
    def golden_gfloat8(data: np.ndarray, **_kwargs) -> Tuple[np.ndarray, dict]:
        """Golden gfloat8 量化"""
        result = fp32_to_gfloat8(data)
        return result, {"format": "gfloat8_golden", "cpp": True}

    logger.info("GFloat Golden 已注册 (C++ 实现)")
