"""BFP Golden API Python 包装器

必须加载 C++ 实现，失败时报错。
"""
import sys
import glob
import numpy as np
from pathlib import Path
from typing import Tuple

from aidevtools.core.log import logger

# BFP 模块目录
_BFP_DIR = Path(__file__).parent
_CPP_DIR = _BFP_DIR / "cpp"

# 加载 C++ 扩展
_cpp = None
_import_error = None
_import_detail = ""

try:
    from aidevtools.formats.custom.bfp import bfp_golden as _cpp
    logger.debug("BFP C++ Golden API 加载成功")
except ImportError as e:
    _import_error = e

    # 收集诊断信息
    so_files = list(_BFP_DIR.glob("*.so")) + list(_BFP_DIR.glob("*.pyd"))
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    expected_suffix = f"cpython-{sys.version_info.major}{sys.version_info.minor}"

    if not so_files:
        _import_detail = f"未找到编译产物 (.so/.pyd 文件)"
    else:
        so_names = [f.name for f in so_files]
        if not any(expected_suffix in name for name in so_names):
            _import_detail = (
                f"Python 版本不匹配\n"
                f"  当前 Python: {python_version} (需要 {expected_suffix})\n"
                f"  已有文件: {so_names}"
            )
        else:
            _import_detail = f"文件存在但加载失败: {so_names}"

    logger.warn(f"BFP C++ Golden API 加载失败: {e}")


def _check_cpp():
    """检查 C++ 扩展是否可用"""
    if _cpp is None:
        error_msg = (
            f"BFP C++ Golden API 加载失败\n"
            f"{'=' * 50}\n"
            f"原因: {_import_detail}\n"
            f"原始错误: {_import_error}\n"
            f"{'=' * 50}\n"
            f"目录: {_BFP_DIR}\n"
            f"{'=' * 50}\n"
            f"解决方法:\n"
            f"  cd {_CPP_DIR}\n"
            f"  bash build.sh\n"
        )
        raise ImportError(error_msg)


def is_cpp_available() -> bool:
    """检查 C++ 实现是否可用"""
    return _cpp is not None


# ==================== 统一接口 ====================

def fp32_to_bfp(data: np.ndarray, block_size: int = 16, mantissa_bits: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    fp32 -> BFP

    Args:
        data: 输入数据 (float32)
        block_size: 块大小
        mantissa_bits: 尾数位数

    Returns:
        (mantissas, shared_exps)
    """
    _check_cpp()
    data = np.ascontiguousarray(data.flatten(), dtype=np.float32)
    return _cpp.fp32_to_bfp(data, block_size, mantissa_bits)


def bfp_to_fp32(mantissas: np.ndarray, shared_exps: np.ndarray,
               block_size: int = 16, mantissa_bits: int = 8) -> np.ndarray:
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
    _check_cpp()
    mantissas = np.ascontiguousarray(mantissas, dtype=np.int8)
    shared_exps = np.ascontiguousarray(shared_exps, dtype=np.int8)
    return _cpp.bfp_to_fp32(mantissas, shared_exps, block_size, mantissa_bits)


# ==================== Golden 注册 ====================

def register_bfp_golden():
    """
    注册 BFP Golden 实现

    必须先编译 C++ 扩展，否则报错。
    """
    _check_cpp()

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
            "cpp": True
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
            "cpp": True
        }

    logger.info("BFP Golden 已注册 (C++ 实现)")
