"""Block Floating Point (BFP) Python Golden 实现

BFP 将数据分块，每块共享一个指数，每个元素只存尾数。

参考:
- AMD Quark BFP16: https://quark.docs.amd.com/latest/onnx/tutorial_bfp16_quantization.html
- Static BFP CNN: https://github.com/os-hxfan/Static_BFP_CNN
"""

from typing import Tuple

import numpy as np

from aidevtools.formats._quantize_registry import register_quantize


def _compute_shared_exponent(data: np.ndarray, block_size: int) -> np.ndarray:
    """
    计算每个块的共享指数

    shared_exp = max(floor(log2(|x|))) for x in block
    """
    # 展平并填充到 block_size 的倍数
    flat = data.flatten().astype(np.float32)
    pad_len = (block_size - len(flat) % block_size) % block_size
    if pad_len > 0:
        flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

    # 重塑为 (num_blocks, block_size)
    blocks = flat.reshape(-1, block_size)

    # 计算每个块的最大绝对值
    max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
    max_abs = np.maximum(max_abs, 1e-10)  # 避免 log(0)

    # 共享指数 = floor(log2(max_abs)) + 1
    shared_exp = np.floor(np.log2(max_abs)).astype(np.int8) + 1

    return shared_exp, blocks, pad_len


def fp32_to_bfp(
    data: np.ndarray, block_size: int = 16, mantissa_bits: int = 8
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    fp32 -> Block Floating Point

    Args:
        data: 输入数据 (float32)
        block_size: 块大小 (默认 16)
        mantissa_bits: 尾数位数 (默认 8)

    Returns:
        (mantissas, shared_exps, meta)
        - mantissas: int8 数组，形状 (num_elements,)
        - shared_exps: int8 数组，形状 (num_blocks,)
        - meta: 元信息
    """
    original_shape = data.shape
    shared_exp, blocks, pad_len = _compute_shared_exponent(data, block_size)

    # 量化：mantissa = round(x * 2^(mantissa_bits-1) / 2^shared_exp)
    # 即 mantissa = round(x * 2^(mantissa_bits-1-shared_exp))
    scale = 2.0 ** (mantissa_bits - 1 - shared_exp)
    mantissas = np.round(blocks * scale).astype(np.int8)

    # 裁剪到有效范围
    max_val = 2 ** (mantissa_bits - 1) - 1
    mantissas = np.clip(mantissas, -max_val, max_val)

    # 展平
    mantissas_flat = mantissas.flatten()
    if pad_len > 0:
        mantissas_flat = mantissas_flat[:-pad_len]

    shared_exp_flat = shared_exp.flatten()

    meta = {
        "format": "bfp",
        "block_size": block_size,
        "mantissa_bits": mantissa_bits,
        "original_shape": original_shape,
        "num_blocks": len(shared_exp_flat),
        "pad_len": pad_len,
    }

    return mantissas_flat, shared_exp_flat, meta


def bfp_to_fp32(
    mantissas: np.ndarray,
    shared_exps: np.ndarray,
    block_size: int = 16,
    mantissa_bits: int = 8,
    original_shape: tuple = None,
) -> np.ndarray:
    """
    Block Floating Point -> fp32

    Args:
        mantissas: 尾数数组 (int8)
        shared_exps: 共享指数数组 (int8)
        block_size: 块大小
        mantissa_bits: 尾数位数
        original_shape: 原始形状

    Returns:
        还原的 float32 数据
    """
    # 填充
    pad_len = (block_size - len(mantissas) % block_size) % block_size
    if pad_len > 0:
        mantissas = np.concatenate([mantissas, np.zeros(pad_len, dtype=np.int8)])

    # 重塑
    blocks = mantissas.reshape(-1, block_size).astype(np.float32)
    shared_exps = shared_exps.reshape(-1, 1)

    # 反量化: x = mantissa * 2^(shared_exp) / 2^(mantissa_bits-1)
    scale = 2.0 ** (shared_exps - (mantissa_bits - 1))
    restored = blocks * scale

    # 展平并恢复形状
    flat = restored.flatten()
    if pad_len > 0:
        flat = flat[:-pad_len]

    if original_shape is not None:
        flat = flat.reshape(original_shape)

    return flat.astype(np.float32)


# ==================== 量化注册 ====================

# BFP 格式配置: (format_name, default_block_size, mantissa_bits, description)
_BFP_FORMATS = [
    ("bfp16", 16, 8, "BFP16 (block_size=16, mantissa=8bit)"),
    ("bfp8", 32, 4, "BFP8 (block_size=32, mantissa=4bit)"),
    ("bfp4", 64, 2, "BFP4 (block_size=64, mantissa=2bit) - 极端量化"),
]


def _make_bfp_quantizer(default_block_size: int, mantissa_bits: int):
    """生成 BFP 量化函数"""

    def quantizer(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        block_size = kwargs.get("block_size", default_block_size)
        mantissas, shared_exps, meta = fp32_to_bfp(
            data, block_size=block_size, mantissa_bits=mantissa_bits
        )
        packed = np.concatenate([shared_exps.astype(np.int8), mantissas.astype(np.int8)])
        return packed, meta

    return quantizer


# 批量注册 BFP 格式
for _name, _block_size, _mantissa_bits, _desc in _BFP_FORMATS:
    _fn = _make_bfp_quantizer(_block_size, _mantissa_bits)
    _fn.__doc__ = f"fp32 -> {_desc}"
    register_quantize(_name)(_fn)
