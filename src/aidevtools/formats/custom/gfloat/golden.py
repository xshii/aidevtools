"""GFloat 自定义浮点格式"""
import numpy as np

from aidevtools.formats._quantize_registry import register_quantize


@register_quantize("gfloat16")
def to_gfloat16(data: np.ndarray, **_kwargs) -> tuple:
    """
    fp32 → gfloat16 (自定义 16 位浮点格式)

    格式: 1 符号 + 8 指数 + 7 尾数
    存储: uint16
    """
    fp32_bits = data.view(np.uint32)
    gf16_bits = (fp32_bits >> 16).astype(np.uint16)
    return gf16_bits, {"format": "gfloat16_as_uint16", "original_shape": data.shape}


@register_quantize("gfloat8")
def to_gfloat8(data: np.ndarray, **_kwargs) -> tuple:
    """
    fp32 → gfloat8 (自定义 8 位浮点格式)

    格式: 1 符号 + 4 指数 + 3 尾数
    存储: uint8
    """
    fp32_bits = data.view(np.uint32)
    gf8_bits = (fp32_bits >> 24).astype(np.uint8)
    return gf8_bits, {"format": "gfloat8_as_uint8", "original_shape": data.shape}


def from_gfloat16(data: np.ndarray, original_shape: tuple = None) -> np.ndarray:
    """
    gfloat16 → fp32 (反量化)

    将 uint16 数据还原为 fp32
    """
    # gfloat16 存储的是 fp32 高 16 位，低 16 位补零
    fp32_bits = data.astype(np.uint32) << 16
    result = fp32_bits.view(np.float32)
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result


def from_gfloat8(data: np.ndarray, original_shape: tuple = None) -> np.ndarray:
    """
    gfloat8 → fp32 (反量化)

    将 uint8 数据还原为 fp32
    """
    # gfloat8 存储的是 fp32 高 8 位，低 24 位补零
    fp32_bits = data.astype(np.uint32) << 24
    result = fp32_bits.view(np.float32)
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result


@register_quantize("gfloat4")
def to_gfloat4(data: np.ndarray, **_kwargs) -> tuple:
    """
    fp32 → gfloat4 (自定义 4 位浮点格式)

    格式: 1 符号 + 2 指数 + 1 尾数
    存储: uint8 (高 4 位有效，低 4 位为 0)

    极端量化格式，用于超低精度场景
    """
    fp32_bits = data.view(np.uint32)
    # 取高 4 位，存储在 uint8 的高 4 位
    gf4_bits = ((fp32_bits >> 28) << 4).astype(np.uint8)
    return gf4_bits, {"format": "gfloat4_as_uint8", "original_shape": data.shape}


def from_gfloat4(data: np.ndarray, original_shape: tuple = None) -> np.ndarray:
    """
    gfloat4 → fp32 (反量化)

    将 uint8 数据 (高4位有效) 还原为 fp32
    """
    # gfloat4 存储在 uint8 高 4 位，还原时放到 fp32 高 4 位
    fp32_bits = (data.astype(np.uint32) >> 4) << 28
    result = fp32_bits.view(np.float32)
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result
