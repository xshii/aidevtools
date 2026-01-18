"""GFloat 自定义浮点格式"""
import numpy as np

from aidevtools.formats.quantize import register_quantize


@register_quantize("gfloat16")
def to_gfloat16(data: np.ndarray, **kwargs) -> tuple:
    """
    fp32 → gfloat16 (自定义 16 位浮点格式)

    格式: 1 符号 + 8 指数 + 7 尾数
    存储: uint16
    """
    fp32_bits = data.view(np.uint32)
    gf16_bits = (fp32_bits >> 16).astype(np.uint16)
    return gf16_bits, {"format": "gfloat16_as_uint16"}


@register_quantize("gfloat8")
def to_gfloat8(data: np.ndarray, **kwargs) -> tuple:
    """
    fp32 → gfloat8 (自定义 8 位浮点格式)

    格式: 1 符号 + 4 指数 + 3 尾数
    存储: uint8
    """
    fp32_bits = data.view(np.uint32)
    gf8_bits = (fp32_bits >> 24).astype(np.uint8)
    return gf8_bits, {"format": "gfloat8_as_uint8"}
