"""GFloat 自定义浮点格式

GFloat 格式以 block_size=1, exp_bytes=0 注册到统一 block format 框架。
"""
import numpy as np

from aidevtools.formats.block_format import BlockFormatSpec, DecodeResult, FormatInfo, register_block_format


# ==================== quantize 函数 ====================

def to_gfloat16(data: np.ndarray, **_kwargs) -> tuple:
    """
    fp32 → gfloat16 (自定义 16 位浮点格式)

    格式: 1 符号 + 8 指数 + 7 尾数
    存储: uint16
    """
    fp32_bits = data.view(np.uint32)
    gf16_bits = (fp32_bits >> 16).astype(np.uint16)
    return gf16_bits, {"format": "gfloat16_as_uint16", "original_shape": data.shape}


def to_gfloat8(data: np.ndarray, **_kwargs) -> tuple:
    """
    fp32 → gfloat8 (自定义 8 位浮点格式)

    格式: 1 符号 + 4 指数 + 3 尾数
    存储: uint8
    """
    fp32_bits = data.view(np.uint32)
    gf8_bits = (fp32_bits >> 24).astype(np.uint8)
    return gf8_bits, {"format": "gfloat8_as_uint8", "original_shape": data.shape}


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


# ==================== dequantize 函数 (返回 DecodeResult) ====================

def _gfloat16_dequantize(data: np.ndarray, meta: dict) -> DecodeResult:
    """gfloat16 → DecodeResult"""
    bits = data.astype(np.uint16)
    exp = ((bits >> 7) & 0xFF).astype(np.int32)
    mant = (bits & 0x7F).astype(np.int32)
    fp32_bits = bits.astype(np.uint32) << 16
    values = fp32_bits.view(np.float32).copy()
    return DecodeResult(values=values, exponent=exp, mantissa=mant)


def _gfloat8_dequantize(data: np.ndarray, meta: dict) -> DecodeResult:
    """gfloat8 → DecodeResult"""
    bits = data.astype(np.uint8)
    exp = ((bits >> 3) & 0x0F).astype(np.int32)
    mant = (bits & 0x07).astype(np.int32)
    fp32_bits = bits.astype(np.uint32) << 24
    values = fp32_bits.view(np.float32).copy()
    return DecodeResult(values=values, exponent=exp, mantissa=mant)


def _gfloat4_dequantize(data: np.ndarray, meta: dict) -> DecodeResult:
    """gfloat4 → DecodeResult"""
    bits = data.astype(np.uint8)
    # 有效数据在高 4 位
    nibble = (bits >> 4) & 0x0F
    exp = ((nibble >> 1) & 0x03).astype(np.int32)
    mant = (nibble & 0x01).astype(np.int32)
    fp32_bits = (nibble.astype(np.uint32)) << 28
    values = fp32_bits.view(np.float32).copy()
    return DecodeResult(values=values, exponent=exp, mantissa=mant)


# ==================== 旧接口保留 (供外部直接调用) ====================

def from_gfloat16(data: np.ndarray, original_shape: tuple = None) -> np.ndarray:
    """gfloat16 → fp32 (反量化)"""
    fp32_bits = data.astype(np.uint32) << 16
    result = fp32_bits.view(np.float32)
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result


def from_gfloat8(data: np.ndarray, original_shape: tuple = None) -> np.ndarray:
    """gfloat8 → fp32 (反量化)"""
    fp32_bits = data.astype(np.uint32) << 24
    result = fp32_bits.view(np.float32)
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result


def from_gfloat4(data: np.ndarray, original_shape: tuple = None) -> np.ndarray:
    """gfloat4 → fp32 (反量化)"""
    fp32_bits = (data.astype(np.uint32) >> 4) << 28
    result = fp32_bits.view(np.float32)
    if original_shape is not None:
        result = result.reshape(original_shape)
    return result


# ==================== 统一注册 ====================

_GFLOAT_FORMATS = [
    # (name, block_size, quantize_fn, dequantize_fn, storage_dtype, bit_layout, description)
    ("gfloat16", 1, to_gfloat16, _gfloat16_dequantize, np.uint16, "S0 E0*8 M0*7",  "GFloat16 (1+8+7)"),
    ("gfp16",    1, to_gfloat16, _gfloat16_dequantize, np.uint16, "S0 E0*8 M0*7",  "GFloat16 alias"),
    ("gfloat8",  1, to_gfloat8,  _gfloat8_dequantize,  np.uint8,  "S0 E0*4 M0*3",  "GFloat8 (1+4+3)"),
    ("gfp8",     1, to_gfloat8,  _gfloat8_dequantize,  np.uint8,  "S0 E0*4 M0*3",  "GFloat8 alias"),
    ("gfloat4",  1, to_gfloat4,  _gfloat4_dequantize,  np.uint8,  "S0 E0*2 M0*1",  "GFloat4 (1+2+1)"),
    ("gfp4",     1, to_gfloat4,  _gfloat4_dequantize,  np.uint8,  "S0 E0*2 M0*1",  "GFloat4 alias"),
]

for _name, _bs, _qfn, _dfn, _sd, _bl, _desc in _GFLOAT_FORMATS:
    register_block_format(BlockFormatSpec(
        info=FormatInfo(
            name=_name,
            storage_dtype=_sd,
            bit_layout=_bl,
            description=_desc,
        ),
        block_size=_bs,
        quantize_fn=_qfn,
        dequantize_fn=_dfn,
    ))
