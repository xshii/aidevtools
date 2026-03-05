"""IEEE 标准浮点格式注册

float32/float16/bfloat16 以 block_size=1 注册到统一 block format 框架。
"""
import numpy as np

from aidevtools.formats.block_format import BlockFormatSpec, DecodeResult, FormatInfo, register_block_format


# ==================== float32 ====================

def _float32_quantize(data: np.ndarray, **_kwargs) -> tuple:
    """fp32 → fp32 (identity)"""
    fp32 = data.astype(np.float32)
    return fp32, {"format": "float32", "original_shape": data.shape}


def _float32_dequantize(data: np.ndarray, meta: dict) -> DecodeResult:
    """fp32 → DecodeResult (IEEE 754 single)"""
    values = data.astype(np.float32)
    bits = values.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).astype(np.int32)
    mant = (bits & 0x7FFFFF).astype(np.int32)
    return DecodeResult(values=values.flatten(),
                        exponent=exp.flatten(), mantissa=mant.flatten())


# ==================== float16 ====================

def _float16_quantize(data: np.ndarray, **_kwargs) -> tuple:
    """fp32 → fp16"""
    fp16 = data.astype(np.float16)
    return fp16, {"format": "float16", "original_shape": data.shape}


def _float16_dequantize(data: np.ndarray, meta: dict) -> DecodeResult:
    """fp16 → DecodeResult (IEEE 754 half)"""
    fp16 = data.astype(np.float16)
    bits = fp16.view(np.uint16)
    exp = ((bits >> 10) & 0x1F).astype(np.int32)
    mant = (bits & 0x3FF).astype(np.int32)
    values = fp16.astype(np.float32)
    return DecodeResult(values=values.flatten(),
                        exponent=exp.flatten(), mantissa=mant.flatten())


# ==================== bfloat16 ====================

def _bfloat16_quantize(data: np.ndarray, **_kwargs) -> tuple:
    """fp32 → bfloat16 (truncate top 16 bits)"""
    fp32_bits = data.astype(np.float32).view(np.uint32)
    bf16_bits = (fp32_bits >> 16).astype(np.uint16)
    return bf16_bits, {"format": "bfloat16", "original_shape": data.shape}


def _bfloat16_dequantize(data: np.ndarray, meta: dict) -> DecodeResult:
    """bfloat16 → DecodeResult"""
    bits = data.astype(np.uint16)
    exp = ((bits >> 7) & 0xFF).astype(np.int32)
    mant = (bits & 0x7F).astype(np.int32)
    fp32_bits = bits.astype(np.uint32) << 16
    values = fp32_bits.view(np.float32).copy()
    return DecodeResult(values=values.flatten(),
                        exponent=exp.flatten(), mantissa=mant.flatten())


# ==================== 统一注册 ====================

_IEEE_FORMATS = [
    # (name, quantize_fn, dequantize_fn, storage_dtype, bit_layout)
    ("float32",  _float32_quantize,  _float32_dequantize,  np.float32, "S0 E0*8 M0*23"),
    ("float16",  _float16_quantize,  _float16_dequantize,  np.float16, "S0 E0*5 M0*10"),
    ("bfloat16", _bfloat16_quantize, _bfloat16_dequantize, np.uint16,  "S0 E0*8 M0*7"),
]

for _name, _qfn, _dfn, _sd, _bl in _IEEE_FORMATS:
    register_block_format(BlockFormatSpec(
        info=FormatInfo(
            name=_name,
            storage_dtype=_sd,
            bit_layout=_bl,
        ),
        quantize_fn=_qfn,
        dequantize_fn=_dfn,
    ))
