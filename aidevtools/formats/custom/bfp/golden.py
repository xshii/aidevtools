"""BFP (Block Floating Point) 真实格式 (待实现)

与 BFPP (Python Golden) 对称的接口，等待硬件对齐后实现。
"""

from typing import Tuple

import numpy as np

from aidevtools.formats.block_format import BlockFormatSpec, DecodeResult, FormatInfo, register_block_format


# ==================== BFP 核心函数 (stub) ====================


def _compute_bfp_shared_exponent(data: np.ndarray, block_size: int) -> np.ndarray:
    """
    计算每个块的共享指数

    Args:
        data: 输入数据 (float32)
        block_size: 块大小

    Returns:
        (shared_exp, blocks, pad_len)
        - shared_exp: int8, shape (num_blocks, 1)
        - blocks: float32, shape (num_blocks, block_size)
        - pad_len: 填充长度
    """
    # TODO: 实现 BFP 共享指数计算
    raise NotImplementedError("BFP _compute_bfp_shared_exponent 待实现")


def fp32_to_bfp_impl(
    data: np.ndarray, block_size: int = 16, mantissa_bits: int = 8
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    fp32 -> BFP (Block Floating Point)

    Args:
        data: 输入数据 (float32)
        block_size: 块大小
        mantissa_bits: 尾数位数

    Returns:
        (mantissas, shared_exps, meta)
        - mantissas: int8 数组，形状 (num_elements,)
        - shared_exps: int8 数组，形状 (num_blocks,)
        - meta: 元信息 dict, 必须包含:
            - "format": str
            - "block_size": int
            - "mantissa_bits": int
            - "original_shape": tuple
            - "num_blocks": int
            - "pad_len": int
    """
    # TODO: 实现 BFP 量化
    raise NotImplementedError("BFP fp32_to_bfp_impl 待实现")


def bfp_to_fp32_impl(
    mantissas: np.ndarray,
    shared_exps: np.ndarray,
    block_size: int = 16,
    mantissa_bits: int = 8,
    original_shape: tuple = None,
) -> np.ndarray:
    """
    BFP -> fp32

    Args:
        mantissas: 尾数数组 (int8)
        shared_exps: 共享指数数组 (int8)
        block_size: 块大小
        mantissa_bits: 尾数位数
        original_shape: 原始形状

    Returns:
        还原的 float32 数据
    """
    # TODO: 实现 BFP 反量化
    raise NotImplementedError("BFP bfp_to_fp32_impl 待实现")


# ==================== 量化注册 ====================


def _make_bfp_quantizer(default_block_size: int, mantissa_bits: int):
    """生成 BFP 量化函数"""

    def quantizer(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        block_size = kwargs.get("block_size", default_block_size)
        mantissas, shared_exps, meta = fp32_to_bfp_impl(
            data, block_size=block_size, mantissa_bits=mantissa_bits
        )
        packed = np.concatenate([shared_exps.astype(np.int8), mantissas.astype(np.int8)])
        return packed, meta

    return quantizer


def _make_bfp_dequantizer(default_block_size: int, mantissa_bits: int):
    """生成 BFP 反量化函数，返回 DecodeResult"""

    def dequantizer(data: np.ndarray, meta: dict) -> DecodeResult:
        block_size = meta.get("block_size", default_block_size)
        m_bits = meta.get("mantissa_bits", mantissa_bits)
        num_blocks = meta.get("num_blocks", 1)
        exp_bytes = meta.get("exp_bytes", 1)
        original_shape = meta.get("original_shape", data.shape)

        exp_total = num_blocks * exp_bytes
        shared_exps = data[:exp_total].view(np.int8)[:num_blocks]
        mantissas = data[exp_total:]

        mant = np.abs(mantissas).astype(np.int32)
        values = bfp_to_fp32_impl(mantissas, shared_exps, block_size, m_bits, original_shape)

        return DecodeResult(
            values=values.flatten(),
            exponent=shared_exps.astype(np.int32),
            mantissa=mant.flatten(),
        )

    return dequantizer


# BFP 格式配置: (name, block_size, mantissa_bits, bytes_per_block, bit_layout, description)
_BFP_FORMATS = [
    ("bfp16", 16, 8, 17,
     "E*8 S0 M0*7 S1 M1*7 S2 M2*7 S3 M3*7 S4 M4*7 S5 M5*7 S6 M6*7 S7 M7*7 "
     "S8 M8*7 S9 M9*7 S10 M10*7 S11 M11*7 S12 M12*7 S13 M13*7 S14 M14*7 S15 M15*7",
     "BFP16: 17 bytes/block (1 exp + 16*int8)"),
    ("bfp8", 32, 4, 33,
     "E*8 S0 M0*3 S1 M1*3 S2 M2*3 S3 M3*3 S4 M4*3 S5 M5*3 S6 M6*3 S7 M7*3 "
     "S8 M8*3 S9 M9*3 S10 M10*3 S11 M11*3 S12 M12*3 S13 M13*3 S14 M14*3 S15 M15*3 "
     "S16 M16*3 S17 M17*3 S18 M18*3 S19 M19*3 S20 M20*3 S21 M21*3 S22 M22*3 S23 M23*3 "
     "S24 M24*3 S25 M25*3 S26 M26*3 S27 M27*3 S28 M28*3 S29 M29*3 S30 M30*3 S31 M31*3",
     "BFP8:  33 bytes/block (1 exp + 32*int8)"),
    ("bfp4", 64, 2, 65,
     "E*8 S0 M0*1 S1 M1*1 S2 M2*1 S3 M3*1 S4 M4*1 S5 M5*1 S6 M6*1 S7 M7*1 "
     "S8 M8*1 S9 M9*1 S10 M10*1 S11 M11*1 S12 M12*1 S13 M13*1 S14 M14*1 S15 M15*1 "
     "S16 M16*1 S17 M17*1 S18 M18*1 S19 M19*1 S20 M20*1 S21 M21*1 S22 M22*1 S23 M23*1 "
     "S24 M24*1 S25 M25*1 S26 M26*1 S27 M27*1 S28 M28*1 S29 M29*1 S30 M30*1 S31 M31*1 "
     "S32 M32*1 S33 M33*1 S34 M34*1 S35 M35*1 S36 M36*1 S37 M37*1 S38 M38*1 S39 M39*1 "
     "S40 M40*1 S41 M41*1 S42 M42*1 S43 M43*1 S44 M44*1 S45 M45*1 S46 M46*1 S47 M47*1 "
     "S48 M48*1 S49 M49*1 S50 M50*1 S51 M51*1 S52 M52*1 S53 M53*1 S54 M54*1 S55 M55*1 "
     "S56 M56*1 S57 M57*1 S58 M58*1 S59 M59*1 S60 M60*1 S61 M61*1 S62 M62*1 S63 M63*1",
     "BFP4:  65 bytes/block (1 exp + 64*int8)"),
]

for _name, _block_size, _mantissa_bits, _bpb, _bl, _desc in _BFP_FORMATS:
    register_block_format(BlockFormatSpec(
        info=FormatInfo(
            name=_name,
            bytes_per_block=_bpb,
            bit_layout=_bl,
            description=_desc,
        ),
        block_size=_block_size,
        quantize_fn=_make_bfp_quantizer(_block_size, _mantissa_bits),
        dequantize_fn=_make_bfp_dequantizer(_block_size, _mantissa_bits),
    ))
