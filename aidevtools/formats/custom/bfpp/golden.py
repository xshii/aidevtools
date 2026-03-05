"""Block Floating Point (BFP) Python Golden 实现

BFP 将数据分块，每块共享一个指数，每个元素只存尾数。

参考:
- AMD Quark BFP16: https://quark.docs.amd.com/latest/onnx/tutorial_bfp16_quantization.html
- Static BFP CNN: https://github.com/os-hxfan/Static_BFP_CNN
"""

from typing import Tuple

import numpy as np

from aidevtools.formats.block_format import BlockFormatSpec, DecodeResult, FormatInfo, register_block_format


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

# BFPP 格式配置: (name, block_size, mantissa_bits, bytes_per_block, bit_layout, description)
_BFP_FORMATS = [
    ("bfpp16", 16, 8, 17,
     "E*8 S0 M0*7 S1 M1*7 S2 M2*7 S3 M3*7 S4 M4*7 S5 M5*7 S6 M6*7 S7 M7*7 "
     "S8 M8*7 S9 M9*7 S10 M10*7 S11 M11*7 S12 M12*7 S13 M13*7 S14 M14*7 S15 M15*7",
     "BFPP16: 17 bytes/block (1 exp + 16*int8)"),
    ("bfpp8", 32, 4, 33,
     "E*8 S0 M0*3 S1 M1*3 S2 M2*3 S3 M3*3 S4 M4*3 S5 M5*3 S6 M6*3 S7 M7*3 "
     "S8 M8*3 S9 M9*3 S10 M10*3 S11 M11*3 S12 M12*3 S13 M13*3 S14 M14*3 S15 M15*3 "
     "S16 M16*3 S17 M17*3 S18 M18*3 S19 M19*3 S20 M20*3 S21 M21*3 S22 M22*3 S23 M23*3 "
     "S24 M24*3 S25 M25*3 S26 M26*3 S27 M27*3 S28 M28*3 S29 M29*3 S30 M30*3 S31 M31*3",
     "BFPP8:  33 bytes/block (1 exp + 32*int8)"),
    ("bfpp4", 64, 2, 65,
     "E*8 S0 M0*1 S1 M1*1 S2 M2*1 S3 M3*1 S4 M4*1 S5 M5*1 S6 M6*1 S7 M7*1 "
     "S8 M8*1 S9 M9*1 S10 M10*1 S11 M11*1 S12 M12*1 S13 M13*1 S14 M14*1 S15 M15*1 "
     "S16 M16*1 S17 M17*1 S18 M18*1 S19 M19*1 S20 M20*1 S21 M21*1 S22 M22*1 S23 M23*1 "
     "S24 M24*1 S25 M25*1 S26 M26*1 S27 M27*1 S28 M28*1 S29 M29*1 S30 M30*1 S31 M31*1 "
     "S32 M32*1 S33 M33*1 S34 M34*1 S35 M35*1 S36 M36*1 S37 M37*1 S38 M38*1 S39 M39*1 "
     "S40 M40*1 S41 M41*1 S42 M42*1 S43 M43*1 S44 M44*1 S45 M45*1 S46 M46*1 S47 M47*1 "
     "S48 M48*1 S49 M49*1 S50 M50*1 S51 M51*1 S52 M52*1 S53 M53*1 S54 M54*1 S55 M55*1 "
     "S56 M56*1 S57 M57*1 S58 M58*1 S59 M59*1 S60 M60*1 S61 M61*1 S62 M62*1 S63 M63*1",
     "BFPP4:  65 bytes/block (1 exp + 64*int8)"),
]


def _make_bfp_quantizer(default_block_size: int, mantissa_bits: int):
    """生成 BFP 量化函数"""

    def quantizer(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        block_size = kwargs.get("block_size", default_block_size)
        mantissas, shared_exps, meta = fp32_to_bfp(
            data, block_size=block_size, mantissa_bits=mantissa_bits
        )
        meta["exp_bytes"] = 1  # BFPP: 每个 shared exponent 占 1 byte
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
        values = bfp_to_fp32(mantissas, shared_exps, block_size, m_bits, original_shape)

        return DecodeResult(
            values=values.flatten(),
            exponent=shared_exps.astype(np.int32),
            mantissa=mant.flatten(),
        )

    return dequantizer


# 批量注册 BFPP 格式 (通过 block_format 统一注册)
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
