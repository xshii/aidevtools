"""量化类型支持"""
from typing import Callable, Dict

import numpy as np

# 从注册表模块导入 (避免循环导入)
from aidevtools.formats._quantize_registry import (
    get_quantize,
    list_quantize as list_quantize,
    register_quantize,
)


def quantize(data: np.ndarray, qtype: str, **kwargs) -> tuple:
    """
    量化数据

    Args:
        data: 输入数据 (fp32)
        qtype: 量化类型名称
        **kwargs: 量化参数

    Returns:
        (quantized_data, meta_info)
    """
    func = get_quantize(qtype)
    return func(data, **kwargs)


def simulate_quantize(data: np.ndarray, qtype: str, **kwargs) -> np.ndarray:
    """
    模拟量化精度损失: quantize -> dequantize

    用于在 golden 计算中模拟量化带来的精度损失。

    Args:
        data: 输入数据 (fp32)
        qtype: 量化类型名称 (bfp4, bfp8, bfp16, gfloat4, gfloat8, gfloat16, float16)
        **kwargs: 量化参数

    Returns:
        还原后的 fp32 数据 (带精度损失)

    Example:
        >>> x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> x_lossy = simulate_quantize(x, "bfp4")  # 模拟 bfp4 量化损失
    """
    if qtype == "float32":
        return data.astype(np.float32)

    packed, meta = quantize(data, qtype, **kwargs)
    return dequantize(packed, qtype, meta)


# 反量化注册表
_dequantize_registry: Dict[str, Callable] = {}


def register_dequantize(name: str):
    """注册反量化函数"""
    def decorator(func: Callable):
        _dequantize_registry[name] = func
        return func
    return decorator


def dequantize(data: np.ndarray, qtype: str, meta: dict = None) -> np.ndarray:
    """
    反量化数据

    Args:
        data: 量化后的数据
        qtype: 量化类型名称
        meta: 量化元信息

    Returns:
        还原的 fp32 数据
    """
    meta = meta or {}

    if qtype == "float32":
        return data.astype(np.float32)

    if qtype == "float16":
        return data.astype(np.float32)

    if qtype in ("bfp16", "bfp8", "bfp4"):
        from aidevtools.formats.custom.bfp.golden import bfp_to_fp32
        # 从打包数据中提取 mantissas 和 shared_exps
        default_block = {"bfp16": 16, "bfp8": 32, "bfp4": 64}.get(qtype, 16)
        default_mantissa = {"bfp16": 8, "bfp8": 4, "bfp4": 2}.get(qtype, 8)
        block_size = meta.get("block_size", default_block)
        mantissa_bits = meta.get("mantissa_bits", default_mantissa)
        num_blocks = meta.get("num_blocks", 1)
        original_shape = meta.get("original_shape", data.shape)

        # 打包格式: [shared_exps..., mantissas...]
        shared_exps = data[:num_blocks]
        mantissas = data[num_blocks:]

        return bfp_to_fp32(mantissas, shared_exps, block_size, mantissa_bits, original_shape)

    if qtype in ("gfloat16", "gfloat8", "gfloat4"):
        from aidevtools.formats.custom.gfloat.golden import (
            from_gfloat4,
            from_gfloat8,
            from_gfloat16,
        )
        if qtype == "gfloat16":
            return from_gfloat16(data, meta.get("original_shape"))
        if qtype == "gfloat8":
            return from_gfloat8(data, meta.get("original_shape"))
        return from_gfloat4(data, meta.get("original_shape"))

    if qtype in _dequantize_registry:
        return _dequantize_registry[qtype](data, meta)

    raise ValueError(f"未知量化类型或无法反量化: {qtype}")


# === 内置量化类型 ===

@register_quantize("float16")
def to_float16(data: np.ndarray, **_kwargs) -> tuple:
    """fp32 → fp16"""
    return data.astype(np.float16), {}


@register_quantize("int8_symmetric")
def to_int8_symmetric(data: np.ndarray, **kwargs) -> tuple:
    """fp32 → int8 对称量化 (留空，待实现)"""
    raise NotImplementedError("int8_symmetric 量化待实现")


@register_quantize("int8_asymmetric")
def to_int8_asymmetric(data: np.ndarray, **kwargs) -> tuple:
    """fp32 → int8 非对称量化 (留空，待实现)"""
    raise NotImplementedError("int8_asymmetric 量化待实现")


def generate_fake_dut(
    reference: np.ndarray,
    qtype: str = "bfp8",
    noise_level: float = 0.001,
) -> np.ndarray:
    """
    生成模拟 DUT 数据

    用于 demo 和测试，模拟真实硬件的量化处理流程：
    1. 从 reference (fp32 精确值) 开始
    2. 应用量化/反量化，模拟 DUT 的格式计算
    3. 添加小噪声，模拟 DUT 的计算误差

    Args:
        reference: fp32 精确计算结果
        qtype: 量化格式 (bfp4, bfp8, bfp16, gfloat4, gfloat8, gfloat16)
        noise_level: 噪声水平

    Returns:
        模拟的 DUT 输出 (fp32)

    Example:
        >>> ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        >>> dut = generate_fake_dut(ref, qtype="bfp8", noise_level=0.001)
    """
    # Step 1: 对 reference 进行量化/反量化，模拟 DUT 的格式处理
    dut_quantized = simulate_quantize(reference.astype(np.float32), qtype)

    # Step 2: 添加小噪声，模拟 DUT 计算误差
    if noise_level > 0:
        noise = np.random.randn(*dut_quantized.shape).astype(np.float32) * noise_level
        dut_quantized = dut_quantized + noise

    return dut_quantized


# 注意: 自定义格式的注册已移至 formats/__init__.py
# 这样可以避免循环导入 (quantize <-> gfloat/bfp golden)
