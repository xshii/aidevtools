"""量化类型支持"""
import numpy as np
from typing import Callable, Dict

# 量化类型注册表
_quantize_registry: Dict[str, Callable] = {}


def register_quantize(name: str):
    """
    注册量化转换函数

    示例:
        @register_quantize("int8_symmetric")
        def to_int8_symmetric(data: np.ndarray, **kwargs) -> np.ndarray:
            scale = np.max(np.abs(data)) / 127
            return np.round(data / scale).astype(np.int8), {"scale": scale}
    """
    def decorator(func: Callable):
        _quantize_registry[name] = func
        return func
    return decorator


def get_quantize(name: str) -> Callable:
    """获取量化函数"""
    if name not in _quantize_registry:
        raise ValueError(f"未知量化类型: {name}, 可用: {list(_quantize_registry.keys())}")
    return _quantize_registry[name]


def list_quantize() -> list:
    """列出所有注册的量化类型"""
    return list(_quantize_registry.keys())


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
        from aidevtools.formats.custom.gfloat.golden import from_gfloat16, from_gfloat8, from_gfloat4
        if qtype == "gfloat16":
            return from_gfloat16(data, meta.get("original_shape"))
        elif qtype == "gfloat8":
            return from_gfloat8(data, meta.get("original_shape"))
        else:
            return from_gfloat4(data, meta.get("original_shape"))

    if qtype in _dequantize_registry:
        return _dequantize_registry[qtype](data, meta)

    raise ValueError(f"未知量化类型或无法反量化: {qtype}")


# === 内置量化类型 ===

@register_quantize("float16")
def to_float16(data: np.ndarray, **kwargs) -> tuple:
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


# 导入自定义格式以触发注册
from aidevtools.formats.custom.gfloat import golden as _gfloat_golden
from aidevtools.formats.custom.bfp import golden as _bfp_golden
