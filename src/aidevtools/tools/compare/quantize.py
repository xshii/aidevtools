"""量化类型支持"""
import numpy as np
from typing import Callable, Dict, Any

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


# === 内置量化类型 ===

@register_quantize("float16")
def to_float16(data: np.ndarray, **kwargs) -> tuple:
    """fp32 → fp16"""
    return data.astype(np.float16), {}


@register_quantize("gfloat16")
def to_gfloat16(data: np.ndarray, **kwargs) -> tuple:
    """
    fp32 → gfloat16 (自定义 16 位浮点格式 demo)

    格式: 1 符号 + 8 指数 + 7 尾数 (与 bfloat16 相同)
    存储: uint16
    """
    fp32_bits = data.view(np.uint32)
    gf16_bits = (fp32_bits >> 16).astype(np.uint16)
    return gf16_bits, {"format": "gfloat16_as_uint16"}


@register_quantize("gfloat8")
def to_gfloat8(data: np.ndarray, **kwargs) -> tuple:
    """
    fp32 → gfloat8 (自定义 8 位浮点格式 demo)

    格式: 1 符号 + 4 指数 + 3 尾数
    存储: uint8
    """
    # 简单实现：截断 fp32 高 8 位
    fp32_bits = data.view(np.uint32)
    gf8_bits = (fp32_bits >> 24).astype(np.uint8)
    return gf8_bits, {"format": "gfloat8_as_uint8"}


@register_quantize("int8_symmetric")
def to_int8_symmetric(data: np.ndarray, **kwargs) -> tuple:
    """fp32 → int8 对称量化 (留空，待实现)"""
    # TODO: 实现对称量化
    # scale = np.max(np.abs(data)) / 127
    # quantized = np.round(data / scale).astype(np.int8)
    # return quantized, {"scale": scale}
    raise NotImplementedError("int8_symmetric 量化待实现")


@register_quantize("int8_asymmetric")
def to_int8_asymmetric(data: np.ndarray, **kwargs) -> tuple:
    """fp32 → int8 非对称量化 (留空，待实现)"""
    # TODO: 实现非对称量化
    raise NotImplementedError("int8_asymmetric 量化待实现")


@register_quantize("custom")
def to_custom(data: np.ndarray, **kwargs) -> tuple:
    """自定义量化 (留空，待实现)"""
    # TODO: 支持用户自定义量化
    raise NotImplementedError("custom 量化待实现")
