"""简化算子 API - 基于 shape 自动生成测试数据

此模块用于快速生成测试数据：
    from aidevtools import ops

    ops.seed(42)
    ops.clear()

    y = ops.matmul((2, 8, 64), (64, 32), dtype="bfp8")
    y = ops.layernorm(y, dtype="bfp8")
    y = ops.softmax(y, dtype="bfp8")

    ops.dump("./workspace")

如需 PyTorch 风格 API，请使用:
    from aidevtools import F

    y = F.linear(x, weight, bias)
    y = F.relu(y)
"""
import numpy as np
from typing import Tuple, Union

from aidevtools.ops import nn as _nn
from aidevtools.ops.base import (
    clear as _clear,
    dump as _dump,
    get_records,
    set_golden_mode,
    get_golden_mode,
    # Profile API (用于 Paper Analysis)
    get_profiles,
    set_profile_enabled,
    get_profile_enabled,
)

# 全局随机种子
_seed: int = 42
_op_counter: int = 0

# 默认 dtype
DEFAULT_DTYPE = "bfp8"


def seed(s: int) -> None:
    """设置随机种子"""
    global _seed, _op_counter
    _seed = s
    _op_counter = 0
    np.random.seed(s)


def clear() -> None:
    """清空记录"""
    global _op_counter
    _op_counter = 0
    _clear()


def dump(output_dir: str = "./workspace", format: str = "raw") -> None:
    """导出所有 bin 文件"""
    _dump(output_dir, format)


def get_seed() -> int:
    """获取当前种子值"""
    return _seed


def _get_seed() -> int:
    """获取当前算子的 seed"""
    global _op_counter
    s = _seed + _op_counter
    _op_counter += 1
    return s


# ============================================================
# dtype 处理
# ============================================================

_DTYPE_ALIASES = {
    "fp32": "float32",
    "fp16": "float16",
    "gfp16": "gfloat16",
    "gfp8": "gfloat8",
    "gfp4": "gfloat4",
    "bfp16": "bfp16",
    "bfp8": "bfp8",
    "bfp4": "bfp4",
}


def _normalize_dtype(dtype: str) -> str:
    """标准化 dtype 名称"""
    return _DTYPE_ALIASES.get(dtype, dtype)


def _quantize_input(x: np.ndarray, dtype: str) -> np.ndarray:
    """对输入进行量化（如果需要）"""
    dtype = _normalize_dtype(dtype)
    if dtype == "float32":
        return x.astype(np.float32)
    from aidevtools.formats.quantize import quantize, dequantize
    quantized, meta = quantize(x, dtype)
    return dequantize(quantized, dtype, meta)


def _maybe_generate(x: Union[Tuple[int, ...], np.ndarray]) -> np.ndarray:
    """如果是 tuple 则生成随机数据，否则直接返回"""
    if isinstance(x, tuple):
        return np.random.randn(*x).astype(np.float32)
    return np.asarray(x, dtype=np.float32)


# ============================================================
# 简化 API (基于 shape 自动生成数据)
# ============================================================

def matmul(
    a: Union[Tuple[int, ...], np.ndarray],
    b: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
    dtype_a: str = None,
    dtype_b: str = None,
) -> np.ndarray:
    """矩阵乘法 (支持 shape 自动生成)

    Args:
        a: 输入 A (shape 或 array)
        b: 输入 B (shape 或 array)
        dtype: 默认量化类型
        dtype_a: A 的量化类型
        dtype_b: B 的量化类型

    Example:
        y = ops.matmul((2, 8, 64), (64, 32), dtype="bfp8")
    """
    s = _get_seed()
    np.random.seed(s)

    if dtype_a is None:
        dtype_a = dtype
    if dtype_b is None:
        dtype_b = dtype

    a = _maybe_generate(a)
    b = _maybe_generate(b)

    a = _quantize_input(a, dtype_a)
    b = _quantize_input(b, dtype_b)

    return _nn.matmul(a, b)


def linear(
    input_shape: Union[Tuple[int, ...], np.ndarray],
    out_features: int,
    bias: bool = True,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Linear 层 (支持 shape 自动生成)

    Args:
        input_shape: 输入 shape 或数据
        out_features: 输出特征数
        bias: 是否使用 bias
        dtype: 量化类型

    Example:
        y = ops.linear((2, 8, 64), out_features=32, dtype="bfp8")
    """
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(input_shape)
    in_features = x.shape[-1]

    # Xavier 初始化
    std = np.sqrt(2.0 / (in_features + out_features))
    w = np.random.randn(in_features, out_features).astype(np.float32) * std
    # bias: 均匀分布
    bound = 1.0 / np.sqrt(in_features)
    b = np.random.uniform(-bound, bound, out_features).astype(np.float32) if bias else None

    x = _quantize_input(x, dtype)
    w = _quantize_input(w, dtype)
    if b is not None:
        b = _quantize_input(b, dtype)

    return _nn.linear(x, w, b)


def layernorm(
    x: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
    eps: float = 1e-5,
) -> np.ndarray:
    """LayerNorm (gamma/beta 自动生成为 1/0)

    Args:
        x: 输入 (shape 或 array)
        dtype: 量化类型
        eps: epsilon

    Example:
        y = ops.layernorm((2, 8, 64), dtype="bfp8")
    """
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(x)
    hidden = x.shape[-1]

    gamma = np.ones(hidden, dtype=np.float32)
    beta = np.zeros(hidden, dtype=np.float32)

    x = _quantize_input(x, dtype)

    return _nn.layernorm(x, gamma, beta, eps=eps)


def softmax(
    x: Union[Tuple[int, ...], np.ndarray],
    axis: int = -1,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Softmax (支持 shape 自动生成)

    Args:
        x: 输入 (shape 或 array)
        axis: 计算维度
        dtype: 量化类型

    Example:
        y = ops.softmax((2, 8, 64), dtype="bfp8")
    """
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(x)
    x = _quantize_input(x, dtype)

    return _nn.softmax(x, axis=axis)


def relu(
    x: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """ReLU (支持 shape 自动生成)"""
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(x)
    x = _quantize_input(x, dtype)

    return _nn.relu(x)


def gelu(
    x: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """GELU (支持 shape 自动生成)"""
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(x)
    x = _quantize_input(x, dtype)

    return _nn.gelu(x)


def attention(
    q: Union[Tuple[int, ...], np.ndarray],
    k: Union[Tuple[int, ...], np.ndarray] = None,
    v: Union[Tuple[int, ...], np.ndarray] = None,
    mask: np.ndarray = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Scaled Dot-Product Attention (支持 shape 自动生成)"""
    s = _get_seed()
    np.random.seed(s)

    q = _maybe_generate(q)
    if k is None:
        k = np.random.randn(*q.shape).astype(np.float32)
    else:
        k = _maybe_generate(k)
    if v is None:
        v = np.random.randn(*q.shape).astype(np.float32)
    else:
        v = _maybe_generate(v)

    q = _quantize_input(q, dtype)
    k = _quantize_input(k, dtype)
    v = _quantize_input(v, dtype)

    return _nn.attention(q, k, v, mask)


def embedding(
    input_ids: np.ndarray,
    vocab_size: int,
    embed_dim: int,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Embedding 层 (自动生成 embed table)"""
    s = _get_seed()
    np.random.seed(s)

    embed_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02
    embed_table = _quantize_input(embed_table, dtype)

    return _nn.embedding(input_ids, embed_table)


def transpose(
    x: Union[Tuple[int, ...], np.ndarray],
    axes: tuple = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """转置 (支持 shape 自动生成)"""
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(x)
    x = _quantize_input(x, dtype)

    return _nn.transpose(x, axes)
