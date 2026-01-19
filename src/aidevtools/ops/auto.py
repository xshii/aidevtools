"""简化算子 API - 自动生成数据

用法:
    from aidevtools import ops

    ops.seed(42)
    ops.clear()

    y = ops.linear((2, 8, 64), 32)              # 自动生成 input, weight, bias
    y = ops.layernorm(y, 32, dtype="gfp16")     # 自动生成 gamma, beta
    y = ops.softmax(y)

    ops.dump("./workspace")
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
)

# 全局随机种子
_seed: int = 42
_op_counter: int = 0

# 默认 dtype
DEFAULT_DTYPE = "bfp8"


def seed(s: int):
    """设置随机种子"""
    global _seed, _op_counter
    _seed = s
    _op_counter = 0
    np.random.seed(s)


def clear():
    """清空记录"""
    global _op_counter
    _op_counter = 0
    _clear()


def dump(output_dir: str = "./workspace", format: str = "raw"):
    """导出所有 bin 文件"""
    _dump(output_dir, format)


def _get_seed():
    """获取当前算子的 seed"""
    global _op_counter
    s = _seed + _op_counter
    _op_counter += 1
    return s


# dtype 别名
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
    # 对于量化类型，先量化再反量化，模拟精度损失
    from aidevtools.formats.quantize import quantize, dequantize
    quantized, meta = quantize(x, dtype)
    return dequantize(quantized, dtype, meta)


# ==================== 算子 API ====================

def linear(
    input_shape: Union[Tuple[int, ...], np.ndarray],
    out_features: int,
    bias: bool = True,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    Linear 层: y = x @ W + b

    Args:
        input_shape: 输入 shape (自动生成) 或 输入数据 (ndarray)
        out_features: 输出特征数
        bias: 是否使用 bias
        dtype: 数据类型 (默认 bfp8)

    Returns:
        输出数据
    """
    s = _get_seed()
    np.random.seed(s)

    # 生成或使用输入
    if isinstance(input_shape, np.ndarray):
        x = input_shape
    else:
        x = np.random.randn(*input_shape).astype(np.float32)

    in_features = x.shape[-1]

    # Xavier 初始化权重
    std = np.sqrt(2.0 / (in_features + out_features))
    w = np.random.randn(in_features, out_features).astype(np.float32) * std

    # Bias
    b = np.zeros(out_features, dtype=np.float32) if bias else None

    # 量化输入
    x = _quantize_input(x, dtype)
    w = _quantize_input(w, dtype)
    if b is not None:
        b = _quantize_input(b, dtype)

    return _nn.linear(x, w, b)


def layernorm(
    x: np.ndarray,
    normalized_shape: int,
    eps: float = 1e-5,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    LayerNorm 层

    Args:
        x: 输入数据
        normalized_shape: 归一化维度大小
        eps: epsilon
        dtype: 数据类型 (默认 bfp8)

    Returns:
        输出数据
    """
    # gamma=1, beta=0
    gamma = np.ones(normalized_shape, dtype=np.float32)
    beta = np.zeros(normalized_shape, dtype=np.float32)

    # 量化
    x = _quantize_input(x, dtype)
    gamma = _quantize_input(gamma, dtype)
    beta = _quantize_input(beta, dtype)

    return _nn.layernorm(x, gamma, beta, eps)


def softmax(
    x: np.ndarray,
    axis: int = -1,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    Softmax 层

    Args:
        x: 输入数据
        axis: softmax 轴
        dtype: 数据类型 (默认 bfp8)

    Returns:
        输出数据
    """
    x = _quantize_input(x, dtype)
    return _nn.softmax(x, axis)


def relu(
    x: np.ndarray,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """ReLU 层"""
    x = _quantize_input(x, dtype)
    return _nn.relu(x)


def gelu(
    x: np.ndarray,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """GELU 层"""
    x = _quantize_input(x, dtype)
    return _nn.gelu(x)


def matmul(
    a: Union[Tuple[int, ...], np.ndarray],
    b: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
    dtype_a: str = None,
    dtype_b: str = None,
) -> np.ndarray:
    """
    矩阵乘法: y = a @ b (支持混合精度)

    Args:
        a: 输入 a (shape 或 ndarray)
        b: 输入 b (shape 或 ndarray)
        dtype: 数据类型 (默认 bfp8)，当 dtype_a/dtype_b 未指定时使用
        dtype_a: A 矩阵的数据类型 (混合精度)
        dtype_b: B 矩阵的数据类型 (混合精度)

    Returns:
        输出数据

    Example:
        # 同精度
        y = matmul((2, 8, 64), (64, 32), dtype="bfp8")

        # 混合精度: A 用 bfp8, B 用 bfp4
        y = matmul((2, 8, 64), (64, 32), dtype_a="bfp8", dtype_b="bfp4")
    """
    s = _get_seed()
    np.random.seed(s)

    # 确定各自的 dtype
    if dtype_a is None:
        dtype_a = dtype
    if dtype_b is None:
        dtype_b = dtype

    if isinstance(a, tuple):
        a = np.random.randn(*a).astype(np.float32)
    if isinstance(b, tuple):
        b = np.random.randn(*b).astype(np.float32)

    a = _quantize_input(a, dtype_a)
    b = _quantize_input(b, dtype_b)

    return _nn.matmul(a, b)


def attention(
    q: Union[Tuple[int, ...], np.ndarray],
    k: Union[Tuple[int, ...], np.ndarray] = None,
    v: Union[Tuple[int, ...], np.ndarray] = None,
    mask: np.ndarray = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    Scaled Dot-Product Attention

    Args:
        q: Query (shape 或 ndarray)
        k: Key (shape 或 ndarray)，None 则与 q 相同
        v: Value (shape 或 ndarray)，None 则与 q 相同
        mask: 注意力 mask
        dtype: 数据类型 (默认 bfp8)

    Returns:
        输出数据
    """
    s = _get_seed()
    np.random.seed(s)

    if isinstance(q, tuple):
        q = np.random.randn(*q).astype(np.float32)

    # k, v 默认与 q 相同 shape
    if k is None:
        k = np.random.randn(*q.shape).astype(np.float32)
    elif isinstance(k, tuple):
        k = np.random.randn(*k).astype(np.float32)

    if v is None:
        v = np.random.randn(*q.shape).astype(np.float32)
    elif isinstance(v, tuple):
        v = np.random.randn(*v).astype(np.float32)

    q = _quantize_input(q, dtype)
    k = _quantize_input(k, dtype)
    v = _quantize_input(v, dtype)

    return _nn.attention(q, k, v, mask)


def add(
    a: Union[Tuple[int, ...], np.ndarray],
    b: Union[Tuple[int, ...], np.ndarray] = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """加法"""
    s = _get_seed()
    np.random.seed(s)

    if isinstance(a, tuple):
        a = np.random.randn(*a).astype(np.float32)
    if b is None:
        b = np.random.randn(*a.shape).astype(np.float32)
    elif isinstance(b, tuple):
        b = np.random.randn(*b).astype(np.float32)

    a = _quantize_input(a, dtype)
    b = _quantize_input(b, dtype)

    return _nn.add(a, b)


def mul(
    a: Union[Tuple[int, ...], np.ndarray],
    b: Union[Tuple[int, ...], np.ndarray] = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """乘法"""
    s = _get_seed()
    np.random.seed(s)

    if isinstance(a, tuple):
        a = np.random.randn(*a).astype(np.float32)
    if b is None:
        b = np.random.randn(*a.shape).astype(np.float32)
    elif isinstance(b, tuple):
        b = np.random.randn(*b).astype(np.float32)

    a = _quantize_input(a, dtype)
    b = _quantize_input(b, dtype)

    return _nn.mul(a, b)


def transpose(
    x: Union[Tuple[int, ...], np.ndarray],
    axes: tuple = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    转置

    Args:
        x: 输入 (shape 或 ndarray)
        axes: 轴顺序，如 (0, 1, 3, 2)。None 则交换最后两个维度
        dtype: 数据类型 (默认 bfp8)

    Returns:
        输出数据
    """
    s = _get_seed()
    np.random.seed(s)

    if isinstance(x, tuple):
        x = np.random.randn(*x).astype(np.float32)

    x = _quantize_input(x, dtype)

    return _nn.transpose(x, axes)
