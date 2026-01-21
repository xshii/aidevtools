"""简化算子 API - 基于注册表元数据自动生成

通过 @register_op 的 auto_gen 配置自动生成简化 API。

用法:
    from aidevtools import ops

    ops.seed(42)
    ops.clear()

    y = ops.linear((2, 8, 64), out_features=32)  # 自动生成 weight, bias
    y = ops.layernorm(y)                          # 自动生成 gamma=1, beta=0
    y = ops.softmax(y)
    y = ops.relu(y)

    ops.dump("./workspace")

auto_gen 策略说明:
    - "input": 主输入，可以是 tuple(shape) 或 ndarray
    - "random": 随机初始化，shape 与第一个输入相同
    - "ones:-1": 全1数组，-1 表示取第一个输入的最后一维
    - "zeros:-1": 全0数组
    - "xavier": Xavier 初始化 (用于 weight)
"""
import numpy as np
from typing import Tuple, Union, Callable, Any, Dict

from aidevtools.ops import nn as _nn
from aidevtools.ops.base import (
    clear as _clear,
    dump as _dump,
    get_records,
    set_golden_mode,
    get_golden_mode,
)
from aidevtools.ops.registry import get_op_meta, get_op_instance, list_ops

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
# 参数生成器
# ============================================================

def _generate_param(
    strategy: str,
    input_arr: np.ndarray,
    out_features: int = None,
    **kwargs
) -> np.ndarray:
    """
    根据策略生成参数

    Args:
        strategy: 生成策略 (ones:-1, zeros:-1, xavier, random, etc.)
        input_arr: 第一个输入数组 (用于推断 shape)
        out_features: 输出特征数 (用于 linear)

    Returns:
        生成的参数数组
    """
    if strategy == "input":
        return input_arr

    # 解析策略
    parts = strategy.split(":")
    gen_type = parts[0]
    dim_spec = parts[1] if len(parts) > 1 else "-1"

    # 计算 shape
    if dim_spec == "-1":
        shape = (input_arr.shape[-1],)
    elif dim_spec.isdigit() or (dim_spec.startswith("-") and dim_spec[1:].isdigit()):
        dim = int(dim_spec)
        shape = (input_arr.shape[dim],)
    else:
        # 可能是 "out_features" 这样的变量名
        shape = (out_features,) if out_features else (input_arr.shape[-1],)

    # 生成数据
    if gen_type == "ones":
        return np.ones(shape, dtype=np.float32)
    elif gen_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif gen_type == "random":
        return np.random.randn(*shape).astype(np.float32)
    elif gen_type == "uniform":
        # 类似 PyTorch Linear bias 初始化: uniform(-bound, bound)
        in_features = input_arr.shape[-1]
        bound = 1.0 / np.sqrt(in_features)
        return np.random.uniform(-bound, bound, shape).astype(np.float32)
    elif gen_type == "xavier":
        in_features = input_arr.shape[-1]
        out_feat = out_features if out_features else in_features
        std = np.sqrt(2.0 / (in_features + out_feat))
        return np.random.randn(in_features, out_feat).astype(np.float32) * std
    else:
        # 默认随机
        return np.random.randn(*shape).astype(np.float32)


# ============================================================
# 通用算子调用器
# ============================================================

def _call_op_auto(
    op_name: str,
    first_input: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
    out_features: int = None,
    **kwargs
) -> np.ndarray:
    """
    通用算子调用器 - 基于 auto_gen 配置自动生成参数

    Args:
        op_name: 算子名称
        first_input: 第一个输入 (shape 或 array)
        dtype: 数据类型
        out_features: 输出特征数 (用于 linear 等)
        **kwargs: 其他参数 (覆盖自动生成)
    """
    s = _get_seed()
    np.random.seed(s)

    # 获取元数据
    meta = get_op_meta(op_name)
    if meta is None:
        raise ValueError(f"未知算子: {op_name}")

    op_instance = get_op_instance(op_name)

    # 生成第一个输入
    input_arr = _maybe_generate(first_input)

    # 根据 auto_gen 生成其他参数
    args = []
    for param_name in meta.inputs:
        if param_name in kwargs:
            # 用户提供的参数
            val = kwargs.pop(param_name)
            if isinstance(val, (tuple, np.ndarray)):
                val = _maybe_generate(val)
                val = _quantize_input(val, dtype)
            args.append(val)
        else:
            # 自动生成
            strategy = meta.auto_gen.get(param_name, "random")
            if strategy == "input":
                arr = _quantize_input(input_arr, dtype)
            else:
                arr = _generate_param(strategy, input_arr, out_features=out_features)
                arr = _quantize_input(arr, dtype)
            args.append(arr)

    # 处理 optional 参数
    for opt_name in meta.optional:
        if opt_name in kwargs:
            val = kwargs.pop(opt_name)
            if isinstance(val, np.ndarray):
                val = _quantize_input(val, dtype)
            kwargs[opt_name] = val

    return op_instance(*args, **kwargs)


# ============================================================
# 特殊算子 (需要额外参数)
# ============================================================

def linear(
    input_shape: Union[Tuple[int, ...], np.ndarray],
    out_features: int,
    bias: bool = True,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """
    Linear 层: y = x @ W + b

    Args:
        input_shape: 输入 shape 或数据
        out_features: 输出特征数
        bias: 是否使用 bias
        dtype: 数据类型
    """
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(input_shape)
    in_features = x.shape[-1]

    # Xavier 初始化
    std = np.sqrt(2.0 / (in_features + out_features))
    w = np.random.randn(in_features, out_features).astype(np.float32) * std
    # bias: 均匀分布 [-bound, bound]，类似 PyTorch
    bound = 1.0 / np.sqrt(in_features)
    b = np.random.uniform(-bound, bound, out_features).astype(np.float32) if bias else None

    x = _quantize_input(x, dtype)
    w = _quantize_input(w, dtype)
    if b is not None:
        b = _quantize_input(b, dtype)

    return _nn.linear(x, w, b)


def matmul(
    a: Union[Tuple[int, ...], np.ndarray],
    b: Union[Tuple[int, ...], np.ndarray],
    dtype: str = DEFAULT_DTYPE,
    dtype_a: str = None,
    dtype_b: str = None,
) -> np.ndarray:
    """矩阵乘法 (支持混合精度)"""
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


def attention(
    q: Union[Tuple[int, ...], np.ndarray],
    k: Union[Tuple[int, ...], np.ndarray] = None,
    v: Union[Tuple[int, ...], np.ndarray] = None,
    mask: np.ndarray = None,
    dtype: str = DEFAULT_DTYPE,
) -> np.ndarray:
    """Scaled Dot-Product Attention"""
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
    """Embedding 层"""
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
    """转置"""
    s = _get_seed()
    np.random.seed(s)

    x = _maybe_generate(x)
    x = _quantize_input(x, dtype)

    return _nn.transpose(x, axes)


# ============================================================
# 自动生成的算子 API (基于 auto_gen)
# ============================================================

def _make_auto_op(op_name: str) -> Callable:
    """基于 auto_gen 配置生成算子包装函数"""
    meta = get_op_meta(op_name)

    def wrapper(
        x: Union[Tuple[int, ...], np.ndarray],
        dtype: str = DEFAULT_DTYPE,
        **kwargs
    ) -> np.ndarray:
        return _call_op_auto(op_name, x, dtype=dtype, **kwargs)

    wrapper.__name__ = op_name
    wrapper.__doc__ = meta.description if meta else f"{op_name} 算子"
    return wrapper


# 自动注册简单算子 (基于 auto_gen)
_SIMPLE_OPS = [
    "relu", "gelu", "sigmoid", "tanh",  # 单输入激活
    "softmax",  # 带 axis 参数
    "layernorm", "batchnorm",  # 归一化 (gamma/beta 自动生成)
    "add", "mul", "div",  # 二元运算
]

for _op_name in _SIMPLE_OPS:
    if get_op_meta(_op_name) is not None:
        globals()[_op_name] = _make_auto_op(_op_name)


# ============================================================
# 动态算子访问 (兜底)
# ============================================================

def __getattr__(name: str) -> Callable:
    """
    动态获取算子

    如果算子已注册，自动生成基于 auto_gen 的包装函数。
    """
    meta = get_op_meta(name)
    if meta is None:
        raise AttributeError(f"模块 'ops' 没有属性 '{name}'")

    return _make_auto_op(name)
