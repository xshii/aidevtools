"""算子 Profile 数据结构"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


def dtype_bytes(dtype: str) -> float:
    """获取 dtype 字节数"""
    mapping = {
        "fp32": 4, "float32": 4,
        "fp16": 2, "float16": 2,
        "bf16": 2, "bfloat16": 2,
        "int8": 1,
        "int4": 0.5,
        "int32": 4,
    }
    return mapping.get(str(dtype).lower(), 2)


@dataclass
class MatMulDtypeConfig:
    """MatMul 混合精度配置"""
    dtype_a: str = "fp16"      # 左矩阵
    dtype_b: str = "fp16"      # 右矩阵
    dtype_acc: str = "fp32"    # 累加器
    dtype_out: str = "fp16"    # 输出

    def __repr__(self):
        return f"{self.dtype_a}×{self.dtype_b}→{self.dtype_out}"


@dataclass
class OpProfile:
    """算子性能 Profile"""

    # 基础信息
    name: str = ""                     # matmul_0
    op_type: str = ""                  # matmul

    # 形状
    shapes: Dict[str, Any] = field(default_factory=dict)

    # 数据类型
    dtype: str = "fp16"
    dtype_config: Optional[MatMulDtypeConfig] = None

    # 计算
    flops: int = 0                     # 浮点运算次数
    compute_unit: str = "vector"       # "cube" | "vector"

    # 访存（细分）
    input_bytes: int = 0               # 输入（有数据依赖）
    weight_bytes: int = 0              # 权重（可预取）
    output_bytes: int = 0              # 输出
    workspace_bytes: int = 0           # 中间变量

    # 访存模式
    memory_pattern: str = "sequential"  # "sequential" | "strided" | "random"

    @property
    def total_bytes(self) -> int:
        """总访存量"""
        return self.input_bytes + self.weight_bytes + self.output_bytes + self.workspace_bytes

    @property
    def arithmetic_intensity(self) -> float:
        """计算访存比 (FLOPs/Byte)"""
        return self.flops / self.total_bytes if self.total_bytes > 0 else 0

    def __repr__(self):
        return (f"OpProfile({self.name}, flops={self.flops/1e9:.2f}G, "
                f"bytes={self.total_bytes/1e6:.1f}MB, AI={self.arithmetic_intensity:.1f})")


# ============================================================
# 各算子的 Profile 函数
# ============================================================

def profile_matmul(a: np.ndarray, b: np.ndarray,
                   dtype_config: MatMulDtypeConfig = None) -> OpProfile:
    """MatMul Profile: C = A @ B"""
    # 提取形状
    if a.ndim == 1:
        M, K = 1, a.shape[0]
    else:
        M, K = a.shape[-2:]

    if b.ndim == 1:
        K2, N = b.shape[0], 1
    else:
        K2, N = b.shape[-2:]

    batch = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1

    # 数据类型
    if dtype_config:
        dtype_a = dtype_config.dtype_a
        dtype_b = dtype_config.dtype_b
        dtype_out = dtype_config.dtype_out
    else:
        dtype_a = dtype_b = dtype_out = "fp16"

    # FLOPs: 2MKN (乘 + 加)
    flops = batch * 2 * M * K * N

    # 访存量
    input_bytes = batch * M * K * dtype_bytes(dtype_a)
    weight_bytes = K * N * dtype_bytes(dtype_b)
    output_bytes = batch * M * N * dtype_bytes(dtype_out)

    return OpProfile(
        op_type="matmul",
        shapes={"batch": batch, "M": M, "K": K, "N": N},
        dtype=dtype_a,
        dtype_config=dtype_config,
        flops=int(flops),
        compute_unit="cube",
        input_bytes=int(input_bytes),
        weight_bytes=int(weight_bytes),
        output_bytes=int(output_bytes),
        memory_pattern="sequential",
    )


def profile_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      eps: float = 1e-5) -> OpProfile:
    """LayerNorm Profile"""
    batch = int(np.prod(x.shape[:-1]))
    hidden = x.shape[-1]
    n = x.size
    dtype = "fp16"

    # FLOPs: ~8 ops/element
    flops = 8 * n

    # 访存量
    db = dtype_bytes(dtype)
    input_bytes = x.nbytes
    weight_bytes = gamma.nbytes + beta.nbytes
    output_bytes = x.nbytes
    workspace_bytes = batch * 2 * db  # mean + var

    return OpProfile(
        op_type="layernorm",
        shapes={"batch": batch, "hidden": hidden},
        dtype=dtype,
        flops=int(flops),
        compute_unit="vector",
        input_bytes=int(input_bytes),
        weight_bytes=int(weight_bytes),
        output_bytes=int(output_bytes),
        workspace_bytes=int(workspace_bytes),
        memory_pattern="sequential",
    )


def profile_softmax(x: np.ndarray, axis: int = -1) -> OpProfile:
    """Softmax Profile"""
    batch = int(np.prod(x.shape[:axis])) if axis != 0 else 1
    seq = x.shape[axis] if axis < x.ndim else x.shape[-1]
    dtype = "fp16"

    # FLOPs: ~5 ops/element
    flops = 5 * x.size

    # 访存量
    db = dtype_bytes(dtype)
    input_bytes = x.nbytes
    output_bytes = x.nbytes
    workspace_bytes = batch * 2 * db  # max + sum

    return OpProfile(
        op_type="softmax",
        shapes={"batch": batch, "seq": seq},
        dtype=dtype,
        flops=int(flops),
        compute_unit="vector",
        input_bytes=int(input_bytes),
        weight_bytes=0,
        output_bytes=int(output_bytes),
        workspace_bytes=int(workspace_bytes),
        memory_pattern="sequential",
    )


def profile_transpose(x: np.ndarray, axes: tuple = None) -> OpProfile:
    """Transpose Profile"""
    shape = x.shape
    ndim = x.ndim
    dtype = "fp16"

    if axes is None:
        axes = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)

    # FLOPs: 几乎为 0
    flops = 0

    # 访存模式: 检查是否是恒等变换
    # 如果 axes 不是 (0, 1, 2, ..., n-1)，则是跨步访问
    identity_axes = tuple(range(ndim))
    if axes != identity_axes:
        memory_pattern = "strided"
    else:
        memory_pattern = "sequential"

    return OpProfile(
        op_type="transpose",
        shapes={"shape": shape, "axes": axes},
        dtype=dtype,
        flops=flops,
        compute_unit="vector",
        input_bytes=int(x.nbytes),
        weight_bytes=0,
        output_bytes=int(x.nbytes),
        memory_pattern=memory_pattern,
    )


def profile_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                      mask: np.ndarray = None) -> OpProfile:
    """Attention Profile: softmax(Q @ K^T / sqrt(d)) @ V"""
    # 提取形状
    if q.ndim == 3:
        batch, seq_q, head_dim = q.shape
        heads = 1
    else:  # ndim == 4
        batch, heads, seq_q, head_dim = q.shape

    seq_kv = k.shape[-2]
    dtype = "fp16"
    db = dtype_bytes(dtype)

    # FLOPs 分解
    # QK: batch * heads * 2 * seq_q * head_dim * seq_kv
    qk_flops = batch * heads * 2 * seq_q * head_dim * seq_kv
    # Softmax: batch * heads * 5 * seq_q * seq_kv
    soft_flops = batch * heads * 5 * seq_q * seq_kv
    # SV: batch * heads * 2 * seq_q * seq_kv * head_dim
    sv_flops = batch * heads * 2 * seq_q * seq_kv * head_dim

    total_flops = qk_flops + soft_flops + sv_flops

    # 访存量
    input_bytes = (q.nbytes + k.nbytes + v.nbytes)
    output_bytes = batch * heads * seq_q * head_dim * db
    # 中间结果 (QK scores)
    workspace_bytes = batch * heads * seq_q * seq_kv * db

    return OpProfile(
        op_type="attention",
        shapes={
            "batch": batch, "heads": heads,
            "seq_q": seq_q, "seq_kv": seq_kv, "head_dim": head_dim
        },
        dtype=dtype,
        flops=int(total_flops),
        compute_unit="cube",  # 主要是 matmul
        input_bytes=int(input_bytes),
        weight_bytes=0,
        output_bytes=int(output_bytes),
        workspace_bytes=int(workspace_bytes),
        memory_pattern="sequential",
    )


def profile_gelu(x: np.ndarray) -> OpProfile:
    """GELU Profile"""
    dtype = "fp16"
    # GELU 约 10 ops/element (tanh 较贵)
    flops = 10 * x.size

    return OpProfile(
        op_type="gelu",
        shapes={"size": x.size},
        dtype=dtype,
        flops=int(flops),
        compute_unit="vector",
        input_bytes=int(x.nbytes),
        weight_bytes=0,
        output_bytes=int(x.nbytes),
        memory_pattern="sequential",
    )


def profile_add(a: np.ndarray, b: np.ndarray) -> OpProfile:
    """Add Profile"""
    dtype = "fp16"
    flops = a.size

    return OpProfile(
        op_type="add",
        shapes={"size": a.size},
        dtype=dtype,
        flops=int(flops),
        compute_unit="vector",
        input_bytes=int(a.nbytes + b.nbytes),
        weight_bytes=0,
        output_bytes=int(a.nbytes),
        memory_pattern="sequential",
    )


# Profile 函数注册表
PROFILE_FUNCS = {
    "matmul": profile_matmul,
    "linear": profile_matmul,
    "layernorm": profile_layernorm,
    "softmax": profile_softmax,
    "transpose": profile_transpose,
    "attention": profile_attention,
    "gelu": profile_gelu,
    "add": profile_add,
}


def get_profile_func(op_type: str):
    """获取 profile 函数"""
    return PROFILE_FUNCS.get(op_type)
