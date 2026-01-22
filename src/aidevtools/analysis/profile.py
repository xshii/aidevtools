"""算子 Profile 数据结构"""

from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


# dtype 字节数映射
_DTYPE_BYTES = {
    "fp32": 4, "float32": 4,
    "fp16": 2, "float16": 2,
    "bf16": 2, "bfloat16": 2,
    "int8": 1, "int4": 0.5, "int32": 4,
}


def dtype_bytes(dtype: str) -> float:
    """获取 dtype 字节数"""
    return _DTYPE_BYTES.get(str(dtype).lower(), 2)


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
# Profile 工厂函数
# ============================================================

def _profile(op_type: str, *, shapes: dict, flops: int,
             input_bytes: int, output_bytes: int, weight_bytes: int = 0,
             workspace_bytes: int = 0, compute_unit: str = "vector",
             memory_pattern: str = "sequential", dtype: str = "fp16") -> OpProfile:
    """创建 OpProfile 的内部工厂函数"""
    return OpProfile(
        op_type=op_type, shapes=shapes, dtype=dtype, flops=int(flops),
        compute_unit=compute_unit, input_bytes=int(input_bytes),
        weight_bytes=int(weight_bytes), output_bytes=int(output_bytes),
        workspace_bytes=int(workspace_bytes), memory_pattern=memory_pattern,
    )


def profile_matmul(a: np.ndarray, b: np.ndarray, dtype: str = "fp16") -> OpProfile:
    """MatMul Profile: C = A @ B"""
    M, K = (1, a.shape[0]) if a.ndim == 1 else a.shape[-2:]
    K2, N = (b.shape[0], 1) if b.ndim == 1 else b.shape[-2:]
    batch = int(np.prod(a.shape[:-2])) if a.ndim > 2 else 1
    db = dtype_bytes(dtype)
    return _profile("matmul", shapes={"batch": batch, "M": M, "K": K, "N": N},
                    flops=batch * 2 * M * K * N, compute_unit="cube",
                    input_bytes=batch * M * K * db, weight_bytes=K * N * db,
                    output_bytes=batch * M * N * db, dtype=dtype)


def profile_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      eps: float = 1e-5) -> OpProfile:
    """LayerNorm Profile"""
    batch, hidden = int(np.prod(x.shape[:-1])), x.shape[-1]
    return _profile("layernorm", shapes={"batch": batch, "hidden": hidden},
                    flops=8 * x.size, input_bytes=x.nbytes,
                    weight_bytes=gamma.nbytes + beta.nbytes, output_bytes=x.nbytes,
                    workspace_bytes=batch * 2 * 2)  # mean + var


def profile_softmax(x: np.ndarray, axis: int = -1) -> OpProfile:
    """Softmax Profile"""
    batch = int(np.prod(x.shape[:axis])) if axis != 0 else 1
    seq = x.shape[axis] if axis < x.ndim else x.shape[-1]
    return _profile("softmax", shapes={"batch": batch, "seq": seq},
                    flops=5 * x.size, input_bytes=x.nbytes, output_bytes=x.nbytes,
                    workspace_bytes=batch * 2 * 2)  # max + sum


def profile_transpose(x: np.ndarray, axes: tuple = None) -> OpProfile:
    """Transpose Profile"""
    ndim = x.ndim
    axes = axes or tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
    pattern = "sequential" if axes == tuple(range(ndim)) else "strided"
    return _profile("transpose", shapes={"shape": x.shape, "axes": axes},
                    flops=0, input_bytes=x.nbytes, output_bytes=x.nbytes,
                    memory_pattern=pattern)


def profile_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                      mask: np.ndarray = None) -> OpProfile:
    """Attention Profile: softmax(Q @ K^T / sqrt(d)) @ V"""
    if q.ndim == 3:
        batch, seq_q, head_dim = q.shape
        heads = 1
    else:
        batch, heads, seq_q, head_dim = q.shape
    seq_kv, db = k.shape[-2], dtype_bytes("fp16")
    # FLOPs: QK + Softmax + SV
    flops = batch * heads * (4 * seq_q * head_dim * seq_kv + 5 * seq_q * seq_kv)
    return _profile("attention", compute_unit="cube",
                    shapes={"batch": batch, "heads": heads, "seq_q": seq_q,
                            "seq_kv": seq_kv, "head_dim": head_dim},
                    flops=flops, input_bytes=q.nbytes + k.nbytes + v.nbytes,
                    output_bytes=batch * heads * seq_q * head_dim * db,
                    workspace_bytes=batch * heads * seq_q * seq_kv * db)


def profile_gelu(x: np.ndarray) -> OpProfile:
    """GELU Profile"""
    return _profile("gelu", shapes={"size": x.size}, flops=10 * x.size,
                    input_bytes=x.nbytes, output_bytes=x.nbytes)


def profile_add(a: np.ndarray, b: np.ndarray) -> OpProfile:
    """Add Profile"""
    return _profile("add", shapes={"size": a.size}, flops=a.size,
                    input_bytes=a.nbytes + b.nbytes, output_bytes=a.nbytes)


