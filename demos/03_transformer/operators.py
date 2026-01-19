"""Transformer 算子导入

直接使用 aidevtools.ops.nn 中的算子实现，无需重复定义。

所有算子已在核心库中实现：
- linear, matmul, relu, gelu, softmax
- layernorm, attention, embedding, add, mul
"""
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aidevtools.ops.nn import (
    linear,
    relu,
    gelu,
    softmax as softmax_safe,  # 别名，兼容旧代码
    layernorm,
    attention,
    embedding,
    matmul,
    add,
    mul,
)

__all__ = [
    "linear",
    "relu",
    "gelu",
    "softmax_safe",
    "layernorm",
    "attention",
    "embedding",
    "matmul",
    "add",
    "mul",
]
