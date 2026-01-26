"""算子 Profile 数据结构"""

from dataclasses import dataclass, field
from typing import Any, Dict

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
