"""共享内存类型定义

此模块定义了跨模块共享的内存相关类型，避免 ops → optimizer 的反向依赖。

使用方式:
    from aidevtools.core.memory_types import MemoryLevel, MemoryRegion

层级关系:
    core.memory_types (基础类型)
    ├── ops.datagen (数据生成，只依赖 core)
    └── optimizer.memory_plan (内存规划，只依赖 core)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class MemoryLevel(Enum):
    """内存层级"""
    L1 = "L1"
    L2 = "L2"
    HBM = "HBM"


class DMADirection(Enum):
    """DMA 传输方向"""
    LOAD = "load"    # HBM/L2 -> L1
    STORE = "store"  # L1 -> L2/HBM


@dataclass
class MemoryRegion:
    """内存区域"""
    name: str
    level: MemoryLevel
    base_addr: int
    size: int
    dtype: str = "fp16"

    @property
    def end_addr(self) -> int:
        return self.base_addr + self.size

    def contains(self, addr: int) -> bool:
        return self.base_addr <= addr < self.end_addr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "base_addr": hex(self.base_addr),
            "size": self.size,
            "end_addr": hex(self.end_addr),
            "dtype": self.dtype,
        }


@dataclass
class TensorAllocation:
    """张量分配"""
    tensor_name: str
    op_name: str
    role: str  # "input", "output", "weight"
    shape: Tuple[int, ...]
    dtype: str
    regions: Dict[MemoryLevel, MemoryRegion] = field(default_factory=dict)

    def get_address(self, level: MemoryLevel) -> Optional[int]:
        """获取指定层级的地址"""
        if level in self.regions:
            return self.regions[level].base_addr
        return None

    def total_size(self) -> int:
        """总字节数"""
        dtype_sizes = {"fp16": 2, "fp32": 4, "bf16": 2, "int8": 1, "bfp16": 2, "bfp8": 1}
        elements = 1
        for dim in self.shape:
            elements *= dim
        return elements * dtype_sizes.get(self.dtype, 2)


@dataclass
class DMAOp:
    """DMA 操作"""
    op_id: int
    tensor_name: str
    direction: DMADirection
    src_addr: int
    dst_addr: int
    size: int
    src_level: MemoryLevel
    dst_level: MemoryLevel

    # 可选参数
    stride: int = 0
    repeat: int = 1
    wait_id: Optional[int] = None  # 依赖的 DMA 操作

    def to_code(self) -> str:
        """生成 DMA 代码"""
        if self.direction == DMADirection.LOAD:
            func = "dma_load"
        else:
            func = "dma_store"

        code = f"{func}({self.op_id}, "
        code += f"src={hex(self.src_addr)}, "
        code += f"dst={hex(self.dst_addr)}, "
        code += f"size={self.size}"

        if self.stride > 0:
            code += f", stride={self.stride}"
        if self.repeat > 1:
            code += f", repeat={self.repeat}"
        if self.wait_id is not None:
            code += f", wait={self.wait_id}"

        code += ")"
        return code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "tensor": self.tensor_name,
            "direction": self.direction.value,
            "src_addr": hex(self.src_addr),
            "dst_addr": hex(self.dst_addr),
            "size": self.size,
            "src_level": self.src_level.value,
            "dst_level": self.dst_level.value,
        }
