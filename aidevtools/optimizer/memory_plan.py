"""
内存规划模块

功能:
- 内存地址分配
- DMA 代码生成
- 内存范围管理

设计模式:
- Builder 模式: 构建 DMA 操作序列
- Strategy 模式: 不同的地址分配策略
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


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
        dtype_sizes = {"fp16": 2, "fp32": 4, "bf16": 2, "int8": 1}
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


class AddressAllocator(ABC):
    """地址分配策略基类"""

    @abstractmethod
    def allocate(self, size: int, alignment: int = 256) -> int:
        """分配地址"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置分配器"""
        pass


class LinearAllocator(AddressAllocator):
    """线性地址分配"""

    def __init__(self, base_addr: int = 0, max_size: int = 256 * 1024):
        self.base_addr = base_addr
        self.max_size = max_size
        self.current_offset = 0

    def allocate(self, size: int, alignment: int = 256) -> int:
        # 对齐
        aligned_offset = (self.current_offset + alignment - 1) // alignment * alignment

        if aligned_offset + size > self.max_size:
            raise MemoryError(f"Out of memory: need {size}, have {self.max_size - aligned_offset}")

        addr = self.base_addr + aligned_offset
        self.current_offset = aligned_offset + size
        return addr

    def reset(self) -> None:
        self.current_offset = 0

    @property
    def used_size(self) -> int:
        return self.current_offset


class DoubleBufferAllocator(AddressAllocator):
    """双缓冲地址分配"""

    def __init__(self, base_addr: int = 0, max_size: int = 256 * 1024):
        self.base_addr = base_addr
        self.max_size = max_size
        self.half_size = max_size // 2
        self.current_buffer = 0  # 0 或 1
        self.offsets = [0, 0]

    def allocate(self, size: int, alignment: int = 256) -> int:
        offset = self.offsets[self.current_buffer]
        aligned_offset = (offset + alignment - 1) // alignment * alignment

        if aligned_offset + size > self.half_size:
            raise MemoryError(f"Buffer {self.current_buffer} out of memory")

        addr = self.base_addr + self.current_buffer * self.half_size + aligned_offset
        self.offsets[self.current_buffer] = aligned_offset + size
        return addr

    def swap_buffer(self) -> None:
        """切换缓冲区"""
        self.current_buffer = 1 - self.current_buffer

    def reset(self) -> None:
        self.offsets = [0, 0]
        self.current_buffer = 0


class MemoryPlan:
    """
    内存规划

    管理张量分配和 DMA 操作生成
    """

    def __init__(self,
                 l1_size: int = 256 * 1024,
                 l2_size: int = 2 * 1024 * 1024,
                 l1_base: int = 0x0,
                 l2_base: int = 0x100000,
                 hbm_base: int = 0x80000000):
        """
        Args:
            l1_size: L1 缓存大小
            l2_size: L2 缓存大小
            l1_base: L1 基地址
            l2_base: L2 基地址
            hbm_base: HBM 基地址
        """
        self.allocators = {
            MemoryLevel.L1: DoubleBufferAllocator(l1_base, l1_size),
            MemoryLevel.L2: LinearAllocator(l2_base, l2_size),
            MemoryLevel.HBM: LinearAllocator(hbm_base, 1024 * 1024 * 1024),  # 1GB
        }

        self.allocations: Dict[str, TensorAllocation] = {}
        self.dma_ops: List[DMAOp] = []
        self.dma_counter = 0

    def allocate_tensor(self, name: str, op_name: str, role: str,
                       shape: Tuple[int, ...], dtype: str = "fp16",
                       levels: Optional[List[MemoryLevel]] = None) -> TensorAllocation:
        """
        分配张量

        Args:
            name: 张量名称
            op_name: 所属算子
            role: 角色 (input/output/weight)
            shape: 形状
            dtype: 数据类型
            levels: 需要分配的内存层级

        Returns:
            TensorAllocation: 分配结果
        """
        if levels is None:
            # 默认：input/weight 在所有层级，output 在 L1/L2
            if role == "output":
                levels = [MemoryLevel.L1, MemoryLevel.L2]
            else:
                levels = [MemoryLevel.L1, MemoryLevel.L2, MemoryLevel.HBM]

        allocation = TensorAllocation(
            tensor_name=name,
            op_name=op_name,
            role=role,
            shape=shape,
            dtype=dtype
        )

        size = allocation.total_size()

        for level in levels:
            allocator = self.allocators[level]
            addr = allocator.allocate(size)

            region = MemoryRegion(
                name=f"{name}_{level.value}",
                level=level,
                base_addr=addr,
                size=size,
                dtype=dtype
            )
            allocation.regions[level] = region

        self.allocations[f"{op_name}.{name}"] = allocation
        return allocation

    def generate_dma_load(self, tensor_name: str,
                         src_level: MemoryLevel = MemoryLevel.L2,
                         dst_level: MemoryLevel = MemoryLevel.L1,
                         wait_for: Optional[int] = None) -> DMAOp:
        """生成 DMA 加载操作"""
        allocation = self.allocations.get(tensor_name)
        if not allocation:
            raise ValueError(f"Tensor {tensor_name} not found")

        src_region = allocation.regions.get(src_level)
        dst_region = allocation.regions.get(dst_level)

        if not src_region or not dst_region:
            raise ValueError(f"Missing region for {tensor_name}")

        dma_op = DMAOp(
            op_id=self.dma_counter,
            tensor_name=tensor_name,
            direction=DMADirection.LOAD,
            src_addr=src_region.base_addr,
            dst_addr=dst_region.base_addr,
            size=src_region.size,
            src_level=src_level,
            dst_level=dst_level,
            wait_id=wait_for
        )

        self.dma_ops.append(dma_op)
        self.dma_counter += 1

        return dma_op

    def generate_dma_store(self, tensor_name: str,
                          src_level: MemoryLevel = MemoryLevel.L1,
                          dst_level: MemoryLevel = MemoryLevel.L2,
                          wait_for: Optional[int] = None) -> DMAOp:
        """生成 DMA 存储操作"""
        allocation = self.allocations.get(tensor_name)
        if not allocation:
            raise ValueError(f"Tensor {tensor_name} not found")

        src_region = allocation.regions.get(src_level)
        dst_region = allocation.regions.get(dst_level)

        if not src_region or not dst_region:
            raise ValueError(f"Missing region for {tensor_name}")

        dma_op = DMAOp(
            op_id=self.dma_counter,
            tensor_name=tensor_name,
            direction=DMADirection.STORE,
            src_addr=src_region.base_addr,
            dst_addr=dst_region.base_addr,
            size=src_region.size,
            src_level=src_level,
            dst_level=dst_level,
            wait_id=wait_for
        )

        self.dma_ops.append(dma_op)
        self.dma_counter += 1

        return dma_op

    def generate_code(self, language: str = "pseudo") -> str:
        """
        生成 DMA 代码

        Args:
            language: 目标语言 (pseudo/c/python)

        Returns:
            生成的代码
        """
        if language == "pseudo":
            return self._generate_pseudo_code()
        elif language == "c":
            return self._generate_c_code()
        elif language == "python":
            return self._generate_python_code()
        else:
            return self._generate_pseudo_code()

    def _generate_pseudo_code(self) -> str:
        """生成伪代码"""
        lines = []
        lines.append("// Memory Allocation")
        lines.append("")

        for name, alloc in self.allocations.items():
            lines.append(f"// {name}: {alloc.role}, shape={alloc.shape}")
            for level, region in alloc.regions.items():
                lines.append(f"//   {level.value}: {hex(region.base_addr)} - {hex(region.end_addr)}")
        lines.append("")

        lines.append("// DMA Operations")
        lines.append("")

        for dma in self.dma_ops:
            lines.append(dma.to_code())

        return "\n".join(lines)

    def _generate_c_code(self) -> str:
        """生成 C 代码"""
        lines = []
        lines.append("#include <dma.h>")
        lines.append("")

        # 地址定义
        lines.append("// Address Definitions")
        for name, alloc in self.allocations.items():
            safe_name = name.replace(".", "_")
            for level, region in alloc.regions.items():
                lines.append(f"#define ADDR_{safe_name}_{level.value} {hex(region.base_addr)}")
        lines.append("")

        # DMA 函数
        lines.append("void execute_dma_plan() {")

        for dma in self.dma_ops:
            direction = "DMA_LOAD" if dma.direction == DMADirection.LOAD else "DMA_STORE"

            lines.append(f"    // {dma.tensor_name}")
            lines.append(f"    dma_transfer({dma.op_id}, {direction},")
            lines.append(f"                 {hex(dma.src_addr)}, {hex(dma.dst_addr)},")
            lines.append(f"                 {dma.size});")

            if dma.wait_id is not None:
                lines.append(f"    dma_wait({dma.wait_id});")

            lines.append("")

        lines.append("}")

        return "\n".join(lines)

    def _generate_python_code(self) -> str:
        """生成 Python 代码"""
        lines = []
        lines.append("from typing import Dict, List")
        lines.append("from dataclasses import dataclass")
        lines.append("")

        lines.append("# Address Definitions")
        lines.append("ADDRESSES = {")
        for name, alloc in self.allocations.items():
            lines.append(f"    '{name}': {{")
            for level, region in alloc.regions.items():
                lines.append(f"        '{level.value}': {hex(region.base_addr)},")
            lines.append("    },")
        lines.append("}")
        lines.append("")

        lines.append("# DMA Operations")
        lines.append("DMA_OPS = [")
        for dma in self.dma_ops:
            lines.append(f"    {{")
            lines.append(f"        'id': {dma.op_id},")
            lines.append(f"        'tensor': '{dma.tensor_name}',")
            lines.append(f"        'direction': '{dma.direction.value}',")
            lines.append(f"        'src': {hex(dma.src_addr)},")
            lines.append(f"        'dst': {hex(dma.dst_addr)},")
            lines.append(f"        'size': {dma.size},")
            lines.append(f"    }},")
        lines.append("]")

        return "\n".join(lines)

    def get_allocation_report(self) -> str:
        """生成分配报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("Memory Allocation Report")
        lines.append("=" * 60)
        lines.append("")

        # 按层级统计
        level_usage = {level: 0 for level in MemoryLevel}

        for alloc in self.allocations.values():
            for level, region in alloc.regions.items():
                level_usage[level] += region.size

        lines.append("Memory Usage by Level:")
        lines.append("-" * 40)
        for level, usage in level_usage.items():
            allocator = self.allocators[level]
            if isinstance(allocator, LinearAllocator):
                max_size = allocator.max_size
            else:
                max_size = allocator.max_size
            pct = usage / max_size * 100 if max_size > 0 else 0
            lines.append(f"  {level.value:4}: {usage:>10,} bytes ({pct:.1f}%)")

        lines.append("")

        # 详细分配
        lines.append("Tensor Allocations:")
        lines.append("-" * 40)

        for name, alloc in self.allocations.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Role: {alloc.role}")
            lines.append(f"  Shape: {alloc.shape}")
            lines.append(f"  Size: {alloc.total_size():,} bytes")
            lines.append(f"  Addresses:")
            for level, region in alloc.regions.items():
                lines.append(f"    {level.value}: {hex(region.base_addr)} - {hex(region.end_addr)}")

        return "\n".join(lines)

    def reset(self) -> None:
        """重置内存规划"""
        for allocator in self.allocators.values():
            allocator.reset()
        self.allocations.clear()
        self.dma_ops.clear()
        self.dma_counter = 0


class MemoryPlanBuilder:
    """
    内存规划构建器 (Builder 模式)
    """

    def __init__(self):
        self._plan: Optional[MemoryPlan] = None
        self._current_op: Optional[str] = None

    def create_plan(self, l1_size: int = 256 * 1024,
                   l2_size: int = 2 * 1024 * 1024) -> "MemoryPlanBuilder":
        """创建内存规划"""
        self._plan = MemoryPlan(l1_size=l1_size, l2_size=l2_size)
        return self

    def for_op(self, op_name: str) -> "MemoryPlanBuilder":
        """设置当前算子"""
        self._current_op = op_name
        return self

    def add_input(self, name: str, shape: Tuple[int, ...],
                 dtype: str = "fp16") -> "MemoryPlanBuilder":
        """添加输入张量"""
        if not self._plan or not self._current_op:
            raise RuntimeError("Must call create_plan() and for_op() first")

        self._plan.allocate_tensor(
            name=name,
            op_name=self._current_op,
            role="input",
            shape=shape,
            dtype=dtype
        )
        return self

    def add_output(self, name: str, shape: Tuple[int, ...],
                  dtype: str = "fp16") -> "MemoryPlanBuilder":
        """添加输出张量"""
        if not self._plan or not self._current_op:
            raise RuntimeError("Must call create_plan() and for_op() first")

        self._plan.allocate_tensor(
            name=name,
            op_name=self._current_op,
            role="output",
            shape=shape,
            dtype=dtype
        )
        return self

    def add_weight(self, name: str, shape: Tuple[int, ...],
                  dtype: str = "fp16") -> "MemoryPlanBuilder":
        """添加权重张量"""
        if not self._plan or not self._current_op:
            raise RuntimeError("Must call create_plan() and for_op() first")

        self._plan.allocate_tensor(
            name=name,
            op_name=self._current_op,
            role="weight",
            shape=shape,
            dtype=dtype
        )
        return self

    def generate_dma(self) -> "MemoryPlanBuilder":
        """为所有张量生成 DMA 操作"""
        if not self._plan:
            raise RuntimeError("Must call create_plan() first")

        for name, alloc in self._plan.allocations.items():
            if alloc.role in ["input", "weight"]:
                # HBM -> L2 -> L1
                if MemoryLevel.HBM in alloc.regions:
                    hbm_op = self._plan.generate_dma_load(
                        name,
                        src_level=MemoryLevel.HBM,
                        dst_level=MemoryLevel.L2
                    )
                    self._plan.generate_dma_load(
                        name,
                        src_level=MemoryLevel.L2,
                        dst_level=MemoryLevel.L1,
                        wait_for=hbm_op.op_id
                    )

            elif alloc.role == "output":
                # L1 -> L2
                self._plan.generate_dma_store(
                    name,
                    src_level=MemoryLevel.L1,
                    dst_level=MemoryLevel.L2
                )

        return self

    def build(self) -> MemoryPlan:
        """构建内存规划"""
        if not self._plan:
            raise RuntimeError("Must call create_plan() first")
        return self._plan
