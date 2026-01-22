"""Pass 机制模块

提供时延分析的各类 Pass:
- RooflinePass: 基础 Roofline 模型
- MemoryEfficiencyPass: 访存效率修正
- ForwardPrefetchPass: 前向预取优化
- BackwardPrefetchPass: 后向预取优化
- CubeVectorParallelPass: Cube/Vector 并行
- OverheadPass: 开销计算
"""

from .base import (
    PassConfig,
    PassResult,
    PassPreset,
    BasePass,
)

from .roofline import RooflinePass
from .memory_efficiency import MemoryEfficiencyPass
from .prefetch import ForwardPrefetchPass, BackwardPrefetchPass
from .parallel import CubeVectorParallelPass
from .overhead import OverheadPass


# 所有 Pass 按执行顺序
ALL_PASSES = [
    RooflinePass,
    MemoryEfficiencyPass,
    ForwardPrefetchPass,
    BackwardPrefetchPass,
    CubeVectorParallelPass,
    OverheadPass,
]


__all__ = [
    # 配置
    "PassConfig",
    "PassResult",
    "PassPreset",
    "BasePass",
    # Passes
    "RooflinePass",
    "MemoryEfficiencyPass",
    "ForwardPrefetchPass",
    "BackwardPrefetchPass",
    "CubeVectorParallelPass",
    "OverheadPass",
    # 列表
    "ALL_PASSES",
]
