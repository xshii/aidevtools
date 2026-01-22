"""Pass 机制模块

提供时延分析的各类 Pass:
- RooflinePass: 基础 Roofline 模型
- MinTrafficPass: 最低流量优化
- MemoryEfficiencyPass: 访存效率修正
- BandwidthConstraintPass: 全局带宽约束
- ForwardPrefetchPass: 前向预取优化
- BackwardPrefetchPass: 后向预取优化
- CubeVectorParallelPass: Cube/Vector 并行
- OverheadPass: 开销计算
- TrafficConstraintPass: 流量约束检查
"""

from .base import (
    PassConfig,
    PassResult,
    PassPreset,
    PassContext,
    BasePass,
)

from .roofline import RooflinePass
from .memory_efficiency import MemoryEfficiencyPass
from .prefetch import ForwardPrefetchPass, BackwardPrefetchPass
from .parallel import CubeVectorParallelPass
from .overhead import OverheadPass
from .bandwidth import (
    BandwidthConstraintPass,
    TrafficConstraintPass,
    MinTrafficPass,
)


# 所有 Pass 按执行顺序 (order 属性决定)
ALL_PASSES = [
    RooflinePass,           # order=1
    MinTrafficPass,         # order=1.5 (最低流量优化)
    MemoryEfficiencyPass,   # order=2
    BandwidthConstraintPass, # order=2.5 (全局带宽约束)
    ForwardPrefetchPass,    # order=3
    BackwardPrefetchPass,   # order=4
    CubeVectorParallelPass, # order=5
    OverheadPass,           # order=6
    TrafficConstraintPass,  # order=7 (流量约束检查)
]


__all__ = [
    # 配置
    "PassConfig",
    "PassResult",
    "PassPreset",
    "PassContext",
    "BasePass",
    # Passes
    "RooflinePass",
    "MinTrafficPass",
    "MemoryEfficiencyPass",
    "BandwidthConstraintPass",
    "ForwardPrefetchPass",
    "BackwardPrefetchPass",
    "CubeVectorParallelPass",
    "OverheadPass",
    "TrafficConstraintPass",
    # 列表
    "ALL_PASSES",
]
