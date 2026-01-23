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


# 所有 Pass 按执行顺序排列
ALL_PASSES = [
    RooflinePass,           # 100
    MinTrafficPass,         # 150
    MemoryEfficiencyPass,   # 200
    BandwidthConstraintPass, # 250
    ForwardPrefetchPass,    # 300
    BackwardPrefetchPass,   # 400
    CubeVectorParallelPass, # 500
    OverheadPass,           # 600
    TrafficConstraintPass,  # 700
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
