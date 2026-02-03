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

from .bandwidth import (
    BandwidthConstraintPass,
    MinTrafficPass,
    TrafficConstraintPass,
)
from .base import (
    BandwidthConfig,
    BasePass,
    MinTrafficConfig,
    OverheadConfig,
    PassConfig,
    PassContext,
    PassPreset,
    PassResult,
    # 子配置类
    PrefetchConfig,
    TrafficConfig,
)
from .memory_efficiency import MemoryEfficiencyPass
from .overhead import OverheadPass
from .parallel import CubeVectorParallelPass
from .prefetch import BackwardPrefetchPass, ForwardPrefetchPass
from .roofline import RooflinePass

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
    # 子配置类
    "PrefetchConfig",
    "OverheadConfig",
    "BandwidthConfig",
    "TrafficConfig",
    "MinTrafficConfig",
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
