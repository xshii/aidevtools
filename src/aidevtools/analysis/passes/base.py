"""Pass 机制基础设施

每个 Pass 负责一类时延优化分析，可独立开关和配置。

Pass 执行顺序:
1. RooflinePass - 基础 Roofline 时延计算
1.5. MinTrafficPass - 最低流量优化 (L2复用/Tiling)
2. MemoryEfficiencyPass - 访存效率修正
2.5. BandwidthConstraintPass - 全局带宽约束
3. ForwardPrefetchPass - 前向预取优化
4. BackwardPrefetchPass - 后向预取优化
5. CubeVectorParallelPass - Cube/Vector 并行优化
6. OverheadPass - 开销计算
7. TrafficConstraintPass - 流量约束检查
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..latency import LatencyBreakdown
    from ..chip import ChipSpec


class PassPreset(Enum):
    """Pass 预设配置"""
    MINIMAL = "minimal"      # 仅 Roofline
    STANDARD = "standard"    # 标准优化
    AGGRESSIVE = "aggressive"  # 激进优化
    CUSTOM = "custom"        # 自定义


@dataclass
class PassConfig:
    """Pass 配置"""

    # === 基础 ===
    enabled: bool = True

    # === 预设 ===
    preset: PassPreset = PassPreset.STANDARD

    # === Roofline ===
    roofline_enabled: bool = True

    # === 访存效率 ===
    memory_efficiency_enabled: bool = True
    use_effective_bandwidth: bool = True  # 使用有效带宽而非峰值

    # === 前向预取 ===
    forward_prefetch_enabled: bool = True
    prefetch_efficiency: float = 0.8  # 预取效率

    # === 后向预取 ===
    backward_prefetch_enabled: bool = True
    backward_prefetch_depth: int = 2  # 后向预取深度

    # === Cube/Vector 并行 ===
    cube_vector_parallel_enabled: bool = True

    # === 开销 ===
    overhead_enabled: bool = True
    kernel_launch_us: float = 5.0  # kernel 启动开销
    sync_overhead_us: float = 2.0  # 同步开销

    # === 全局带宽约束 ===
    bandwidth_constraint_enabled: bool = True
    concurrent_streams: int = 1  # 并发流数量
    bandwidth_contention_model: str = "linear"  # "linear" | "sqrt" | "none"

    # === 流量约束模式 ===
    traffic_constraint_enabled: bool = False
    max_traffic_bytes: int = 0  # 最大允许流量 (0=无限制)
    traffic_budget_mode: str = "none"  # "none" | "strict" | "soft"

    # === 最低流量模式 ===
    min_traffic_mode_enabled: bool = False
    cache_line_bytes: int = 64  # Cache line 大小
    l2_reuse_factor: float = 1.0  # L2 缓存复用因子 (1.0=无复用)
    tiling_efficiency: float = 1.0  # Tiling 效率 (1.0=无 tiling)

    # === 扩展参数 ===
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_preset(cls, preset: PassPreset) -> 'PassConfig':
        """从预设创建配置"""
        config = cls(preset=preset)

        if preset == PassPreset.MINIMAL:
            config.roofline_enabled = True
            config.memory_efficiency_enabled = False
            config.forward_prefetch_enabled = False
            config.backward_prefetch_enabled = False
            config.cube_vector_parallel_enabled = False
            config.overhead_enabled = False

        elif preset == PassPreset.STANDARD:
            config.roofline_enabled = True
            config.memory_efficiency_enabled = True
            config.forward_prefetch_enabled = True
            config.backward_prefetch_enabled = False
            config.cube_vector_parallel_enabled = True
            config.overhead_enabled = True

        elif preset == PassPreset.AGGRESSIVE:
            config.roofline_enabled = True
            config.memory_efficiency_enabled = True
            config.forward_prefetch_enabled = True
            config.backward_prefetch_enabled = True
            config.cube_vector_parallel_enabled = True
            config.overhead_enabled = True
            config.prefetch_efficiency = 0.9
            config.backward_prefetch_depth = 3
            # 激进模式启用最低流量优化
            config.min_traffic_mode_enabled = True
            config.l2_reuse_factor = 0.8
            config.tiling_efficiency = 0.9

        return config


@dataclass
class PassResult:
    """单个 Pass 的执行结果"""
    pass_name: str
    enabled: bool = True

    # 时延变化
    latency_before_us: float = 0.0
    latency_after_us: float = 0.0
    latency_saved_us: float = 0.0

    # 详细信息
    details: Dict[str, Any] = field(default_factory=dict)

    # 警告/建议
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def improvement_ratio(self) -> float:
        """改进比例"""
        if self.latency_before_us == 0:
            return 0
        return self.latency_saved_us / self.latency_before_us


class BasePass(ABC):
    """Pass 基类"""

    name: str = "base"
    description: str = "Base pass"
    order: int = 0  # 执行顺序

    def __init__(self, config: PassConfig = None):
        self.config = config or PassConfig()

    @abstractmethod
    def is_enabled(self) -> bool:
        """检查 Pass 是否启用"""
        pass

    @abstractmethod
    def run(self, latency_breakdown: 'LatencyBreakdown',
            chip_spec: 'ChipSpec') -> PassResult:
        """
        执行 Pass

        Args:
            latency_breakdown: 当前时延分解
            chip_spec: 芯片规格

        Returns:
            PassResult
        """
        pass

    def validate(self, latency_breakdown: 'LatencyBreakdown') -> List[str]:
        """验证输入数据，返回警告列表"""
        return []
