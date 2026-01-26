"""Pass 机制基础设施

每个 Pass 负责一类时延优化分析，可独立开关和配置。

Pass 执行顺序 (order):
100. RooflinePass - 基础 Roofline 时延计算
150. MinTrafficPass - 最低流量优化 (L2复用/Tiling)
200. MemoryEfficiencyPass - 访存效率修正
250. BandwidthConstraintPass - 全局带宽约束
300. ForwardPrefetchPass - 前向预取优化
400. BackwardPrefetchPass - 后向预取优化
500. CubeVectorParallelPass - Cube/Vector 并行优化
600. OverheadPass - 开销计算
700. TrafficConstraintPass - 流量约束检查
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..chip import ChipSpec
    from ..latency import LatencyBreakdown
    from ..profile import OpProfile


class PassPreset(Enum):
    """Pass 预设配置"""
    MINIMAL = "minimal"      # 仅 Roofline
    STANDARD = "standard"    # 标准优化
    AGGRESSIVE = "aggressive"  # 激进优化
    CUSTOM = "custom"        # 自定义


@dataclass
class PrefetchConfig:
    """预取配置"""
    forward_enabled: bool = True
    backward_enabled: bool = True
    efficiency: float = 0.8          # 预取效率
    backward_depth: int = 2          # 后向预取深度


@dataclass
class OverheadConfig:
    """开销配置"""
    enabled: bool = True
    kernel_launch_us: float = 5.0    # kernel 启动开销
    sync_us: float = 2.0             # 同步开销
    context_switch_us: float = 1.0   # 算子切换时延
    tiling_us: float = 0.5           # Tiling 调度开销 (per tile)
    tiling_count: int = 1            # 默认 tile 数量


@dataclass
class BandwidthConfig:
    """带宽约束配置"""
    enabled: bool = True
    concurrent_streams: int = 1      # 并发流数量
    contention_model: str = "linear" # "linear" | "sqrt" | "none"


@dataclass
class TrafficConfig:
    """流量约束配置"""
    enabled: bool = False
    max_bytes: int = 0               # 最大允许流量 (0=无限制)
    budget_mode: str = "none"        # "none" | "strict" | "soft"


@dataclass
class MinTrafficConfig:
    """最低流量优化配置"""
    enabled: bool = False
    cache_line_bytes: int = 64       # Cache line 大小
    l2_reuse_factor: float = 1.0     # L2 缓存复用因子 (1.0=无复用)
    tiling_efficiency: float = 1.0   # Tiling 效率 (1.0=无 tiling)


@dataclass
class PassConfig:
    """Pass 配置 (使用组合模式减少属性数量)"""

    # === 基础 ===
    enabled: bool = True
    preset: PassPreset = PassPreset.STANDARD

    # === Pass 开关 ===
    roofline_enabled: bool = True
    memory_efficiency_enabled: bool = True
    use_effective_bandwidth: bool = True
    cube_vector_parallel_enabled: bool = True

    # === 子配置 (组合模式) ===
    prefetch: PrefetchConfig = field(default_factory=PrefetchConfig)
    overhead: OverheadConfig = field(default_factory=OverheadConfig)
    bandwidth: BandwidthConfig = field(default_factory=BandwidthConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    min_traffic: MinTrafficConfig = field(default_factory=MinTrafficConfig)

    # === 扩展参数 ===
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_preset(cls, preset: PassPreset) -> 'PassConfig':
        """从预设创建配置"""
        if preset == PassPreset.MINIMAL:
            return cls(
                preset=preset,
                roofline_enabled=True,
                memory_efficiency_enabled=False,
                cube_vector_parallel_enabled=False,
                prefetch=PrefetchConfig(forward_enabled=False, backward_enabled=False),
                overhead=OverheadConfig(enabled=False),
            )
        if preset == PassPreset.STANDARD:
            return cls(
                preset=preset,
                roofline_enabled=True,
                memory_efficiency_enabled=True,
                cube_vector_parallel_enabled=True,
                prefetch=PrefetchConfig(forward_enabled=True, backward_enabled=False),
                overhead=OverheadConfig(enabled=True),
            )
        if preset == PassPreset.AGGRESSIVE:
            return cls(
                preset=preset,
                roofline_enabled=True,
                memory_efficiency_enabled=True,
                cube_vector_parallel_enabled=True,
                prefetch=PrefetchConfig(
                    forward_enabled=True,
                    backward_enabled=True,
                    efficiency=0.9,
                    backward_depth=3,
                ),
                overhead=OverheadConfig(enabled=True),
                min_traffic=MinTrafficConfig(
                    enabled=True,
                    l2_reuse_factor=0.8,
                    tiling_efficiency=0.9,
                ),
            )
        return cls(preset=preset)


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


@dataclass
class PassContext:
    """Pass 运行时上下文，提供邻近算子信息"""
    next_profile: Optional['OpProfile'] = None
    prev_profile: Optional['OpProfile'] = None
    future_profiles: List['OpProfile'] = field(default_factory=list)


class BasePass(ABC):
    """Pass 基类

    子类需要实现:
    - _do_run(): 实际执行逻辑

    可选覆盖:
    - is_enabled(): 检查 Pass 是否启用 (默认使用 config_key)
    """

    name: str = "base"
    description: str = "Base pass"
    order: int = 0
    config_key: Optional[str] = None  # 如 "roofline" -> config.roofline_enabled

    def __init__(self, config: PassConfig = None):
        self.config = config or PassConfig()

    def is_enabled(self) -> bool:
        """检查 Pass 是否启用 (基于 config_key)"""
        if not self.config.enabled:
            return False
        if self.config_key is None:
            return True
        return getattr(self.config, f"{self.config_key}_enabled", True)

    def run(self, latency_breakdown: 'LatencyBreakdown',
            chip_spec: 'ChipSpec',
            context: PassContext = None) -> PassResult:
        """
        执行 Pass (模板方法)

        Args:
            latency_breakdown: 当前时延分解
            chip_spec: 芯片规格
            context: Pass 上下文 (可选)

        Returns:
            PassResult
        """
        result = PassResult(pass_name=self.name, enabled=self.is_enabled())

        if not self.is_enabled():
            return result

        return self._do_run(latency_breakdown, chip_spec, result, context)

    @abstractmethod
    def _do_run(self, latency_breakdown: 'LatencyBreakdown',
                chip_spec: 'ChipSpec',
                result: PassResult,
                context: PassContext = None) -> PassResult:
        """
        实际执行逻辑 (子类实现)

        Args:
            latency_breakdown: 当前时延分解
            chip_spec: 芯片规格
            result: 预创建的结果对象
            context: Pass 上下文

        Returns:
            PassResult
        """

    def _skip(self, result: PassResult, latency_us: float, reason: str) -> PassResult:
        """跳过 Pass 时填充结果"""
        result.latency_before_us = latency_us
        result.latency_after_us = latency_us
        result.details = {"reason": reason}
        return result
