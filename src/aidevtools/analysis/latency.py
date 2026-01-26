"""时延计算结果数据结构"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .profile import OpProfile

if TYPE_CHECKING:
    from .analyzer import AnalysisSummary
    from .chip import ChipSpec
    from .passes.base import PassConfig, PassResult


@dataclass
class TimingMetrics:
    """时延指标"""

    compute_us: float = 0.0  # 计算时延
    memory_us: float = 0.0  # 访存时延
    roofline_us: float = 0.0  # Roofline 时延 = max(compute, memory)
    overhead_us: float = 0.0  # 开销（启动、tiling等）
    total_us: float = 0.0  # 最终时延


@dataclass
class OptimizationSavings:
    """优化节省"""

    prefetch_us: float = 0.0  # 前向预取节省
    backward_prefetch_us: float = 0.0  # 后向预取节省
    parallel_us: float = 0.0  # 并行节省


@dataclass
class BandwidthMetrics:
    """带宽指标"""

    min_gbps: float = 0.0  # 最低带宽需求
    headroom: float = 0.0  # 带宽余量
    effective_gbps: float = 0.0  # 有效带宽 (考虑并发竞争)


@dataclass
class TrafficMetrics:
    """流量指标"""

    original_bytes: int = 0  # 原始流量
    optimized_bytes: int = 0  # 优化后流量 (L2复用/Tiling后)


@dataclass
class LatencyBreakdown:
    """单算子时延分解 (使用组合模式)"""

    profile: OpProfile

    # === 组合子对象 ===
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    savings: OptimizationSavings = field(default_factory=OptimizationSavings)
    bandwidth: BandwidthMetrics = field(default_factory=BandwidthMetrics)
    traffic: TrafficMetrics = field(default_factory=TrafficMetrics)

    # === 瓶颈分析 ===
    bottleneck: str = "memory"  # "compute" | "memory"

    # === 额外信息 ===
    details: Dict[str, Any] = field(default_factory=dict)

    def update_roofline(self):
        """更新 Roofline 时延"""
        self.timing.roofline_us = max(self.timing.compute_us, self.timing.memory_us)
        self.bottleneck = "compute" if self.timing.compute_us >= self.timing.memory_us else "memory"

    @property
    def compute_utilization(self) -> float:
        """算力利用率"""
        if self.timing.total_us == 0:
            return 0
        return (self.timing.compute_us / self.timing.total_us) * 100

    @property
    def bandwidth_utilization(self) -> float:
        """带宽利用率"""
        if self.timing.total_us == 0:
            return 0
        return (self.timing.memory_us / self.timing.total_us) * 100


@dataclass
class ResultTotals:
    """结果汇总"""

    latency_us: float = 0.0
    flops: int = 0
    memory_bytes: int = 0


@dataclass
class PipelineEffects:
    """流水效果"""

    serial_latency_us: float = 0.0  # 串行时延（无优化）
    prefetch_saved_us: float = 0.0
    parallel_saved_us: float = 0.0


@dataclass
class UtilizationMetrics:
    """利用率指标"""

    compute: float = 0.0
    bandwidth: float = 0.0


@dataclass
class LatencyResult:
    """完整时延分析结果 (使用组合模式)"""

    # 芯片信息
    chip_name: str = ""
    chip_spec: Optional["ChipSpec"] = None
    pass_config: Optional["PassConfig"] = None

    # 算子时延 (主字段)
    breakdowns: List[LatencyBreakdown] = field(default_factory=list)

    # === 组合子对象 ===
    totals: ResultTotals = field(default_factory=ResultTotals)
    pipeline: PipelineEffects = field(default_factory=PipelineEffects)
    utilization: UtilizationMetrics = field(default_factory=UtilizationMetrics)

    # === 其他 ===
    summary: Optional["AnalysisSummary"] = None
    pass_results: List["PassResult"] = field(default_factory=list)
    gantt_data: Optional["GanttData"] = None

    def compute_summary(self):
        """计算汇总数据"""
        ops = self.breakdowns
        self.totals.flops = sum(op.profile.flops for op in ops)
        self.totals.memory_bytes = sum(op.profile.total_bytes for op in ops)
        self.totals.latency_us = sum(op.timing.total_us for op in ops)
        self.pipeline.serial_latency_us = sum(
            op.timing.roofline_us + op.timing.overhead_us for op in ops
        )
        self.pipeline.prefetch_saved_us = sum(
            op.savings.prefetch_us + op.savings.backward_prefetch_us for op in ops
        )
        self.pipeline.parallel_saved_us = sum(op.savings.parallel_us for op in ops)


@dataclass
class GanttItem:
    """甘特图项

    Attributes:
        op_name: 算子名称
        unit: 执行单元 ("cube" | "vector" | "dma")
        start_us: 开始时间 (微秒)
        end_us: 结束时间 (微秒)
        category: 类别 ("execution" | "prefetch" | "parallel")
        label: 显示标签
        color: 颜色代码 (十六进制)
    """

    op_name: str
    unit: str = ""  # "cube" | "vector" | "dma"
    start_us: float = 0.0
    end_us: float = 0.0
    category: str = ""  # "execution" | "prefetch" | "parallel"
    label: str = ""
    color: str = ""

    @property
    def duration_us(self) -> float:
        """持续时间 (微秒)"""
        return self.end_us - self.start_us


@dataclass
class GanttData:
    """甘特图数据

    Attributes:
        items: 甘特图项列表
        total_duration_us: 总持续时间 (微秒)
        chip_name: 芯片名称
    """

    items: List[GanttItem] = field(default_factory=list)
    total_duration_us: float = 0.0
    chip_name: str = ""

    def add_item(self, item: GanttItem):
        """添加甘特图项"""
        self.items.append(item)
        self.total_duration_us = max(self.total_duration_us, item.end_us)
