"""时延计算结果数据结构"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .profile import OpProfile

if TYPE_CHECKING:
    from .chip import ChipSpec
    from .passes.base import PassConfig, PassResult
    from .analyzer import AnalysisSummary


@dataclass
class LatencyBreakdown:
    """单算子时延分解"""

    profile: OpProfile

    # === 基础时延 ===
    compute_time_us: float = 0.0       # 计算时延
    memory_time_us: float = 0.0        # 访存时延
    roofline_time_us: float = 0.0      # Roofline 时延 = max(compute, memory)

    # === 访存细分 ===
    input_load_us: float = 0.0         # 输入加载
    weight_load_us: float = 0.0        # 权重加载
    output_store_us: float = 0.0       # 输出存储

    # === 流水优化 ===
    prefetch_saved_us: float = 0.0     # 前向预取节省
    backward_prefetch_saved_us: float = 0.0  # 后向预取节省
    parallel_saved_us: float = 0.0     # 并行节省

    # === 开销 ===
    overhead_us: float = 0.0           # 开销（启动、tiling等）

    # === 最终时延 ===
    total_time_us: float = 0.0

    # === 瓶颈分析 ===
    bottleneck: str = "memory"         # "compute" | "memory"

    # === 带宽分析 ===
    min_bandwidth_gbps: float = 0.0    # 最低带宽需求
    bandwidth_headroom: float = 0.0    # 带宽余量
    effective_bandwidth_gbps: float = 0.0  # 有效带宽 (考虑并发竞争)

    # === 流量分析 ===
    original_traffic_bytes: int = 0    # 原始流量
    optimized_traffic_bytes: int = 0   # 优化后流量 (L2复用/Tiling后)

    # === 额外信息 ===
    details: Dict[str, Any] = field(default_factory=dict)

    def update_roofline(self):
        """更新 Roofline 时延"""
        self.roofline_time_us = max(self.compute_time_us, self.memory_time_us)
        self.bottleneck = "compute" if self.compute_time_us >= self.memory_time_us else "memory"

    @property
    def compute_utilization(self) -> float:
        """算力利用率"""
        if self.total_time_us == 0:
            return 0
        return (self.compute_time_us / self.total_time_us) * 100

    @property
    def bandwidth_utilization(self) -> float:
        """带宽利用率"""
        if self.total_time_us == 0:
            return 0
        return (self.memory_time_us / self.total_time_us) * 100


@dataclass
class LatencyResult:
    """完整时延分析结果"""

    # 芯片信息
    chip_name: str = ""
    chip_spec: Optional['ChipSpec'] = None
    pass_config: Optional['PassConfig'] = None

    # 算子时延 (支持两种命名)
    op_latencies: List[LatencyBreakdown] = field(default_factory=list)
    breakdowns: List[LatencyBreakdown] = field(default_factory=list)  # alias

    # === 汇总 ===
    total_latency_us: float = 0.0
    total_flops: int = 0
    total_memory_bytes: int = 0
    summary: Optional['AnalysisSummary'] = None

    # === 流水效果 ===
    serial_latency_us: float = 0.0     # 串行时延（无优化）
    total_prefetch_saved_us: float = 0.0
    total_parallel_saved_us: float = 0.0

    # === 整体利用率 ===
    overall_compute_util: float = 0.0
    overall_bandwidth_util: float = 0.0

    # === Pass 结果 ===
    pass_results: List['PassResult'] = field(default_factory=list)

    # === Gantt 数据 ===
    gantt_data: Optional['GanttData'] = None

    def __post_init__(self):
        """初始化后同步数据"""
        # 同步 breakdowns 和 op_latencies
        if self.breakdowns and not self.op_latencies:
            self.op_latencies = self.breakdowns
        elif self.op_latencies and not self.breakdowns:
            self.breakdowns = self.op_latencies

    def compute_summary(self):
        """计算汇总数据"""
        ops = self.op_latencies or self.breakdowns
        self.total_flops = sum(op.profile.flops for op in ops)
        self.total_memory_bytes = sum(op.profile.total_bytes for op in ops)
        self.total_latency_us = sum(op.total_time_us for op in ops)
        self.serial_latency_us = sum(op.roofline_time_us + op.overhead_us for op in ops)
        self.total_prefetch_saved_us = sum(
            op.prefetch_saved_us + op.backward_prefetch_saved_us for op in ops
        )
        self.total_parallel_saved_us = sum(op.parallel_saved_us for op in ops)


@dataclass
class GanttItem:
    """甘特图项"""
    op_name: str
    unit: str = ""             # "cube" | "vector" | "dma"
    resource: str = ""         # legacy: "DMA" | "Cube" | "Vector"
    start_us: float = 0.0
    end_us: float = 0.0
    category: str = ""         # "execution" | "prefetch" | "parallel"
    label: str = ""
    color: str = ""


@dataclass
class GanttData:
    """甘特图数据"""
    items: List[GanttItem] = field(default_factory=list)
    total_duration_us: float = 0.0
    total_time_us: float = 0.0  # alias
    chip_name: str = ""

    def add_item(self, item: GanttItem):
        self.items.append(item)
        self.total_duration_us = max(self.total_duration_us, item.end_us)
        self.total_time_us = self.total_duration_us
