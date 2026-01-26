"""PaperAnalyzer - 模型时延分析主类

功能:
- 收集算子 profile 信息
- 执行 Pass 链进行时延分析
- 生成时延、带宽、算力分析报告
- 支持流水图 (Gantt) 可视化
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .chip import ChipSpec, load_chip_spec
from .latency import GanttData, GanttItem, LatencyBreakdown, LatencyResult
from .passes import (
    BackwardPrefetchPass,
    BandwidthConstraintPass,
    CubeVectorParallelPass,
    ForwardPrefetchPass,
    MemoryEfficiencyPass,
    MinTrafficPass,
    OverheadPass,
    PassConfig,
    PassContext,
    PassPreset,
    PassResult,
    RooflinePass,
    TrafficConstraintPass,
)
from .profile import OpProfile


@dataclass
class SummaryTotals:
    """汇总指标"""

    latency_us: float = 0.0
    compute_time_us: float = 0.0
    memory_time_us: float = 0.0
    flops: int = 0
    bytes: int = 0


@dataclass
class BottleneckStats:
    """瓶颈统计"""

    compute_bound_ops: int = 0
    memory_bound_ops: int = 0


@dataclass
class OptimizationStats:
    """优化效果统计"""

    prefetch_saved_us: float = 0.0
    parallel_saved_us: float = 0.0
    overhead_us: float = 0.0


@dataclass
class ThroughputStats:
    """吞吐量统计"""

    achieved_tflops: float = 0.0
    achieved_bandwidth_gbps: float = 0.0


@dataclass
class UnitStats:
    """计算单元时间统计"""

    cube_time_us: float = 0.0
    vector_time_us: float = 0.0


@dataclass
class TrafficStats:
    """流量统计"""

    original_bytes: int = 0
    optimized_bytes: int = 0
    saved_ratio: float = 0.0
    effective_bandwidth_gbps: float = 0.0


@dataclass
class AnalysisSummary:
    """分析摘要 (使用组合模式，无兼容性属性)"""

    # === 组合子对象 ===
    totals: SummaryTotals = field(default_factory=SummaryTotals)
    bottleneck: BottleneckStats = field(default_factory=BottleneckStats)
    optimization: OptimizationStats = field(default_factory=OptimizationStats)
    throughput: ThroughputStats = field(default_factory=ThroughputStats)
    unit: UnitStats = field(default_factory=UnitStats)
    traffic: TrafficStats = field(default_factory=TrafficStats)


class PaperAnalyzer:
    """模型时延分析器

    用法:
        >>> analyzer = PaperAnalyzer(chip="npu_910")
        >>> analyzer.add_profile(profile)
        >>> result = analyzer.analyze()
        >>> print(result.summary.total_latency_us)

    高级配置:
        >>> config = PassConfig.from_preset(PassPreset.AGGRESSIVE)
        >>> analyzer = PaperAnalyzer(chip="npu_910", pass_config=config)

    导出结果:
        >>> from aidevtools.analysis import export_xlsx
        >>> export_xlsx(result, "report.xlsx")
    """

    def __init__(
        self, chip: str = "npu_910", chip_spec: ChipSpec = None, pass_config: PassConfig = None
    ):
        """
        初始化分析器。

        Args:
            chip: 芯片名称，支持 "npu_310", "npu_910", "gpu_a100"
            chip_spec: 自定义芯片规格 (如果提供，chip 参数被忽略)
            pass_config: Pass 配置 (默认使用 STANDARD 预设)
        """
        self.chip_spec = chip_spec or load_chip_spec(chip)
        self.pass_config = pass_config or PassConfig.from_preset(PassPreset.STANDARD)

        # 算子 profile 列表
        self._profiles: List[OpProfile] = []

        # 分析结果
        self._breakdowns: List[LatencyBreakdown] = []
        self._pass_results: List[List[PassResult]] = []
        self._summary: Optional[AnalysisSummary] = None
        self._gantt_data: Optional[GanttData] = None

    def add_profile(self, profile: OpProfile):
        """添加算子 profile"""
        self._profiles.append(profile)

    def add_profiles(self, profiles: List[OpProfile]):
        """批量添加算子 profile"""
        self._profiles.extend(profiles)

    def clear_profiles(self):
        """清空 profiles"""
        self._profiles.clear()
        self._breakdowns.clear()
        self._pass_results.clear()
        self._summary = None
        self._gantt_data = None

    def analyze(self) -> LatencyResult:
        """执行分析

        Returns:
            LatencyResult 包含所有算子的时延分析结果
        """
        self._breakdowns = []
        self._pass_results = []

        # 对每个算子执行 Pass 链
        for i, profile in enumerate(self._profiles):
            breakdown = LatencyBreakdown(profile=profile)
            op_pass_results = []

            # 构建 PassContext
            depth = self.pass_config.prefetch.backward_depth
            context = PassContext(
                next_profile=self._profiles[i + 1] if i + 1 < len(self._profiles) else None,
                prev_profile=self._profiles[i - 1] if i > 0 else None,
                future_profiles=self._profiles[i + 1 : i + 1 + depth],
            )

            # Pass 列表
            passes = [
                RooflinePass(self.pass_config),
                MinTrafficPass(self.pass_config),
                MemoryEfficiencyPass(self.pass_config),
                BandwidthConstraintPass(self.pass_config),
                ForwardPrefetchPass(self.pass_config),
                BackwardPrefetchPass(self.pass_config),
                CubeVectorParallelPass(self.pass_config),
                OverheadPass(self.pass_config),
                TrafficConstraintPass(self.pass_config),
            ]

            for p in passes:
                result = p.run(breakdown, self.chip_spec, context)
                op_pass_results.append(result)

            self._breakdowns.append(breakdown)
            self._pass_results.append(op_pass_results)

        # 生成摘要
        self._summary = self._generate_summary()

        # 生成 Gantt 数据
        self._gantt_data = self._generate_gantt()

        return LatencyResult(
            chip_spec=self.chip_spec,
            pass_config=self.pass_config,
            breakdowns=self._breakdowns,
            pass_results=self._pass_results,
            summary=self._summary,
            gantt_data=self._gantt_data,
        )

    def _generate_summary(self) -> AnalysisSummary:
        """生成分析摘要"""
        summary = AnalysisSummary()

        total_effective_bw = 0.0
        bw_count = 0

        for bd in self._breakdowns:
            summary.totals.latency_us += bd.timing.total_us
            summary.totals.compute_time_us += bd.timing.compute_us
            summary.totals.memory_time_us += bd.timing.memory_us
            summary.totals.flops += bd.profile.flops
            summary.totals.bytes += (
                bd.profile.input_bytes + bd.profile.weight_bytes + bd.profile.output_bytes
            )

            if bd.bottleneck == "compute":
                summary.bottleneck.compute_bound_ops += 1
            else:
                summary.bottleneck.memory_bound_ops += 1

            summary.optimization.prefetch_saved_us += (
                bd.savings.prefetch_us + bd.savings.backward_prefetch_us
            )
            summary.optimization.parallel_saved_us += bd.savings.parallel_us
            summary.optimization.overhead_us += bd.timing.overhead_us

            if bd.profile.compute_unit == "cube":
                summary.unit.cube_time_us += bd.timing.roofline_us
            else:
                summary.unit.vector_time_us += bd.timing.roofline_us

            # 流量统计
            original = (
                bd.profile.input_bytes
                + bd.profile.weight_bytes
                + bd.profile.output_bytes
                + bd.profile.workspace_bytes
            )
            summary.traffic.original_bytes += original
            if bd.traffic.optimized_bytes > 0:
                summary.traffic.optimized_bytes += bd.traffic.optimized_bytes
            else:
                summary.traffic.optimized_bytes += original

            # 有效带宽统计
            if bd.bandwidth.effective_gbps > 0:
                total_effective_bw += bd.bandwidth.effective_gbps
                bw_count += 1

        # 计算实际吞吐量
        if summary.totals.latency_us > 0:
            summary.throughput.achieved_tflops = (
                summary.totals.flops / (summary.totals.latency_us * 1e-6) / 1e12
            )
            summary.throughput.achieved_bandwidth_gbps = (
                summary.totals.bytes / (summary.totals.latency_us * 1e-6) / 1e9
            )

        # 流量节省比例
        if summary.traffic.original_bytes > 0:
            summary.traffic.saved_ratio = (
                1 - summary.traffic.optimized_bytes / summary.traffic.original_bytes
            )

        # 平均有效带宽
        if bw_count > 0:
            summary.traffic.effective_bandwidth_gbps = total_effective_bw / bw_count

        return summary

    def _generate_gantt(self) -> GanttData:
        """生成 Gantt 图数据"""
        items = []
        current_time = 0.0

        for bd in self._breakdowns:
            # 主执行时间
            item = GanttItem(
                op_name=bd.profile.name,
                unit=bd.profile.compute_unit,
                start_us=current_time,
                end_us=current_time + bd.timing.total_us,
                category="execution",
            )
            items.append(item)

            # 如果有预取，添加预取条目
            if bd.savings.prefetch_us > 0:
                prefetch_item = GanttItem(
                    op_name=f"{bd.profile.name}_prefetch",
                    unit="dma",
                    start_us=current_time,
                    end_us=current_time + bd.savings.prefetch_us,
                    category="prefetch",
                )
                items.append(prefetch_item)

            current_time += bd.timing.total_us

        return GanttData(
            items=items,
            total_duration_us=current_time,
            chip_name=self.chip_spec.name,
        )

    def get_result(self) -> Optional[LatencyResult]:
        """获取分析结果"""
        if not self._breakdowns:
            return None
        return LatencyResult(
            chip_spec=self.chip_spec,
            pass_config=self.pass_config,
            breakdowns=self._breakdowns,
            pass_results=self._pass_results,
            summary=self._summary,
            gantt_data=self._gantt_data,
        )

    def get_summary(self) -> Optional[AnalysisSummary]:
        """获取摘要"""
        return self._summary

    def get_gantt_data(self) -> Optional[GanttData]:
        """获取 Gantt 数据"""
        return self._gantt_data

    def print_summary(self):
        """打印摘要"""
        if not self._summary:
            print("No analysis result. Call analyze() first.")
            return

        s = self._summary
        print(f"\n{'=' * 60}")
        print(f"Paper Analysis Summary - {self.chip_spec.name}")
        print(f"{'=' * 60}")
        print(f"Total Operators: {len(self._profiles)}")
        print(f"Total Latency: {s.totals.latency_us:.2f} us ({s.totals.latency_us / 1000:.3f} ms)")
        print("\n--- Breakdown ---")
        print(f"Compute Time: {s.totals.compute_time_us:.2f} us")
        print(f"Memory Time: {s.totals.memory_time_us:.2f} us")
        print(f"Overhead: {s.optimization.overhead_us:.2f} us")
        print("\n--- Bottleneck ---")
        print(f"Compute Bound Ops: {s.bottleneck.compute_bound_ops}")
        print(f"Memory Bound Ops: {s.bottleneck.memory_bound_ops}")
        print("\n--- Optimizations ---")
        print(f"Prefetch Saved: {s.optimization.prefetch_saved_us:.2f} us")
        print(f"Parallel Saved: {s.optimization.parallel_saved_us:.2f} us")
        print("\n--- Throughput ---")
        print(f"Achieved TFLOPS: {s.throughput.achieved_tflops:.2f}")
        print(f"Achieved Bandwidth: {s.throughput.achieved_bandwidth_gbps:.2f} GB/s")
        if s.traffic.effective_bandwidth_gbps > 0:
            print(
                f"Effective Bandwidth (contention): {s.traffic.effective_bandwidth_gbps:.2f} GB/s"
            )
        print("\n--- Traffic Analysis ---")
        print(f"Original Traffic: {s.traffic.original_bytes / (1024 * 1024):.2f} MB")
        print(f"Optimized Traffic: {s.traffic.optimized_bytes / (1024 * 1024):.2f} MB")
        if s.traffic.saved_ratio > 0:
            print(f"Traffic Saved: {s.traffic.saved_ratio * 100:.1f}%")
        print("\n--- Unit Utilization ---")
        cube_pct = s.unit.cube_time_us / s.totals.latency_us * 100
        vec_pct = s.unit.vector_time_us / s.totals.latency_us * 100
        print(f"Cube Time: {s.unit.cube_time_us:.2f} us ({cube_pct:.1f}%)")
        print(f"Vector Time: {s.unit.vector_time_us:.2f} us ({vec_pct:.1f}%)")
        print(f"{'=' * 60}\n")

    def to_dataframe(self):
        """转换为 pandas DataFrame"""
        import pandas as pd

        rows = []
        for bd in self._breakdowns:
            p = bd.profile
            row = {
                "Op Name": p.name,
                "Op Type": p.op_type,
                "Compute Unit": p.compute_unit,
                "Dtype": p.dtype,
                "FLOPs": p.flops,
                "Input Bytes": p.input_bytes,
                "Weight Bytes": p.weight_bytes,
                "Output Bytes": p.output_bytes,
                "Compute Time (us)": bd.timing.compute_us,
                "Memory Time (us)": bd.timing.memory_us,
                "Roofline Time (us)": bd.timing.roofline_us,
                "Prefetch Saved (us)": bd.savings.prefetch_us,
                "Parallel Saved (us)": bd.savings.parallel_us,
                "Overhead (us)": bd.timing.overhead_us,
                "Total Time (us)": bd.timing.total_us,
                "Bottleneck": bd.bottleneck,
                "Min Bandwidth (GB/s)": bd.bandwidth.min_gbps,
            }
            rows.append(row)

        return pd.DataFrame(rows)
