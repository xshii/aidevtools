"""PaperAnalyzer - 模型时延分析主类

功能:
- 收集算子 profile 信息
- 执行 Pass 链进行时延分析
- 生成时延、带宽、算力分析报告
- 支持流水图 (Gantt) 可视化
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from .profile import OpProfile
from .chip import ChipSpec, load_chip_spec
from .latency import LatencyBreakdown, LatencyResult, GanttItem, GanttData
from .passes import (
    PassConfig,
    PassResult,
    PassPreset,
    RooflinePass,
    MinTrafficPass,
    MemoryEfficiencyPass,
    BandwidthConstraintPass,
    ForwardPrefetchPass,
    BackwardPrefetchPass,
    CubeVectorParallelPass,
    OverheadPass,
    TrafficConstraintPass,
)


@dataclass
class AnalysisSummary:
    """分析摘要"""
    total_latency_us: float = 0.0
    total_compute_time_us: float = 0.0
    total_memory_time_us: float = 0.0
    total_flops: int = 0
    total_bytes: int = 0

    # 瓶颈统计
    compute_bound_ops: int = 0
    memory_bound_ops: int = 0

    # 优化效果
    total_prefetch_saved_us: float = 0.0
    total_parallel_saved_us: float = 0.0
    total_overhead_us: float = 0.0

    # 吞吐量
    achieved_tflops: float = 0.0
    achieved_bandwidth_gbps: float = 0.0

    # Cube/Vector 占比
    cube_time_us: float = 0.0
    vector_time_us: float = 0.0

    # 流量分析 (新增)
    total_original_traffic_bytes: int = 0   # 原始总流量
    total_optimized_traffic_bytes: int = 0  # 优化后总流量
    traffic_saved_ratio: float = 0.0        # 流量节省比例
    effective_bandwidth_gbps: float = 0.0   # 平均有效带宽


class PaperAnalyzer:
    """模型时延分析器

    基本用法:
        >>> analyzer = PaperAnalyzer(chip="npu_910")
        >>> analyzer.add_profile(profile)
        >>> result = analyzer.analyze()
        >>> print(result.summary.total_latency_us)

    使用工厂方法:
        >>> analyzer = PaperAnalyzer.from_chip_name("npu_910")
        >>> analyzer = PaperAnalyzer.from_chip_spec(my_chip_spec)

    高级配置:
        >>> config = PassConfig.from_preset(PassPreset.AGGRESSIVE)
        >>> analyzer = PaperAnalyzer(chip="npu_910", pass_config=config)

    导出结果:
        >>> from aidevtools.analysis import export_xlsx
        >>> export_xlsx(result, "report.xlsx")
    """

    def __init__(self,
                 chip: str = "npu_910",
                 chip_spec: ChipSpec = None,
                 pass_config: PassConfig = None):
        """
        初始化分析器。

        Args:
            chip: 芯片名称，支持 "npu_310", "npu_910", "gpu_a100"
            chip_spec: 自定义芯片规格 (如果提供，chip 参数被忽略)
            pass_config: Pass 配置 (默认使用 STANDARD 预设)

        Raises:
            ValueError: 当 chip 名称无效且未提供 chip_spec 时
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

    @classmethod
    def from_chip_name(cls, chip_name: str,
                       pass_config: PassConfig = None) -> 'PaperAnalyzer':
        """
        使用芯片名称创建分析器。

        Args:
            chip_name: 芯片名称 ("npu_310", "npu_910", "gpu_a100")
            pass_config: Pass 配置

        Returns:
            PaperAnalyzer 实例

        Example:
            >>> analyzer = PaperAnalyzer.from_chip_name("npu_910")
        """
        return cls(chip=chip_name, pass_config=pass_config)

    @classmethod
    def from_chip_spec(cls, chip_spec: ChipSpec,
                       pass_config: PassConfig = None) -> 'PaperAnalyzer':
        """
        使用自定义芯片规格创建分析器。

        Args:
            chip_spec: 芯片规格对象
            pass_config: Pass 配置

        Returns:
            PaperAnalyzer 实例

        Example:
            >>> chip = ChipSpec(name="Custom NPU", ...)
            >>> analyzer = PaperAnalyzer.from_chip_spec(chip)
        """
        return cls(chip_spec=chip_spec, pass_config=pass_config)

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

            # 1. Roofline Pass
            roofline_pass = RooflinePass(self.pass_config)
            result = roofline_pass.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 1.5. Min Traffic Pass (最低流量优化)
            min_traffic_pass = MinTrafficPass(self.pass_config)
            result = min_traffic_pass.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 2. Memory Efficiency Pass
            mem_pass = MemoryEfficiencyPass(self.pass_config)
            result = mem_pass.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 2.5. Bandwidth Constraint Pass (全局带宽约束)
            bw_pass = BandwidthConstraintPass(self.pass_config)
            result = bw_pass.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 3. Forward Prefetch Pass
            next_weight_bytes = 0
            next_op_name = ""
            if i + 1 < len(self._profiles):
                next_profile = self._profiles[i + 1]
                next_weight_bytes = next_profile.weight_bytes
                next_op_name = next_profile.name

            fwd_prefetch = ForwardPrefetchPass(
                self.pass_config,
                next_op_weight_bytes=next_weight_bytes,
                next_op_name=next_op_name
            )
            result = fwd_prefetch.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 4. Backward Prefetch Pass
            future_cube_ops = []
            for j in range(i + 1, min(i + 1 + self.pass_config.backward_prefetch_depth,
                                      len(self._profiles))):
                future_profile = self._profiles[j]
                if future_profile.compute_unit == "cube":
                    future_cube_ops.append({
                        "name": future_profile.name,
                        "weight_bytes": future_profile.weight_bytes
                    })

            bwd_prefetch = BackwardPrefetchPass(
                self.pass_config,
                future_cube_ops=future_cube_ops
            )
            result = bwd_prefetch.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 5. Cube/Vector Parallel Pass
            adjacent_unit = ""
            adjacent_time = 0.0
            adjacent_name = ""
            if i + 1 < len(self._profiles):
                next_profile = self._profiles[i + 1]
                adjacent_unit = next_profile.compute_unit
                adjacent_name = next_profile.name
                # 需要先分析下一个算子才能获取时间，这里用估算
                if i > 0 and self._breakdowns:
                    # 使用前一个同类型算子的时间估算
                    for prev_bd in reversed(self._breakdowns):
                        if prev_bd.profile.compute_unit == adjacent_unit:
                            adjacent_time = prev_bd.roofline_time_us
                            break

            parallel_pass = CubeVectorParallelPass(
                self.pass_config,
                adjacent_op_unit=adjacent_unit,
                adjacent_op_time_us=adjacent_time,
                adjacent_op_name=adjacent_name
            )
            result = parallel_pass.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 6. Overhead Pass
            overhead_pass = OverheadPass(self.pass_config)
            result = overhead_pass.run(breakdown, self.chip_spec)
            op_pass_results.append(result)

            # 7. Traffic Constraint Pass (流量约束检查)
            traffic_pass = TrafficConstraintPass(self.pass_config)
            result = traffic_pass.run(breakdown, self.chip_spec)
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
            summary.total_latency_us += bd.total_time_us
            summary.total_compute_time_us += bd.compute_time_us
            summary.total_memory_time_us += bd.memory_time_us
            summary.total_flops += bd.profile.flops
            summary.total_bytes += (bd.profile.input_bytes + bd.profile.weight_bytes +
                                    bd.profile.output_bytes)

            if bd.bottleneck == "compute":
                summary.compute_bound_ops += 1
            else:
                summary.memory_bound_ops += 1

            summary.total_prefetch_saved_us += bd.prefetch_saved_us + bd.backward_prefetch_saved_us
            summary.total_parallel_saved_us += bd.parallel_saved_us
            summary.total_overhead_us += bd.overhead_us

            if bd.profile.compute_unit == "cube":
                summary.cube_time_us += bd.roofline_time_us
            else:
                summary.vector_time_us += bd.roofline_time_us

            # 流量统计
            original = (bd.profile.input_bytes + bd.profile.weight_bytes +
                        bd.profile.output_bytes + bd.profile.workspace_bytes)
            summary.total_original_traffic_bytes += original
            if bd.optimized_traffic_bytes > 0:
                summary.total_optimized_traffic_bytes += bd.optimized_traffic_bytes
            else:
                summary.total_optimized_traffic_bytes += original

            # 有效带宽统计
            if bd.effective_bandwidth_gbps > 0:
                total_effective_bw += bd.effective_bandwidth_gbps
                bw_count += 1

        # 计算实际吞吐量
        if summary.total_latency_us > 0:
            summary.achieved_tflops = summary.total_flops / (summary.total_latency_us * 1e-6) / 1e12
            summary.achieved_bandwidth_gbps = summary.total_bytes / (summary.total_latency_us * 1e-6) / 1e9

        # 流量节省比例
        if summary.total_original_traffic_bytes > 0:
            summary.traffic_saved_ratio = (
                1 - summary.total_optimized_traffic_bytes / summary.total_original_traffic_bytes
            )

        # 平均有效带宽
        if bw_count > 0:
            summary.effective_bandwidth_gbps = total_effective_bw / bw_count

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
                end_us=current_time + bd.total_time_us,
                category="execution",
            )
            items.append(item)

            # 如果有预取，添加预取条目
            if bd.prefetch_saved_us > 0:
                prefetch_item = GanttItem(
                    op_name=f"{bd.profile.name}_prefetch",
                    unit="dma",
                    start_us=current_time,
                    end_us=current_time + bd.prefetch_saved_us,
                    category="prefetch",
                )
                items.append(prefetch_item)

            current_time += bd.total_time_us

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
        print(f"\n{'='*60}")
        print(f"Paper Analysis Summary - {self.chip_spec.name}")
        print(f"{'='*60}")
        print(f"Total Operators: {len(self._profiles)}")
        print(f"Total Latency: {s.total_latency_us:.2f} us ({s.total_latency_us/1000:.3f} ms)")
        print(f"\n--- Breakdown ---")
        print(f"Compute Time: {s.total_compute_time_us:.2f} us")
        print(f"Memory Time: {s.total_memory_time_us:.2f} us")
        print(f"Overhead: {s.total_overhead_us:.2f} us")
        print(f"\n--- Bottleneck ---")
        print(f"Compute Bound Ops: {s.compute_bound_ops}")
        print(f"Memory Bound Ops: {s.memory_bound_ops}")
        print(f"\n--- Optimizations ---")
        print(f"Prefetch Saved: {s.total_prefetch_saved_us:.2f} us")
        print(f"Parallel Saved: {s.total_parallel_saved_us:.2f} us")
        print(f"\n--- Throughput ---")
        print(f"Achieved TFLOPS: {s.achieved_tflops:.2f}")
        print(f"Achieved Bandwidth: {s.achieved_bandwidth_gbps:.2f} GB/s")
        if s.effective_bandwidth_gbps > 0:
            print(f"Effective Bandwidth (contention): {s.effective_bandwidth_gbps:.2f} GB/s")
        print(f"\n--- Traffic Analysis ---")
        print(f"Original Traffic: {s.total_original_traffic_bytes/(1024*1024):.2f} MB")
        print(f"Optimized Traffic: {s.total_optimized_traffic_bytes/(1024*1024):.2f} MB")
        if s.traffic_saved_ratio > 0:
            print(f"Traffic Saved: {s.traffic_saved_ratio*100:.1f}%")
        print(f"\n--- Unit Utilization ---")
        print(f"Cube Time: {s.cube_time_us:.2f} us ({s.cube_time_us/s.total_latency_us*100:.1f}%)")
        print(f"Vector Time: {s.vector_time_us:.2f} us ({s.vector_time_us/s.total_latency_us*100:.1f}%)")
        print(f"{'='*60}\n")

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
                "Compute Time (us)": bd.compute_time_us,
                "Memory Time (us)": bd.memory_time_us,
                "Roofline Time (us)": bd.roofline_time_us,
                "Prefetch Saved (us)": bd.prefetch_saved_us,
                "Parallel Saved (us)": bd.parallel_saved_us,
                "Overhead (us)": bd.overhead_us,
                "Total Time (us)": bd.total_time_us,
                "Bottleneck": bd.bottleneck,
                "Min Bandwidth (GB/s)": bd.min_bandwidth_gbps,
            }
            rows.append(row)

        return pd.DataFrame(rows)
