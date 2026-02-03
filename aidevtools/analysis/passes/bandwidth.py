"""Bandwidth Constraint Pass - 全局带宽约束与流量优化

全局带宽约束:
  当多个算子/流并发执行时，它们共享有限的 HBM 带宽
  effective_bandwidth = total_bandwidth / f(concurrent_streams)

流量约束模式:
  限制总数据搬运量，用于功耗/热设计约束场景

最低流量模式:
  通过 tiling、cache 复用等技术减少实际流量

Example - 全局带宽约束:
    NPU 910: HBM 带宽 = 1200 GB/s
    并发流数 = 4 (Cube DMA + Vector DMA + 预取 + Host)
    contention_model = "linear"
    effective_bandwidth = 1200 / 4 = 300 GB/s

    MatMul 数据量 = 7.1 MB
    原始 memory_time = 7.1 / 1200 = 5.92us
    约束后 memory_time = 7.1 / 300 = 23.67us

Example - 最低流量模式:
    MatMul [1024, 1024] @ [1024, 1024] 权重:
    原始流量 = 1024 * 1024 * 2 = 2MB (fp16)

    启用 L2 缓存复用 (l2_reuse_factor=0.3):
    - 权重分块加载，多次复用
    - 有效流量 = 2MB * 0.3 = 0.6MB

    启用 Tiling (tiling_efficiency=0.8):
    - 输入/输出分块，减少重复搬运
    - 最终流量 = 0.6MB * 0.8 = 0.48MB
"""

import math

from ..constants import BOTTLENECK_COMPUTE, BOTTLENECK_MEMORY, GBPS_TO_BPS, MB_TO_BYTES, S_TO_US
from .base import BasePass, PassContext, PassResult


class BandwidthConstraintPass(BasePass):
    """全局带宽约束 Pass"""

    name = "bandwidth_constraint"
    description = "全局带宽约束，考虑多流并发带宽竞争"
    order = 250
    config_key = "bandwidth_constraint"

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """执行带宽约束分析"""
        latency_before = latency_breakdown.timing.roofline_us
        original_memory_time = latency_breakdown.timing.memory_us

        # 获取并发流数
        concurrent_streams = self.config.bandwidth.concurrent_streams
        if concurrent_streams <= 1:
            # 无并发，无需约束
            result.details = {
                "reason": "单流执行，无带宽竞争",
                "concurrent_streams": concurrent_streams,
            }
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 计算带宽竞争因子
        contention_model = self.config.bandwidth.contention_model
        contention_factor = self._calculate_contention_factor(
            concurrent_streams, contention_model
        )

        # 计算有效带宽
        hbm_bandwidth = chip_spec.memory.hbm.bandwidth_gbps
        effective_bandwidth = hbm_bandwidth / contention_factor

        # 重新计算访存时间

        adjusted_memory_time = original_memory_time * contention_factor

        # 更新 roofline 时延
        new_roofline_time = max(latency_breakdown.timing.compute_us, adjusted_memory_time)

        # 更新瓶颈
        if adjusted_memory_time > latency_breakdown.timing.compute_us:
            latency_breakdown.bottleneck = BOTTLENECK_MEMORY
        else:
            latency_breakdown.bottleneck = BOTTLENECK_COMPUTE

        latency_delta = new_roofline_time - latency_before

        # 更新 breakdown
        latency_breakdown.timing.memory_us = adjusted_memory_time
        latency_breakdown.timing.roofline_us = new_roofline_time

        # 记录有效带宽到 breakdown
        latency_breakdown.bandwidth.effective_gbps = effective_bandwidth

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = new_roofline_time
        result.latency_saved_us = -latency_delta  # 负数表示增加

        result.details = {
            "concurrent_streams": concurrent_streams,
            "contention_model": contention_model,
            "contention_factor": contention_factor,
            "hbm_bandwidth_gbps": hbm_bandwidth,
            "effective_bandwidth_gbps": effective_bandwidth,
            "original_memory_time_us": original_memory_time,
            "adjusted_memory_time_us": adjusted_memory_time,
            "latency_increase_us": latency_delta,
        }

        if latency_delta > 0:
            result.warnings.append(
                f"多流并发({concurrent_streams}流)导致带宽竞争，"
                f"有效带宽降至 {effective_bandwidth:.0f} GB/s ({100/contention_factor:.0f}%)，"
                f"时延增加 {latency_delta:.2f}us"
            )
            result.suggestions.append(
                "考虑减少并发流数或优化流水线调度以减少带宽竞争"
            )

        return result

    def _calculate_contention_factor(self, streams: int, model: str) -> float:
        """计算带宽竞争因子"""
        if model == "none" or streams <= 1:
            return 1.0
        if model == "linear":
            return float(streams)
        if model == "sqrt":
            return math.sqrt(streams)
        return float(streams)


class TrafficConstraintPass(BasePass):
    """流量约束 Pass"""

    name = "traffic_constraint"
    description = "流量约束模式，限制总数据搬运量"
    order = 700
    config_key = "traffic_constraint"

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """执行流量约束检查"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.timing.total_us or latency_breakdown.timing.roofline_us
        total_traffic = profile.total_bytes

        # 获取约束
        max_traffic = self.config.traffic.max_bytes
        budget_mode = self.config.traffic.budget_mode

        # 检查是否超限
        over_budget = 0 < max_traffic < total_traffic
        over_ratio = total_traffic / max_traffic if max_traffic > 0 else 0

        result.latency_before_us = latency_before
        result.latency_after_us = latency_before  # 流量约束不直接影响时延
        result.latency_saved_us = 0

        result.details = {
            "total_traffic_bytes": total_traffic,
            "total_traffic_mb": total_traffic / (1024 * 1024),
            "max_traffic_bytes": max_traffic,
            "max_traffic_mb": max_traffic / (1024 * 1024) if max_traffic > 0 else 0,
            "budget_mode": budget_mode,
            "over_budget": over_budget,
            "over_ratio": over_ratio,
            "traffic_breakdown": {
                "input_bytes": profile.input_bytes,
                "weight_bytes": profile.weight_bytes,
                "output_bytes": profile.output_bytes,
                "workspace_bytes": profile.workspace_bytes,
            }
        }

        if over_budget:
            if budget_mode == "strict":
                result.warnings.append(
                    f"流量超限! 实际 {total_traffic/(1024*1024):.2f}MB > "
                    f"预算 {max_traffic/(1024*1024):.2f}MB ({over_ratio:.1%})"
                )
            elif budget_mode == "soft":
                result.warnings.append(
                    f"流量超出软限制: {total_traffic/(1024*1024):.2f}MB "
                    f"(预算: {max_traffic/(1024*1024):.2f}MB)"
                )

            result.suggestions.append(
                "考虑: 1) 算子融合减少中间张量 2) 权重量化减少搬运量 "
                "3) 激活重计算减少显存"
            )

        return result


class MinTrafficPass(BasePass):
    """最低流量优化 Pass"""

    name = "min_traffic"
    description = "最低流量模式，通过 cache 复用和 tiling 减少流量"
    order = 150
    config_key = "min_traffic_mode"

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """执行最低流量优化"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.timing.roofline_us

        # 获取优化参数
        l2_reuse_factor = self.config.min_traffic.l2_reuse_factor
        tiling_efficiency = self.config.min_traffic.tiling_efficiency
        cache_line_bytes = self.config.min_traffic.cache_line_bytes
        original_traffic = profile.total_bytes

        # 应用 L2 缓存复用 (主要影响权重)
        # 权重通常可以在 L2 中缓存并复用
        effective_weight_bytes = profile.weight_bytes * l2_reuse_factor

        # 应用 Tiling 效率 (影响所有数据)
        # Tiling 可以减少重复搬运
        effective_input_bytes = profile.input_bytes * tiling_efficiency
        effective_output_bytes = profile.output_bytes * tiling_efficiency
        effective_workspace_bytes = profile.workspace_bytes * tiling_efficiency

        # 计算优化后流量
        optimized_traffic = (effective_input_bytes + effective_weight_bytes +
                             effective_output_bytes + effective_workspace_bytes)

        # Cache line 对齐开销
        cache_overhead_ratio = 1.0
        if cache_line_bytes > 0:
            # 小张量可能有 cache line 对齐浪费
            min_dim_bytes = min(
                profile.input_bytes or 1,
                profile.output_bytes or 1
            )
            if min_dim_bytes < cache_line_bytes * 16:
                # 小张量有较大对齐开销
                cache_overhead_ratio = 1.1

        optimized_traffic *= cache_overhead_ratio

        # 计算流量节省比例
        traffic_saved = original_traffic - optimized_traffic
        traffic_saved_ratio = traffic_saved / original_traffic if original_traffic > 0 else 0

        # 重新计算访存时间
        hbm_bandwidth = chip_spec.memory.hbm.bandwidth_gbps
        original_memory_time = latency_breakdown.timing.memory_us
        optimized_memory_time = optimized_traffic / (hbm_bandwidth * GBPS_TO_BPS) * S_TO_US

        # 更新 roofline 时延
        new_roofline_time = max(latency_breakdown.timing.compute_us, optimized_memory_time)

        latency_saved = latency_before - new_roofline_time

        # 更新 breakdown
        latency_breakdown.timing.memory_us = optimized_memory_time
        latency_breakdown.timing.roofline_us = new_roofline_time

        # 更新瓶颈
        if optimized_memory_time > latency_breakdown.timing.compute_us:
            latency_breakdown.bottleneck = BOTTLENECK_MEMORY
        else:
            latency_breakdown.bottleneck = BOTTLENECK_COMPUTE

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = new_roofline_time
        result.latency_saved_us = latency_saved

        result.details = {
            "l2_reuse_factor": l2_reuse_factor,
            "tiling_efficiency": tiling_efficiency,
            "cache_line_bytes": cache_line_bytes,
            "cache_overhead_ratio": cache_overhead_ratio,
            "original_traffic_bytes": original_traffic,
            "original_traffic_mb": original_traffic / MB_TO_BYTES,
            "optimized_traffic_bytes": optimized_traffic,
            "optimized_traffic_mb": optimized_traffic / MB_TO_BYTES,
            "traffic_saved_bytes": traffic_saved,
            "traffic_saved_ratio": traffic_saved_ratio,
            "original_memory_time_us": original_memory_time,
            "optimized_memory_time_us": optimized_memory_time,
            "traffic_breakdown": {
                "effective_input_bytes": effective_input_bytes,
                "effective_weight_bytes": effective_weight_bytes,
                "effective_output_bytes": effective_output_bytes,
                "effective_workspace_bytes": effective_workspace_bytes,
            }
        }

        if traffic_saved_ratio > 0.1:
            result.suggestions.append(
                f"通过 L2 复用 ({l2_reuse_factor:.0%}) 和 Tiling ({tiling_efficiency:.0%})，"
                f"流量从 {original_traffic/(1024*1024):.2f}MB 降至 "
                f"{optimized_traffic/(1024*1024):.2f}MB，节省 {traffic_saved_ratio:.0%}"
            )

        if l2_reuse_factor >= 1.0:
            result.suggestions.append(
                "权重未被缓存复用，考虑调整分块策略以利用 L2 缓存"
            )

        if tiling_efficiency >= 1.0:
            result.suggestions.append(
                "未启用 Tiling 优化，考虑对大张量进行分块以减少流量"
            )

        return result
