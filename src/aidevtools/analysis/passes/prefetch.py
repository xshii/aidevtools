"""Prefetch Passes - 预取优化

前向预取 (Forward Prefetch):
  当前算子 Cube 计算时，DMA 可以预取下一个算子的权重
  saved_time = min(current_compute_time, next_weight_load_time) * prefetch_efficiency

后向预取 (Backward Prefetch):
  当前算子 Vector 计算时，DMA 可以预取后续 Cube 算子的权重

Example - Forward Prefetch:
    当前: MatMul (Cube), compute=10us, memory=6us => idle=4us
    下一个: Linear weight=2MB, load_time=1.67us
    prefetch_efficiency = 0.8
    prefetchable = min(4, 1.67) * 0.8 = 1.34us
    => 下一个算子权重加载时间节省 1.34us

Example - Backward Prefetch:
    当前: LayerNorm (Vector), roofline_time=8us
    后续 Cube 算子: [FFN1 weight=4MB, FFN2 weight=4MB]
    在 Vector 执行的 8us 内可预取 FFN1/FFN2 的权重
    prefetch_depth=2, efficiency=0.8
    total_saved = min(8, 3.33+3.33) * 0.8 = 5.33us
"""

from typing import List, Optional
from .base import BasePass, PassConfig, PassResult, PassContext


class ForwardPrefetchPass(BasePass):
    """前向预取优化 Pass"""

    name = "forward_prefetch"
    description = "Cube 计算时预取下一算子权重"
    order = 3

    def __init__(self, config: PassConfig = None,
                 next_op_weight_bytes: int = 0,
                 next_op_name: str = ""):
        """
        Args:
            next_op_weight_bytes: 下一算子权重字节数 (deprecated, 使用 PassContext)
            next_op_name: 下一算子名称 (deprecated, 使用 PassContext)
        """
        super().__init__(config)
        self.next_op_weight_bytes = next_op_weight_bytes
        self.next_op_name = next_op_name

    def is_enabled(self) -> bool:
        return self.config.enabled and self.config.forward_prefetch_enabled

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """执行前向预取优化"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.roofline_time_us

        # 只有 Cube 算子才能执行前向预取
        if profile.compute_unit != "cube":
            result.details = {"reason": "非 Cube 算子，无法执行前向预取"}
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 计算当前算子的空闲 DMA 时间 (compute_time - memory_time)
        compute_time = latency_breakdown.compute_time_us
        memory_time = latency_breakdown.memory_time_us

        # 只有计算瓶颈时才有空闲时间用于预取
        idle_time = max(0, compute_time - memory_time)

        if idle_time <= 0 or self.next_op_weight_bytes <= 0:
            result.details = {
                "reason": "无空闲时间或无后续权重",
                "idle_time_us": idle_time,
                "next_weight_bytes": self.next_op_weight_bytes
            }
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 计算预取下一个算子权重所需时间
        hbm_bandwidth = chip_spec.memory.hbm.bandwidth_gbps
        next_weight_load_time = self.next_op_weight_bytes / (hbm_bandwidth * 1e9) * 1e6

        # 可预取时间 = min(空闲时间, 权重加载时间) * 效率
        prefetch_efficiency = self.config.prefetch_efficiency
        prefetchable_time = min(idle_time, next_weight_load_time) * prefetch_efficiency

        # 更新 breakdown
        latency_breakdown.prefetch_saved_us = prefetchable_time

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = latency_before  # 本算子时延不变，节省的是下一个算子
        result.latency_saved_us = prefetchable_time

        result.details = {
            "compute_time_us": compute_time,
            "memory_time_us": memory_time,
            "idle_time_us": idle_time,
            "next_op_name": self.next_op_name,
            "next_weight_bytes": self.next_op_weight_bytes,
            "next_weight_load_time_us": next_weight_load_time,
            "prefetch_efficiency": prefetch_efficiency,
            "prefetchable_time_us": prefetchable_time,
        }

        if prefetchable_time > 0:
            result.suggestions.append(
                f"可预取下一算子 '{self.next_op_name}' 的 {self.next_op_weight_bytes/1024/1024:.2f}MB 权重，"
                f"节省 {prefetchable_time:.2f}us"
            )

        return result


class BackwardPrefetchPass(BasePass):
    """后向预取优化 Pass"""

    name = "backward_prefetch"
    description = "Vector 计算时预取后续 Cube 算子权重"
    order = 4

    def __init__(self, config: PassConfig = None,
                 future_cube_ops: List[dict] = None):
        """
        Args:
            future_cube_ops: 后续 Cube 算子列表 (deprecated, 使用 PassContext)
                每项包含: name (算子名), weight_bytes (权重字节数)
        """
        super().__init__(config)
        self.future_cube_ops = future_cube_ops or []

    def is_enabled(self) -> bool:
        return self.config.enabled and self.config.backward_prefetch_enabled

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """执行后向预取优化"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.roofline_time_us

        # 只有 Vector 算子才能执行后向预取
        if profile.compute_unit != "vector":
            result.details = {"reason": "非 Vector 算子，无法执行后向预取"}
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # Vector 算子的完整执行时间都可用于预取
        available_time = latency_breakdown.roofline_time_us

        if available_time <= 0 or not self.future_cube_ops:
            result.details = {
                "reason": "无可用时间或无后续 Cube 算子",
                "available_time_us": available_time,
                "future_cube_ops": len(self.future_cube_ops)
            }
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 在预取深度内预取后续 Cube 算子权重
        hbm_bandwidth = chip_spec.memory.hbm.bandwidth_gbps
        prefetch_depth = self.config.backward_prefetch_depth
        prefetch_efficiency = self.config.prefetch_efficiency

        total_prefetched = 0
        total_saved = 0.0
        prefetch_details = []

        remaining_time = available_time
        for i, op in enumerate(self.future_cube_ops[:prefetch_depth]):
            if remaining_time <= 0:
                break

            weight_load_time = op["weight_bytes"] / (hbm_bandwidth * 1e9) * 1e6
            prefetch_time = min(remaining_time, weight_load_time) * prefetch_efficiency

            prefetch_details.append({
                "op_name": op["name"],
                "weight_bytes": op["weight_bytes"],
                "weight_load_time_us": weight_load_time,
                "prefetched_time_us": prefetch_time,
            })

            total_prefetched += op["weight_bytes"]
            total_saved += prefetch_time
            remaining_time -= weight_load_time

        # 更新 breakdown
        latency_breakdown.backward_prefetch_saved_us = total_saved

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = latency_before
        result.latency_saved_us = total_saved

        result.details = {
            "available_time_us": available_time,
            "prefetch_depth": prefetch_depth,
            "prefetch_efficiency": prefetch_efficiency,
            "total_prefetched_bytes": total_prefetched,
            "total_saved_us": total_saved,
            "prefetch_details": prefetch_details,
        }

        if total_saved > 0:
            result.suggestions.append(
                f"Vector 算子执行期间可预取 {len(prefetch_details)} 个后续 Cube 算子的权重 "
                f"({total_prefetched/1024/1024:.2f}MB)，总计节省 {total_saved:.2f}us"
            )

        return result
