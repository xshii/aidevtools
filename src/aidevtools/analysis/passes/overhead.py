"""Overhead Pass - 开销计算

计算各类系统开销:
- Kernel 启动开销
- 同步开销
- 内存分配开销

Example:
    MatMul 算子 on NPU 910:
    - roofline_time = 10us
    - kernel_launch_us = 3us (NPU 910 配置)
    - sync_overhead_us = 1us
    - total_overhead = 3 + 1 = 4us

    最终时延计算:
    final = roofline + overhead - prefetch_saved - parallel_saved
    final = 10 + 4 - 1.5 - 0 = 12.5us

    警告: 如果 overhead/final > 10%，建议算子融合减少 kernel 数
"""

from .base import BasePass, PassConfig, PassResult, PassContext


class OverheadPass(BasePass):
    """开销计算 Pass"""

    name = "overhead"
    description = "计算 kernel 启动、同步等开销"
    order = 6

    def is_enabled(self) -> bool:
        return self.config.enabled and self.config.overhead_enabled

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """计算开销"""
        latency_before = latency_breakdown.roofline_time_us

        # 获取开销参数
        kernel_launch_us = self.config.kernel_launch_us
        sync_overhead_us = self.config.sync_overhead_us

        # 总开销
        total_overhead = kernel_launch_us + sync_overhead_us

        # 更新 breakdown
        latency_breakdown.overhead_us = total_overhead

        # 计算最终时延
        final_latency = latency_before + total_overhead

        # 减去预取和并行节省
        final_latency -= latency_breakdown.prefetch_saved_us
        final_latency -= latency_breakdown.backward_prefetch_saved_us
        final_latency -= latency_breakdown.parallel_saved_us

        # 确保不为负
        final_latency = max(0, final_latency)

        latency_breakdown.total_time_us = final_latency

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = final_latency
        result.latency_saved_us = latency_before - final_latency

        result.details = {
            "kernel_launch_us": kernel_launch_us,
            "sync_overhead_us": sync_overhead_us,
            "total_overhead_us": total_overhead,
            "prefetch_saved_us": latency_breakdown.prefetch_saved_us,
            "backward_prefetch_saved_us": latency_breakdown.backward_prefetch_saved_us,
            "parallel_saved_us": latency_breakdown.parallel_saved_us,
            "roofline_time_us": latency_before,
            "final_latency_us": final_latency,
        }

        # 检查开销占比
        overhead_ratio = total_overhead / final_latency if final_latency > 0 else 0
        if overhead_ratio > 0.1:
            result.warnings.append(
                f"开销占比较高 ({overhead_ratio*100:.1f}%)，考虑算子融合减少 kernel 数量"
            )

        return result
