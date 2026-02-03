"""Overhead Pass - 开销计算

计算各类系统开销:
- Kernel 启动开销 (kernel_launch_us)
- 同步开销 (sync_overhead_us)
- 算子切换时延 (context_switch_us)
- Tiling 调度开销 (tiling_overhead_us × tiling_count)

Example:
    MatMul [4096, 4096] @ [4096, 4096] on NPU 910:
    - roofline_time = 10us
    - kernel_launch_us = 5us
    - sync_overhead_us = 2us
    - context_switch_us = 1us
    - tiling_overhead_us = 0.5us, tiling_count = 4 (2x2 分块)
    - total_overhead = 5 + 2 + 1 + 0.5*4 = 10us

    最终时延计算:
    final = roofline + overhead - prefetch_saved - parallel_saved
    final = 10 + 10 - 1.5 - 0 = 18.5us

Tiling Count 计算:
    对于 MatMul [M, K] @ [K, N]:
    - 如果 M*N > L2_SIZE，需要分块
    - tiling_count ≈ ceil(M/tile_m) * ceil(N/tile_n)
    - 典型 tile_size: 256~1024 (根据 L2 大小)

警告阈值:
    如果 overhead/final > 10%，建议算子融合减少 kernel 数
"""

from .base import BasePass, PassContext, PassResult


class OverheadPass(BasePass):
    """开销计算 Pass

    计算各类系统开销:
    - kernel_launch: kernel 启动开销
    - sync: 同步开销
    - context_switch: 算子切换时延
    - tiling: Tiling 调度开销 (per tile × tile count)
    """

    name = "overhead"
    description = "计算 kernel 启动、同步、切换、tiling 等开销"
    order = 600
    config_key = "overhead"

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """计算开销"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.timing.roofline_us

        # 获取开销参数
        kernel_launch_us = self.config.overhead.kernel_launch_us
        sync_overhead_us = self.config.overhead.sync_us
        context_switch_us = self.config.overhead.context_switch_us
        tiling_overhead_us = self.config.overhead.tiling_us

        # 计算 tiling count (从 shapes 或使用默认值)
        tiling_count = self._estimate_tiling_count(profile, chip_spec)

        # 分项开销
        tiling_total_us = tiling_overhead_us * tiling_count

        # 总开销
        total_overhead = (kernel_launch_us + sync_overhead_us +
                          context_switch_us + tiling_total_us)

        # 更新 breakdown
        latency_breakdown.timing.overhead_us = total_overhead

        # 计算最终时延
        final_latency = latency_before + total_overhead

        # 减去预取和并行节省
        final_latency -= latency_breakdown.savings.prefetch_us
        final_latency -= latency_breakdown.savings.backward_prefetch_us
        final_latency -= latency_breakdown.savings.parallel_us

        # 确保不为负
        final_latency = max(0, final_latency)

        latency_breakdown.timing.total_us = final_latency

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = final_latency
        result.latency_saved_us = latency_before - final_latency

        result.details = {
            "kernel_launch_us": kernel_launch_us,
            "sync_overhead_us": sync_overhead_us,
            "context_switch_us": context_switch_us,
            "tiling_overhead_us": tiling_overhead_us,
            "tiling_count": tiling_count,
            "tiling_total_us": tiling_total_us,
            "total_overhead_us": total_overhead,
            "prefetch_saved_us": latency_breakdown.savings.prefetch_us,
            "backward_prefetch_saved_us": latency_breakdown.savings.backward_prefetch_us,
            "parallel_saved_us": latency_breakdown.savings.parallel_us,
            "roofline_time_us": latency_before,
            "final_latency_us": final_latency,
            "overhead_breakdown": {
                "kernel_launch": kernel_launch_us,
                "sync": sync_overhead_us,
                "context_switch": context_switch_us,
                "tiling": tiling_total_us,
            }
        }

        # 检查开销占比
        overhead_ratio = total_overhead / final_latency if final_latency > 0 else 0
        if overhead_ratio > 0.1:
            result.warnings.append(
                f"开销占比较高 ({overhead_ratio*100:.1f}%)，考虑算子融合减少 kernel 数量"
            )

        # Tiling 开销过高警告
        if tiling_count > 1 and tiling_total_us > kernel_launch_us:
            result.suggestions.append(
                f"Tiling 开销 ({tiling_total_us:.2f}us, {tiling_count} tiles) 较大，"
                f"考虑增大 tile size 或使用更大的 L2 缓存"
            )

        return result

    def _estimate_tiling_count(self, profile, chip_spec) -> int:
        """估算 tiling count"""
        # 如果配置了固定值，直接使用
        if self.config.overhead.tiling_count > 1:
            return self.config.overhead.tiling_count

        shapes = profile.shapes or {}
        l2_size = getattr(getattr(chip_spec.memory, 'l2', None), 'capacity_bytes', 32 * 1024 * 1024)
        tile_size = 512
        op_type = profile.op_type.lower()

        # 分发到具体计算函数
        if op_type in ["matmul", "gemm", "linear"]:
            return self._tiling_for_matmul(shapes, l2_size, tile_size)
        if op_type in ["conv2d", "conv"]:
            return self._tiling_for_conv2d(shapes, l2_size, tile_size)
        return 1

    def _tiling_for_matmul(self, shapes: dict, l2_size: int, tile_size: int) -> int:
        """计算 MatMul 的 tiling count"""
        m = shapes.get("M") or shapes.get("m") or shapes.get("batch", 1) * shapes.get("seq_len", 1)
        n = shapes.get("N") or shapes.get("n") or shapes.get("out_features", 0)

        if not (m and n):
            return 1

        output_size = m * n * 2  # fp16
        if output_size <= l2_size:
            return 1

        tiles_m = max(1, (m + tile_size - 1) // tile_size)
        tiles_n = max(1, (n + tile_size - 1) // tile_size)
        return tiles_m * tiles_n

    def _tiling_for_conv2d(self, shapes: dict, l2_size: int, tile_size: int) -> int:
        """计算 Conv2D 的 tiling count"""
        h = shapes.get("H") or shapes.get("height", 0)
        w = shapes.get("W") or shapes.get("width", 0)
        c_out = shapes.get("C_out") or shapes.get("out_channels", 0)

        if not (h and w and c_out):
            return 1

        output_size = h * w * c_out * 2
        if output_size <= l2_size:
            return 1

        tiles_h = max(1, (h + tile_size - 1) // tile_size)
        tiles_w = max(1, (w + tile_size - 1) // tile_size)
        return tiles_h * tiles_w
