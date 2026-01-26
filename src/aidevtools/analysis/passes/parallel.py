"""Parallel Pass - Cube/Vector 并行优化

当 Cube 和 Vector 可以并行执行时:
- 连续的 Cube 算子和 Vector 算子可以重叠
- 总时延 = max(cube_time, vector_time) 而非 cube_time + vector_time

Example:
    算子序列: MatMul(Cube, 10us) -> LayerNorm(Vector, 3us) -> ...

    串行执行: 10 + 3 = 13us
    并行执行: max(10, 3) = 10us (LayerNorm 与 MatMul 重叠)
    节省: 13 - 10 = 3us

    前提条件:
    1. 芯片支持 cube_vector_parallel (如 NPU 910)
    2. 两个相邻算子使用不同计算单元 (Cube vs Vector)
    3. 无数据依赖阻塞
"""

from ..constants import GBPS_TO_BPS, S_TO_US, TFLOPS_TO_FLOPS, UNIT_CUBE
from .base import BasePass, PassContext, PassResult


class CubeVectorParallelPass(BasePass):
    """Cube/Vector 并行优化 Pass"""

    name = "cube_vector_parallel"
    description = "Cube 和 Vector 单元并行执行优化"
    order = 500
    config_key = "cube_vector_parallel"

    def _do_run(
        self, latency_breakdown, chip_spec, result: PassResult, context: PassContext = None
    ) -> PassResult:
        """执行 Cube/Vector 并行优化"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.timing.roofline_us

        # 检查芯片是否支持 Cube/Vector 并行
        if not chip_spec.pipeline.cube_vector_parallel:
            return self._skip(result, latency_before, "芯片不支持 Cube/Vector 并行")

        # 从 context 获取下一算子信息
        next_profile = context.next_profile if context else None
        if not next_profile:
            return self._skip(result, latency_before, "无下一算子")

        current_unit = profile.compute_unit
        adjacent_unit = next_profile.compute_unit
        adjacent_op_name = next_profile.name

        # 检查是否有相邻的不同类型算子
        if current_unit == adjacent_unit:
            result.details = {
                "reason": "相邻算子使用相同计算单元",
                "current_unit": current_unit,
                "adjacent_unit": adjacent_unit,
            }
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 估算相邻算子时间 (基于 Roofline 模型)
        adjacent_time = self._estimate_op_time(next_profile, chip_spec)

        # 计算并行节省
        current_time = latency_breakdown.timing.roofline_us
        serial_time = current_time + adjacent_time
        parallel_time = max(current_time, adjacent_time)
        saved_time = serial_time - parallel_time

        # 更新 breakdown
        latency_breakdown.savings.parallel_us = saved_time

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = latency_before
        result.latency_saved_us = saved_time

        result.details = {
            "current_unit": current_unit,
            "current_time_us": current_time,
            "adjacent_unit": adjacent_unit,
            "adjacent_op_name": adjacent_op_name,
            "adjacent_time_us": adjacent_time,
            "serial_time_us": serial_time,
            "parallel_time_us": parallel_time,
            "saved_time_us": saved_time,
        }

        if saved_time > 0:
            result.suggestions.append(
                f"{current_unit.capitalize()} 算子与相邻 {adjacent_unit.capitalize()} 算子 "
                f"'{adjacent_op_name}' 可并行执行，节省 {saved_time:.2f}us "
                f"(串行 {serial_time:.2f}us → 并行 {parallel_time:.2f}us)"
            )

        return result

    def _estimate_op_time(self, profile, chip_spec) -> float:
        """估算算子执行时间 (基于 Roofline 模型)"""
        if profile.compute_unit == UNIT_CUBE:
            tflops = chip_spec.cube.fp16_tflops
        else:
            tflops = chip_spec.vector.fp16_gflops / 1000.0

        compute_time = profile.flops / (tflops * TFLOPS_TO_FLOPS) * S_TO_US if tflops > 0 else 0

        bandwidth = chip_spec.memory.hbm.bandwidth_gbps
        memory_time = (
            profile.total_bytes / (bandwidth * GBPS_TO_BPS) * S_TO_US if bandwidth > 0 else 0
        )

        return max(compute_time, memory_time)
