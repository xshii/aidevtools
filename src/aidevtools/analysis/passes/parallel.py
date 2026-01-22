"""Parallel Pass - Cube/Vector 并行优化

当 Cube 和 Vector 可以并行执行时:
- 连续的 Cube 算子和 Vector 算子可以重叠
- 总时延 = max(cube_time, vector_time) 而非 cube_time + vector_time
"""

from typing import Optional
from .base import BasePass, PassConfig, PassResult


class CubeVectorParallelPass(BasePass):
    """Cube/Vector 并行优化 Pass"""

    name = "cube_vector_parallel"
    description = "Cube 和 Vector 单元并行执行优化"
    order = 5

    def __init__(self, config: PassConfig = None,
                 adjacent_op_unit: str = "",
                 adjacent_op_time_us: float = 0.0,
                 adjacent_op_name: str = ""):
        """
        Args:
            adjacent_op_unit: 相邻算子的计算单元 ("cube" | "vector")
            adjacent_op_time_us: 相邻算子的执行时间
            adjacent_op_name: 相邻算子名称
        """
        super().__init__(config)
        self.adjacent_op_unit = adjacent_op_unit
        self.adjacent_op_time_us = adjacent_op_time_us
        self.adjacent_op_name = adjacent_op_name

    def is_enabled(self) -> bool:
        return self.config.enabled and self.config.cube_vector_parallel_enabled

    def run(self, latency_breakdown, chip_spec) -> PassResult:
        """执行 Cube/Vector 并行优化"""
        result = PassResult(pass_name=self.name, enabled=self.is_enabled())

        if not self.is_enabled():
            return result

        profile = latency_breakdown.profile
        latency_before = latency_breakdown.roofline_time_us

        # 检查芯片是否支持 Cube/Vector 并行
        if not chip_spec.pipeline.cube_vector_parallel:
            result.details = {"reason": "芯片不支持 Cube/Vector 并行"}
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 检查是否有相邻的不同类型算子
        current_unit = profile.compute_unit
        if not self.adjacent_op_unit or current_unit == self.adjacent_op_unit:
            result.details = {
                "reason": "无相邻不同类型算子",
                "current_unit": current_unit,
                "adjacent_unit": self.adjacent_op_unit
            }
            result.latency_before_us = latency_before
            result.latency_after_us = latency_before
            return result

        # 计算并行节省
        current_time = latency_breakdown.roofline_time_us
        adjacent_time = self.adjacent_op_time_us

        # 串行时延 vs 并行时延
        serial_time = current_time + adjacent_time
        parallel_time = max(current_time, adjacent_time)
        saved_time = serial_time - parallel_time

        # 更新 breakdown
        latency_breakdown.parallel_saved_us = saved_time

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = latency_before
        result.latency_saved_us = saved_time

        result.details = {
            "current_unit": current_unit,
            "current_time_us": current_time,
            "adjacent_unit": self.adjacent_op_unit,
            "adjacent_op_name": self.adjacent_op_name,
            "adjacent_time_us": adjacent_time,
            "serial_time_us": serial_time,
            "parallel_time_us": parallel_time,
            "saved_time_us": saved_time,
        }

        if saved_time > 0:
            result.suggestions.append(
                f"{current_unit.capitalize()} 算子与相邻 {self.adjacent_op_unit.capitalize()} 算子 "
                f"'{self.adjacent_op_name}' 可并行执行，节省 {saved_time:.2f}us "
                f"(串行 {serial_time:.2f}us → 并行 {parallel_time:.2f}us)"
            )

        return result
