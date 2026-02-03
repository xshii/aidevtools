"""Memory Efficiency Pass - 访存效率修正

根据访存模式修正访存时间:
- sequential: 85% 效率 (连续访问)
- strided: 50% 效率 (跨步访问, 如 transpose)
- random: 25% 效率 (随机访问)

Example:
    Transpose [4, 12, 512, 64] axes=(0,2,1,3) on NPU 910:
    - memory_pattern = "strided"
    - pattern_efficiency = 0.50
    - hbm_efficiency["strided"] = 0.55
    - combined_efficiency = 0.50 * 0.55 = 0.275
    - original_memory_time = 5.0us
    - adjusted_memory_time = 5.0 / 0.275 = 18.18us

    说明: 跨步访问导致实际带宽利用率下降，时延增加
"""

from ..constants import (
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY,
    PATTERN_RANDOM,
    PATTERN_SEQUENTIAL,
    PATTERN_STRIDED,
)
from .base import BasePass, PassContext, PassResult

# 访存模式效率
MEMORY_PATTERN_EFFICIENCY = {
    PATTERN_SEQUENTIAL: 0.85,
    PATTERN_STRIDED: 0.50,
    PATTERN_RANDOM: 0.25,
}


class MemoryEfficiencyPass(BasePass):
    """访存效率修正 Pass"""

    name = "memory_efficiency"
    description = "根据访存模式修正访存时延"
    order = 200
    config_key = "memory_efficiency"

    def _do_run(self, latency_breakdown, chip_spec, result: PassResult,
                context: PassContext = None) -> PassResult:
        """执行访存效率修正"""
        profile = latency_breakdown.profile
        latency_before = latency_breakdown.timing.roofline_us

        # 获取访存效率
        pattern = profile.memory_pattern
        efficiency = MEMORY_PATTERN_EFFICIENCY.get(pattern, 0.85)

        # 如果配置了使用有效带宽
        if self.config.use_effective_bandwidth:
            # 考虑 HBM 效率 (efficiency 是一个 dict 或 float)
            hbm_efficiency = chip_spec.memory.hbm.efficiency
            if isinstance(hbm_efficiency, dict):
                # 使用对应访存模式的效率
                hbm_eff_value = hbm_efficiency.get(pattern, 0.85)
            else:
                hbm_eff_value = hbm_efficiency if hbm_efficiency else 0.85
            efficiency = efficiency * hbm_eff_value

        # 修正访存时间
        original_memory_time = latency_breakdown.timing.memory_us
        adjusted_memory_time = original_memory_time / efficiency

        # 重新计算 roofline 时延
        new_roofline_time = max(latency_breakdown.timing.compute_us, adjusted_memory_time)

        # 更新瓶颈判断
        if adjusted_memory_time > latency_breakdown.timing.compute_us:
            latency_breakdown.bottleneck = BOTTLENECK_MEMORY
        else:
            latency_breakdown.bottleneck = BOTTLENECK_COMPUTE

        # 计算变化
        latency_delta = new_roofline_time - latency_before

        # 更新 breakdown
        latency_breakdown.timing.memory_us = adjusted_memory_time
        latency_breakdown.timing.roofline_us = new_roofline_time

        # 填充结果
        result.latency_before_us = latency_before
        result.latency_after_us = new_roofline_time
        result.latency_saved_us = -latency_delta  # 负数表示增加

        result.details = {
            "memory_pattern": pattern,
            "pattern_efficiency": MEMORY_PATTERN_EFFICIENCY.get(pattern, 0.85),
            "hbm_efficiency": chip_spec.memory.hbm.efficiency,
            "combined_efficiency": efficiency,
            "original_memory_time_us": original_memory_time,
            "adjusted_memory_time_us": adjusted_memory_time,
        }

        if latency_delta > 0:
            result.warnings.append(
                f"访存模式 '{pattern}' 效率较低 ({efficiency*100:.0f}%)，"
                f"访存时间增加 {latency_delta:.2f}us"
            )

        if pattern == PATTERN_STRIDED:
            result.suggestions.append(
                "跨步访问模式效率较低，考虑数据重排或使用连续内存布局"
            )
        elif pattern == PATTERN_RANDOM:
            result.suggestions.append(
                "随机访问模式效率最低，强烈建议优化数据访问模式"
            )

        return result
