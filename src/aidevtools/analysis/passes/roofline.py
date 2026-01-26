"""Roofline Pass - 基础 Roofline 模型时延计算

Roofline 模型: latency = max(compute_time, memory_time)
- compute_time = FLOPs / (peak_tflops * 1e12) * 1e6 (us)
- memory_time = bytes / (bandwidth_gbps * 1e9) * 1e6 (us)

Example:
    MatMul [4, 512, 768] @ [768, 768] on NPU 910:
    - FLOPs = 2 * 4 * 512 * 768 * 768 = 2.42G
    - Bytes = input(3MB) + weight(1.1MB) + output(3MB) = 7.1MB
    - Arithmetic Intensity = 2.42G / 7.1MB = 341 FLOPs/Byte
    - Ridge Point = 256 TFLOPS / 1200 GB/s = 213 FLOPs/Byte
    - AI > Ridge => Compute Bound
    - compute_time = 2.42G / 256T = 9.45us
    - memory_time = 7.1MB / 1200GB/s = 5.92us
    - roofline_time = max(9.45, 5.92) = 9.45us
"""

from dataclasses import dataclass

from ..constants import (
    BOTTLENECK_COMPUTE,
    BOTTLENECK_MEMORY,
    GBPS_TO_BPS,
    S_TO_US,
    TFLOPS_TO_FLOPS,
    UNIT_CUBE,
)
from .base import BasePass, PassContext, PassResult


@dataclass
class RooflineMetrics:
    """Roofline 指标"""

    flops: int = 0
    total_bytes: int = 0
    arithmetic_intensity: float = 0.0  # FLOPs / Bytes
    ridge_point: float = 0.0  # 拐点
    is_compute_bound: bool = False
    is_memory_bound: bool = False


class RooflinePass(BasePass):
    """Roofline 时延计算 Pass"""

    name = "roofline"
    description = "基础 Roofline 模型时延计算"
    order = 100
    config_key = "roofline"

    def _do_run(
        self, latency_breakdown, chip_spec, result: PassResult, context: PassContext = None
    ) -> PassResult:
        """执行 Roofline 计算"""
        profile = latency_breakdown.profile

        # 获取计算单元规格
        if profile.compute_unit == UNIT_CUBE:
            peak_tflops = self._get_cube_tflops(chip_spec, profile.dtype)
        else:
            peak_gflops = self._get_vector_gflops(chip_spec, profile.dtype)
            peak_tflops = peak_gflops / 1000.0

        # 计算时间 (us)
        compute_time_us = 0.0
        if peak_tflops > 0 and profile.flops > 0:
            compute_time_us = profile.flops / (peak_tflops * TFLOPS_TO_FLOPS) * S_TO_US

        # 访存时间 (us)
        total_bytes = profile.total_bytes
        hbm_bandwidth_gbps = chip_spec.memory.hbm.bandwidth_gbps

        memory_time_us = 0.0
        if hbm_bandwidth_gbps > 0 and total_bytes > 0:
            memory_time_us = total_bytes / (hbm_bandwidth_gbps * GBPS_TO_BPS) * S_TO_US

        # Roofline: max(compute, memory)
        roofline_time_us = max(compute_time_us, memory_time_us)

        # 计算算术强度和拐点
        arithmetic_intensity = profile.flops / total_bytes if total_bytes > 0 else 0
        ridge_point = (
            (peak_tflops * TFLOPS_TO_FLOPS) / (hbm_bandwidth_gbps * GBPS_TO_BPS)
            if hbm_bandwidth_gbps > 0
            else 0
        )

        # 判断瓶颈
        is_compute_bound = arithmetic_intensity >= ridge_point
        is_memory_bound = not is_compute_bound

        # 更新 latency_breakdown
        latency_breakdown.timing.compute_us = compute_time_us
        latency_breakdown.timing.memory_us = memory_time_us
        latency_breakdown.timing.roofline_us = roofline_time_us
        latency_breakdown.bottleneck = BOTTLENECK_COMPUTE if is_compute_bound else BOTTLENECK_MEMORY

        # 计算最小带宽需求
        if compute_time_us > 0:
            min_bandwidth_gbps = total_bytes / (compute_time_us / S_TO_US) / GBPS_TO_BPS
            latency_breakdown.bandwidth.min_gbps = min_bandwidth_gbps

        # 填充结果
        result.latency_before_us = 0
        result.latency_after_us = roofline_time_us
        result.details = {
            "compute_time_us": compute_time_us,
            "memory_time_us": memory_time_us,
            "roofline_time_us": roofline_time_us,
            "flops": profile.flops,
            "total_bytes": total_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "ridge_point": ridge_point,
            "is_compute_bound": is_compute_bound,
            "is_memory_bound": is_memory_bound,
            "peak_tflops": peak_tflops,
            "hbm_bandwidth_gbps": hbm_bandwidth_gbps,
            "min_bandwidth_gbps": latency_breakdown.bandwidth.min_gbps,
        }

        # 添加建议
        if is_memory_bound:
            result.suggestions.append(
                f"算子为访存瓶颈 (AI={arithmetic_intensity:.2f} < ridge={ridge_point:.2f})，"
                f"考虑算子融合或提升复用"
            )
        else:
            result.suggestions.append(
                f"算子为算力瓶颈 (AI={arithmetic_intensity:.2f} >= ridge={ridge_point:.2f})，"
                f"已充分利用计算资源"
            )

        return result

    # dtype 到属性名的映射
    _DTYPE_MAP = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "int8": "int8",
        "int8_t": "int8",
    }

    def _get_cube_tflops(self, chip_spec, dtype: str) -> float:
        """获取 Cube 单元峰值算力"""
        key = self._DTYPE_MAP.get(dtype.lower(), "fp16")
        attr = f"{key}_tflops" if key != "int8" else "int8_tops"
        return getattr(chip_spec.cube, attr, chip_spec.cube.fp16_tflops)

    def _get_vector_gflops(self, chip_spec, dtype: str) -> float:
        """获取 Vector 单元峰值算力"""
        key = self._DTYPE_MAP.get(dtype.lower(), "fp16")
        # Vector 只有 fp16/fp32，bf16 fallback 到 fp16
        if key == "bf16":
            key = "fp16"
        attr = f"{key}_gflops"
        return getattr(chip_spec.vector, attr, chip_spec.vector.fp16_gflops)
