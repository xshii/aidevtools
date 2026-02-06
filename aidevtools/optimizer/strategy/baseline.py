"""
Baseline 策略

设计模式:
- Strategy 模式: 实现基准策略
"""

from typing import Dict, List, Optional

from .base import (
    TilingStrategy, TilingResult, TileConfig, FusionConfig,
    StrategyRegistry
)
from ..benchmark import Benchmark, OpSpec


@StrategyRegistry.register("baseline")
class BaselineStrategy(TilingStrategy):
    """
    基准策略

    特点:
    - 不进行融合
    - 简单的 tile 划分
    - 作为其他策略的对比基准
    """

    def __init__(self, default_buffer_size: int = 256 * 1024):
        """
        Args:
            default_buffer_size: 默认 buffer 大小 (bytes)
        """
        self.default_buffer_size = default_buffer_size

    @property
    def name(self) -> str:
        return "baseline"

    @property
    def description(self) -> str:
        return "基准策略：无融合，简单 tile 划分"

    def tile(self, benchmark: Benchmark,
             constraints: Optional[Dict] = None) -> TilingResult:
        """
        执行基准 tiling

        策略：
        1. 每个算子独立 tile
        2. 不进行算子融合
        3. 使用简单的均匀划分
        """
        constraints = constraints or {}
        buffer_limit = constraints.get("max_buffer_size", self.default_buffer_size)

        unfused_configs = {}
        total_cycles = 0
        total_memory = 0

        for op_name, op_spec in benchmark.ops.items():
            # 计算 tile 大小
            tile_sizes = self._calculate_tile_for_op(op_spec, buffer_limit)

            # 计算 buffer 大小
            buffer_sizes = self._calculate_buffers(op_spec, tile_sizes)

            # 确定循环顺序
            loop_order = self._determine_loop_order(op_spec)

            # 创建配置
            config = TileConfig(
                op_name=op_name,
                tile_sizes=tile_sizes,
                loop_order=loop_order,
                parallel_axes=self._get_parallel_axes(op_spec),
                buffer_sizes=buffer_sizes,
                double_buffer=True,
                cube_ratio=1.0 if op_spec.compute_unit in ["cube", "auto"] else 0.0,
                vector_ratio=1.0 if op_spec.compute_unit in ["vector", "auto"] else 0.0,
            )

            unfused_configs[op_name] = config

            # 估算
            cycles = self._estimate_op_cycles(op_spec, config, buffer_limit)
            total_cycles += cycles
            total_memory = max(total_memory, config.total_buffer_size())

        # 计算利用率
        compute_util = self._estimate_compute_utilization(benchmark, unfused_configs)
        memory_util = total_memory / buffer_limit if buffer_limit > 0 else 0

        return TilingResult(
            benchmark=benchmark,
            fusion_configs=[],  # baseline 不融合
            unfused_configs=unfused_configs,
            estimated_cycles=total_cycles,
            memory_footprint=total_memory,
            compute_utilization=compute_util,
            memory_utilization=min(1.0, memory_util),
            breakdown={
                "per_op_cycles": {
                    name: self._estimate_op_cycles(
                        benchmark.ops[name], config, buffer_limit
                    )
                    for name, config in unfused_configs.items()
                }
            }
        )

    def _calculate_tile_for_op(self, op_spec: OpSpec,
                               buffer_limit: int) -> Dict[str, int]:
        """计算算子的 tile 大小"""
        shapes = op_spec.shapes

        # 根据算子类型确定主要维度
        if op_spec.op_type.value == "matmul":
            # MatMul: (M, K) x (K, N) -> (M, N)
            M = shapes.get("M", shapes.get("m", 1))
            N = shapes.get("N", shapes.get("n", 1))
            K = shapes.get("K", shapes.get("k", 1))

            # 简单启发式：尽量让 tile 适合 buffer
            # 需要 A_tile(M_t, K_t) + B_tile(K_t, N_t) + C_tile(M_t, N_t)
            dtype_size = 2  # fp16

            # 假设 M_t = N_t = K_t = tile_size
            # 3 * tile_size^2 * dtype_size <= buffer_limit
            max_tile = int((buffer_limit / (3 * dtype_size)) ** 0.5)

            tile_size = min(max_tile, M, N, K, 256)  # 上限 256
            tile_size = max(tile_size, 16)  # 下限 16

            return {"M": min(tile_size, M),
                    "N": min(tile_size, N),
                    "K": min(tile_size, K)}

        elif op_spec.op_type.value in ["add", "mul", "relu", "gelu", "softmax"]:
            # Element-wise 或 reduction 算子
            total_elements = 1
            for v in shapes.values():
                total_elements *= v

            dtype_size = 2
            # 输入 + 输出
            elements_per_buffer = buffer_limit // (2 * dtype_size)

            if total_elements <= elements_per_buffer:
                return dict(shapes)

            # 缩放
            scale = (elements_per_buffer / total_elements) ** (1 / max(1, len(shapes)))
            return {k: max(1, int(v * scale)) for k, v in shapes.items()}

        else:
            # 通用处理
            return self._calculate_tile_size(shapes, buffer_limit)

    def _calculate_buffers(self, op_spec: OpSpec,
                          tile_sizes: Dict[str, int]) -> Dict[str, int]:
        """计算所需 buffer 大小"""
        dtype_size = 2  # fp16
        buffers = {}

        if op_spec.op_type.value == "matmul":
            M = tile_sizes.get("M", 1)
            N = tile_sizes.get("N", 1)
            K = tile_sizes.get("K", 1)

            buffers["A"] = M * K * dtype_size
            buffers["B"] = K * N * dtype_size
            buffers["C"] = M * N * dtype_size
        else:
            # 通用：输入 + 输出
            elements = 1
            for v in tile_sizes.values():
                elements *= v

            buffers["input"] = elements * dtype_size
            buffers["output"] = elements * dtype_size

        return buffers

    def _determine_loop_order(self, op_spec: OpSpec) -> List[str]:
        """确定循环顺序"""
        if op_spec.op_type.value == "matmul":
            return ["M", "N", "K"]  # 标准顺序
        return list(op_spec.shapes.keys())

    def _get_parallel_axes(self, op_spec: OpSpec) -> List[str]:
        """获取可并行的维度"""
        if op_spec.op_type.value == "matmul":
            return ["M", "N"]  # M 和 N 可并行

        # 对于 element-wise，所有维度都可并行
        return list(op_spec.shapes.keys())

    def _estimate_op_cycles(self, op_spec: OpSpec,
                           config: TileConfig,
                           buffer_limit: int) -> int:
        """估算算子执行周期"""
        # 计算 tile 数量
        num_tiles = 1
        for axis, tile_size in config.tile_sizes.items():
            full_size = op_spec.shapes.get(axis, 1)
            num_tiles *= (full_size + tile_size - 1) // tile_size

        # 单个 tile 的计算量
        tile_compute = 1
        for v in config.tile_sizes.values():
            tile_compute *= v

        if op_spec.op_type.value == "matmul":
            # MatMul: 2*M*N*K FLOPs
            M = config.tile_sizes.get("M", 1)
            N = config.tile_sizes.get("N", 1)
            K = config.tile_sizes.get("K", 1)
            tile_compute = 2 * M * N * K

        # 考虑效率
        efficiency = 0.5  # baseline 保守估计

        total_cycles = int(num_tiles * tile_compute / efficiency)
        return total_cycles

    def _estimate_compute_utilization(self, benchmark: Benchmark,
                                      configs: Dict[str, TileConfig]) -> float:
        """估算计算利用率"""
        # baseline 策略利用率较低
        return 0.4
