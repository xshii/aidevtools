"""
融合加速策略

设计模式:
- Strategy 模式: 实现融合优化策略
"""

from typing import Dict, List, Optional, Tuple, Set

from .base import (
    TilingStrategy, TilingResult, TileConfig, FusionConfig,
    StrategyRegistry
)
from ..benchmark import Benchmark, OpSpec, FusePair


@StrategyRegistry.register("fuse_speedup")
class FuseSpeedupStrategy(TilingStrategy):
    """
    融合加速策略

    特点:
    - 最大化融合收益
    - 基于 fuse_speedup 参数优化
    - 激进的流水线策略
    """

    def __init__(self,
                 default_buffer_size: int = 256 * 1024,
                 min_speedup_threshold: float = 1.1,
                 max_fusion_depth: int = 4):
        """
        Args:
            default_buffer_size: 默认 buffer 大小
            min_speedup_threshold: 最小融合加速阈值
            max_fusion_depth: 最大融合深度
        """
        self.default_buffer_size = default_buffer_size
        self.min_speedup_threshold = min_speedup_threshold
        self.max_fusion_depth = max_fusion_depth

    @property
    def name(self) -> str:
        return "fuse_speedup"

    @property
    def description(self) -> str:
        return "融合加速策略：最大化融合收益"

    def tile(self, benchmark: Benchmark,
             constraints: Optional[Dict] = None) -> TilingResult:
        """
        执行融合加速 tiling

        策略:
        1. 基于 fuse_speedup 排序融合候选
        2. 贪心选择高收益融合
        3. 激进的 tile 策略以支持深度流水
        """
        constraints = constraints or {}
        buffer_limit = constraints.get("max_buffer_size", self.default_buffer_size)

        # 1. 排序融合候选
        ranked_pairs = self._rank_fusion_candidates(benchmark)

        # 2. 贪心选择融合组
        fusion_groups = self._greedy_fusion_selection(
            benchmark, ranked_pairs, buffer_limit
        )

        # 3. 生成融合配置
        fusion_configs = []
        for group in fusion_groups:
            config = self._generate_aggressive_fusion_config(
                benchmark, group, buffer_limit
            )
            fusion_configs.append(config)

        # 4. 处理剩余算子
        fused_ops = set()
        for fc in fusion_configs:
            fused_ops.update(fc.fused_ops)

        unfused_configs = {}
        for op_name, op_spec in benchmark.ops.items():
            if op_name not in fused_ops:
                config = self._generate_unfused_config(op_spec, buffer_limit)
                unfused_configs[op_name] = config

        # 5. 计算指标
        total_cycles, memory_footprint = self._estimate_total_cost(
            benchmark, fusion_configs, unfused_configs, buffer_limit
        )

        compute_util = self._estimate_compute_utilization(fusion_configs)
        memory_util = memory_footprint / buffer_limit if buffer_limit > 0 else 0

        return TilingResult(
            benchmark=benchmark,
            fusion_configs=fusion_configs,
            unfused_configs=unfused_configs,
            estimated_cycles=total_cycles,
            memory_footprint=memory_footprint,
            compute_utilization=compute_util,
            memory_utilization=min(1.0, memory_util),
            breakdown=self._generate_breakdown(
                benchmark, fusion_configs, unfused_configs, ranked_pairs
            )
        )

    def _rank_fusion_candidates(self, benchmark: Benchmark) -> List[FusePair]:
        """
        排序融合候选

        优先级：fuse_speedup > ratio
        """
        pairs = list(benchmark.fuse_pairs)

        def score(pair: FusePair) -> float:
            # 融合加速比优先
            speedup_score = pair.fuse_speedup if pair.fuse_speedup else 1.0
            # 效率比次之
            ratio_score = pair.ratio
            # 综合得分
            return speedup_score * 2 + ratio_score

        pairs.sort(key=score, reverse=True)
        return pairs

    def _greedy_fusion_selection(self, benchmark: Benchmark,
                                ranked_pairs: List[FusePair],
                                buffer_limit: int) -> List[List[str]]:
        """
        贪心选择融合组

        策略：
        1. 从高收益对开始
        2. 尝试扩展融合组
        3. 检查 buffer 约束
        """
        used_ops: Set[str] = set()
        fusion_groups: List[List[str]] = []

        for pair in ranked_pairs:
            # 检查加速阈值
            speedup = pair.fuse_speedup if pair.fuse_speedup else pair.ratio
            if speedup < self.min_speedup_threshold:
                continue

            # 检查算子是否已被使用
            if pair.op_a in used_ops or pair.op_b in used_ops:
                # 尝试加入现有组
                extended = self._try_extend_group(
                    pair, fusion_groups, used_ops, benchmark, buffer_limit
                )
                if extended:
                    continue
                # 无法扩展，跳过
                continue

            # 创建新融合组
            group = [pair.op_a, pair.op_b]

            # 尝试扩展
            group = self._extend_fusion_group(
                group, benchmark, ranked_pairs, used_ops, buffer_limit
            )

            # 检查 buffer 约束
            if self._check_buffer_constraint(benchmark, group, buffer_limit):
                fusion_groups.append(group)
                used_ops.update(group)

        return fusion_groups

    def _try_extend_group(self, pair: FusePair,
                         groups: List[List[str]],
                         used_ops: Set[str],
                         benchmark: Benchmark,
                         buffer_limit: int) -> bool:
        """尝试将 pair 中的算子加入现有组"""
        for group in groups:
            # 检查是否有一个算子在组内
            if pair.op_a in group and pair.op_b not in used_ops:
                new_op = pair.op_b
            elif pair.op_b in group and pair.op_a not in used_ops:
                new_op = pair.op_a
            else:
                continue

            # 检查深度限制
            if len(group) >= self.max_fusion_depth:
                continue

            # 检查 buffer 约束
            test_group = group + [new_op]
            if self._check_buffer_constraint(benchmark, test_group, buffer_limit):
                group.append(new_op)
                used_ops.add(new_op)
                return True

        return False

    def _extend_fusion_group(self, group: List[str],
                            benchmark: Benchmark,
                            ranked_pairs: List[FusePair],
                            used_ops: Set[str],
                            buffer_limit: int) -> List[str]:
        """尝试扩展融合组"""
        while len(group) < self.max_fusion_depth:
            best_candidate = None
            best_score = 0

            for pair in ranked_pairs:
                # 检查是否可以扩展
                if pair.op_a in group and pair.op_b not in group and pair.op_b not in used_ops:
                    candidate = pair.op_b
                    score = pair.fuse_speedup if pair.fuse_speedup else pair.ratio
                elif pair.op_b in group and pair.op_a not in group and pair.op_a not in used_ops:
                    candidate = pair.op_a
                    score = pair.fuse_speedup if pair.fuse_speedup else pair.ratio
                else:
                    continue

                # 检查 buffer 约束
                test_group = group + [candidate]
                if not self._check_buffer_constraint(benchmark, test_group, buffer_limit):
                    continue

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate:
                group.append(best_candidate)
            else:
                break

        return group

    def _check_buffer_constraint(self, benchmark: Benchmark,
                                group: List[str],
                                buffer_limit: int) -> bool:
        """检查融合组是否满足 buffer 约束"""
        dtype_size = 2
        total_buffer = 0

        for op_name in group:
            op_spec = benchmark.ops[op_name]

            # 估算该算子所需的 buffer
            elements = 1
            for v in op_spec.shapes.values():
                elements *= v

            # 简化估算：输入 + 输出
            op_buffer = elements * dtype_size * 2
            total_buffer += op_buffer

        # 考虑 double buffer
        return total_buffer * 2 <= buffer_limit

    def _generate_aggressive_fusion_config(self, benchmark: Benchmark,
                                          group: List[str],
                                          buffer_limit: int) -> FusionConfig:
        """生成激进的融合配置"""
        tile_configs = {}

        # 计算紧凑的 tile 大小
        per_op_budget = buffer_limit // (len(group) * 4)  # 为流水预留空间

        for op_name in group:
            op_spec = benchmark.ops[op_name]

            tile_sizes = self._calculate_compact_tiles(op_spec, per_op_budget)
            buffer_sizes = self._calculate_buffers(op_spec, tile_sizes)

            config = TileConfig(
                op_name=op_name,
                tile_sizes=tile_sizes,
                loop_order=self._optimize_loop_order(op_spec, group),
                parallel_axes=self._get_parallel_axes(op_spec),
                buffer_sizes=buffer_sizes,
                double_buffer=True,
                cube_ratio=self._get_compute_ratio(op_spec)[0],
                vector_ratio=self._get_compute_ratio(op_spec)[1],
            )
            tile_configs[op_name] = config

        # 计算共享 buffer（激进共享）
        shared_buffers = self._calculate_aggressive_shared_buffers(
            benchmark, group, tile_configs
        )

        # 深度流水
        pipeline_depth = min(len(group) + 1, 4)

        return FusionConfig(
            fused_ops=group,
            tile_configs=tile_configs,
            shared_buffers=shared_buffers,
            pipeline_depth=pipeline_depth
        )

    def _calculate_compact_tiles(self, op_spec: OpSpec,
                                budget: int) -> Dict[str, int]:
        """计算紧凑的 tile 大小"""
        dtype_size = 2
        shapes = op_spec.shapes

        # 优先保证主要维度
        if op_spec.op_type.value == "matmul":
            # MatMul 优先 M, N
            M = shapes.get("M", shapes.get("m", 128))
            N = shapes.get("N", shapes.get("n", 128))
            K = shapes.get("K", shapes.get("k", 128))

            # 计算可用的 tile 大小
            # A(M,K) + B(K,N) + C(M,N) <= budget
            max_tile = int((budget / (3 * dtype_size)) ** 0.5)
            tile = min(max_tile, 128)
            tile = max(tile, 32)

            return {
                "M": min(tile, M),
                "N": min(tile, N),
                "K": min(tile // 2, K),  # K 可以小一些
            }

        # 通用处理
        total = 1
        for v in shapes.values():
            total *= v

        max_elements = budget // (2 * dtype_size)

        if total <= max_elements:
            return dict(shapes)

        scale = (max_elements / total) ** (1 / len(shapes))
        return {k: max(16, int(v * scale)) for k, v in shapes.items()}

    def _calculate_buffers(self, op_spec: OpSpec,
                          tile_sizes: Dict[str, int]) -> Dict[str, int]:
        """计算 buffer 大小"""
        dtype_size = 2

        if op_spec.op_type.value == "matmul":
            M = tile_sizes.get("M", 1)
            N = tile_sizes.get("N", 1)
            K = tile_sizes.get("K", 1)
            return {
                "A": M * K * dtype_size,
                "B": K * N * dtype_size,
                "C": M * N * dtype_size,
            }

        elements = 1
        for v in tile_sizes.values():
            elements *= v

        return {
            "input": elements * dtype_size,
            "output": elements * dtype_size,
        }

    def _optimize_loop_order(self, op_spec: OpSpec,
                            group: List[str]) -> List[str]:
        """优化循环顺序以支持流水"""
        if op_spec.op_type.value == "matmul":
            # K 在最内层以支持累加
            return ["M", "N", "K"]

        # 把最大维度放外层
        sorted_axes = sorted(
            op_spec.shapes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [axis for axis, _ in sorted_axes]

    def _get_parallel_axes(self, op_spec: OpSpec) -> List[str]:
        """获取可并行维度"""
        if op_spec.op_type.value == "matmul":
            return ["M", "N"]
        return list(op_spec.shapes.keys())[:2]

    def _get_compute_ratio(self, op_spec: OpSpec) -> Tuple[float, float]:
        """获取计算单元比例 (cube, vector)"""
        if op_spec.compute_unit == "cube":
            return (1.0, 0.0)
        elif op_spec.compute_unit == "vector":
            return (0.0, 1.0)
        elif op_spec.compute_unit == "mixed":
            return (0.7, 0.3)
        return (0.8, 0.2)  # auto

    def _calculate_aggressive_shared_buffers(self, benchmark: Benchmark,
                                            group: List[str],
                                            tile_configs: Dict[str, TileConfig]) -> Dict[str, int]:
        """计算激进的共享 buffer"""
        shared = {}

        # 相邻算子共享输出/输入
        for i in range(len(group) - 1):
            op_a, op_b = group[i], group[i + 1]

            config_a = tile_configs[op_a]
            config_b = tile_configs[op_b]

            # 共享大小取较小者
            size_a = sum(config_a.buffer_sizes.values())
            size_b = sum(config_b.buffer_sizes.values())
            shared[f"shared_{op_a}_{op_b}"] = min(size_a, size_b) // 2

        return shared

    def _generate_unfused_config(self, op_spec: OpSpec,
                                buffer_limit: int) -> TileConfig:
        """生成未融合算子的配置"""
        tile_sizes = self._calculate_compact_tiles(op_spec, buffer_limit // 2)
        buffer_sizes = self._calculate_buffers(op_spec, tile_sizes)

        cube, vector = self._get_compute_ratio(op_spec)

        return TileConfig(
            op_name=op_spec.name,
            tile_sizes=tile_sizes,
            loop_order=list(op_spec.shapes.keys()),
            parallel_axes=self._get_parallel_axes(op_spec),
            buffer_sizes=buffer_sizes,
            double_buffer=True,
            cube_ratio=cube,
            vector_ratio=vector,
        )

    def _estimate_total_cost(self, benchmark: Benchmark,
                            fusion_configs: List[FusionConfig],
                            unfused_configs: Dict[str, TileConfig],
                            buffer_limit: int) -> Tuple[int, int]:
        """估算总代价"""
        total_cycles = 0
        max_memory = 0

        for fc in fusion_configs:
            group_cycles = 0

            for op_name in fc.fused_ops:
                op_spec = benchmark.ops[op_name]
                config = fc.tile_configs[op_name]
                cycles = self._estimate_cycles(config, op_spec)
                group_cycles += cycles

            # 融合收益
            speedup = 1.0
            for pair in benchmark.fuse_pairs:
                if pair.op_a in fc.fused_ops and pair.op_b in fc.fused_ops:
                    if pair.fuse_speedup:
                        speedup = max(speedup, pair.fuse_speedup)

            total_cycles += int(group_cycles / speedup)
            max_memory = max(max_memory, fc.total_buffer_size())

        for op_name, config in unfused_configs.items():
            op_spec = benchmark.ops[op_name]
            cycles = self._estimate_cycles(config, op_spec)
            total_cycles += cycles
            max_memory = max(max_memory, config.total_buffer_size())

        return total_cycles, max_memory

    def _estimate_compute_utilization(self,
                                      fusion_configs: List[FusionConfig]) -> float:
        """估算计算利用率"""
        if not fusion_configs:
            return 0.5

        # 融合越多，利用率越高
        avg_depth = sum(len(fc.fused_ops) for fc in fusion_configs) / len(fusion_configs)
        return min(0.85, 0.5 + avg_depth * 0.1)

    def _generate_breakdown(self, benchmark: Benchmark,
                           fusion_configs: List[FusionConfig],
                           unfused_configs: Dict[str, TileConfig],
                           ranked_pairs: List[FusePair]) -> Dict:
        """生成详细分析"""
        return {
            "fusion_groups": [
                {
                    "ops": fc.fused_ops,
                    "pipeline_depth": fc.pipeline_depth,
                    "total_buffer": fc.total_buffer_size(),
                    "shared_buffer": sum(fc.shared_buffers.values()),
                }
                for fc in fusion_configs
            ],
            "unfused_ops": {
                name: config.to_dict()
                for name, config in unfused_configs.items()
            },
            "fusion_candidates": [
                {
                    "op_a": p.op_a,
                    "op_b": p.op_b,
                    "ratio": p.ratio,
                    "speedup": p.fuse_speedup,
                }
                for p in ranked_pairs[:10]  # top 10
            ],
        }
