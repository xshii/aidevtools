"""
效率感知策略

设计模式:
- Strategy 模式: 实现效率优化策略
"""

from typing import Dict, List, Optional, Tuple

from .base import (
    TilingStrategy, TilingResult, TileConfig, FusionConfig,
    StrategyRegistry
)
from ..benchmark import Benchmark, OpSpec, FusePair


@StrategyRegistry.register("efficiency_aware")
class EfficiencyAwareStrategy(TilingStrategy):
    """
    效率感知策略

    特点:
    - 根据用户指定的效率比进行优化
    - 考虑计算单元的实际效率
    - 平衡计算和访存
    """

    def __init__(self,
                 default_buffer_size: int = 256 * 1024,
                 target_compute_util: float = 0.7,
                 target_memory_util: float = 0.8):
        """
        Args:
            default_buffer_size: 默认 buffer 大小
            target_compute_util: 目标计算利用率
            target_memory_util: 目标内存利用率
        """
        self.default_buffer_size = default_buffer_size
        self.target_compute_util = target_compute_util
        self.target_memory_util = target_memory_util

    @property
    def name(self) -> str:
        return "efficiency_aware"

    @property
    def description(self) -> str:
        return "效率感知策略：基于效率比优化 tile 配置"

    def tile(self, benchmark: Benchmark,
             constraints: Optional[Dict] = None) -> TilingResult:
        """
        执行效率感知 tiling

        策略:
        1. 分析算子间的效率比
        2. 根据效率比确定融合边界
        3. 优化每个融合组的 tile 配置
        """
        constraints = constraints or {}
        buffer_limit = constraints.get("max_buffer_size", self.default_buffer_size)

        # 1. 分析效率比，确定融合组
        fusion_groups = self._analyze_fusion_groups(benchmark)

        # 2. 为每个融合组生成配置
        fusion_configs = []
        for group in fusion_groups:
            config = self._generate_fusion_config(benchmark, group, buffer_limit)
            fusion_configs.append(config)

        # 3. 处理未融合的算子
        fused_ops = set()
        for config in fusion_configs:
            fused_ops.update(config.fused_ops)

        unfused_configs = {}
        for op_name, op_spec in benchmark.ops.items():
            if op_name not in fused_ops:
                config = self._generate_single_config(op_spec, buffer_limit)
                unfused_configs[op_name] = config

        # 4. 计算整体指标
        total_cycles, memory_footprint = self._estimate_total_cost(
            benchmark, fusion_configs, unfused_configs, buffer_limit
        )

        compute_util = self._estimate_compute_utilization(
            benchmark, fusion_configs, unfused_configs
        )
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
                benchmark, fusion_configs, unfused_configs, buffer_limit
            )
        )

    def _analyze_fusion_groups(self, benchmark: Benchmark) -> List[List[str]]:
        """
        分析算子关系，确定融合组

        基于效率比判断是否值得融合
        """
        if not benchmark.fuse_pairs:
            return []

        # 构建算子图
        adj: Dict[str, List[str]] = {op: [] for op in benchmark.ops}
        beneficial_pairs: Dict[Tuple[str, str], FusePair] = {}

        for pair in benchmark.fuse_pairs:
            # 判断融合是否有收益
            if self._is_fusion_beneficial(pair, benchmark):
                adj[pair.op_a].append(pair.op_b)
                adj[pair.op_b].append(pair.op_a)
                beneficial_pairs[(pair.op_a, pair.op_b)] = pair
                beneficial_pairs[(pair.op_b, pair.op_a)] = pair

        # 贪心合并融合组
        visited = set()
        groups = []

        for op in benchmark.ops:
            if op in visited:
                continue

            # BFS 扩展融合组
            group = [op]
            visited.add(op)
            queue = [op]

            while queue:
                current = queue.pop(0)
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        # 检查与组内所有算子的兼容性
                        compatible = all(
                            self._can_fuse_with_group(neighbor, g, benchmark)
                            for g in group
                        )
                        if compatible:
                            group.append(neighbor)
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _is_fusion_beneficial(self, pair: FusePair,
                             benchmark: Benchmark) -> bool:
        """判断融合是否有收益"""
        # 基于效率比判断
        # ratio > 1 表示融合后效率更高
        if pair.ratio > 1.0:
            return True

        # 如果指定了融合加速比
        if pair.fuse_speedup and pair.fuse_speedup > 1.0:
            return True

        # 默认：ratio > 0.8 时考虑融合（减少访存开销）
        return pair.ratio > 0.8

    def _can_fuse_with_group(self, op: str, group_op: str,
                            benchmark: Benchmark) -> bool:
        """检查算子是否可以加入融合组"""
        # 检查是否有定义效率比
        for pair in benchmark.fuse_pairs:
            if (pair.op_a == op and pair.op_b == group_op) or \
               (pair.op_b == op and pair.op_a == group_op):
                return self._is_fusion_beneficial(pair, benchmark)
        return False

    def _generate_fusion_config(self, benchmark: Benchmark,
                               group: List[str],
                               buffer_limit: int) -> FusionConfig:
        """生成融合组的配置"""
        tile_configs = {}
        shared_buffers = {}

        # 分析组内算子的形状
        group_shapes = self._analyze_group_shapes(benchmark, group)

        # 确定统一的 tile 大小
        unified_tiles = self._calculate_unified_tiles(
            benchmark, group, group_shapes, buffer_limit
        )

        for op_name in group:
            op_spec = benchmark.ops[op_name]

            # 适配统一 tile 到具体算子
            tile_sizes = self._adapt_tiles_to_op(unified_tiles, op_spec)

            config = TileConfig(
                op_name=op_name,
                tile_sizes=tile_sizes,
                loop_order=self._determine_fused_loop_order(group, op_name),
                parallel_axes=self._get_parallel_axes(op_spec),
                buffer_sizes=self._calculate_buffers(op_spec, tile_sizes),
                double_buffer=True,
                cube_ratio=self._get_cube_ratio(op_spec),
                vector_ratio=self._get_vector_ratio(op_spec),
            )
            tile_configs[op_name] = config

        # 计算共享 buffer
        shared_buffers = self._calculate_shared_buffers(benchmark, group, tile_configs)

        # 确定流水深度
        pipeline_depth = self._determine_pipeline_depth(group, buffer_limit)

        return FusionConfig(
            fused_ops=group,
            tile_configs=tile_configs,
            shared_buffers=shared_buffers,
            pipeline_depth=pipeline_depth
        )

    def _analyze_group_shapes(self, benchmark: Benchmark,
                             group: List[str]) -> Dict[str, int]:
        """分析融合组的统一形状"""
        # 收集所有维度
        all_shapes = {}
        for op_name in group:
            op_spec = benchmark.ops[op_name]
            for axis, size in op_spec.shapes.items():
                if axis in all_shapes:
                    all_shapes[axis] = max(all_shapes[axis], size)
                else:
                    all_shapes[axis] = size
        return all_shapes

    def _calculate_unified_tiles(self, benchmark: Benchmark,
                                group: List[str],
                                shapes: Dict[str, int],
                                buffer_limit: int) -> Dict[str, int]:
        """计算统一的 tile 大小"""
        dtype_size = 2

        # 估算需要的 buffer 数量
        num_buffers = len(group) + 1  # 每个算子的输出 + 共享输入

        # 计算每个 buffer 的预算
        per_buffer_budget = buffer_limit // (num_buffers * 2)  # double buffer

        # 根据预算计算 tile 大小
        total_elements = 1
        for v in shapes.values():
            total_elements *= v

        max_elements = per_buffer_budget // dtype_size

        if total_elements <= max_elements:
            return dict(shapes)

        # 缩放
        scale = (max_elements / total_elements) ** (1 / max(1, len(shapes)))
        tiles = {}
        for axis, size in shapes.items():
            tile = max(16, int(size * scale))
            tiles[axis] = min(tile, size)

        return tiles

    def _adapt_tiles_to_op(self, unified_tiles: Dict[str, int],
                          op_spec: OpSpec) -> Dict[str, int]:
        """将统一 tile 适配到具体算子"""
        adapted = {}
        for axis, size in op_spec.shapes.items():
            if axis in unified_tiles:
                adapted[axis] = min(unified_tiles[axis], size)
            else:
                adapted[axis] = size
        return adapted

    def _determine_fused_loop_order(self, group: List[str],
                                   op_name: str) -> List[str]:
        """确定融合后的循环顺序"""
        # 简化：使用标准顺序
        return ["M", "N", "K", "batch", "seq", "hidden"]

    def _get_parallel_axes(self, op_spec: OpSpec) -> List[str]:
        """获取可并行维度"""
        if op_spec.op_type.value == "matmul":
            return ["M", "N"]
        return list(op_spec.shapes.keys())[:2]

    def _calculate_buffers(self, op_spec: OpSpec,
                          tile_sizes: Dict[str, int]) -> Dict[str, int]:
        """计算 buffer 大小"""
        dtype_size = 2
        elements = 1
        for v in tile_sizes.values():
            elements *= v

        if op_spec.op_type.value == "matmul":
            M = tile_sizes.get("M", 1)
            N = tile_sizes.get("N", 1)
            K = tile_sizes.get("K", 1)
            return {
                "A": M * K * dtype_size,
                "B": K * N * dtype_size,
                "C": M * N * dtype_size,
            }

        return {
            "input": elements * dtype_size,
            "output": elements * dtype_size,
        }

    def _get_cube_ratio(self, op_spec: OpSpec) -> float:
        """获取 Cube 单元使用比例"""
        if op_spec.compute_unit == "cube":
            return 1.0
        elif op_spec.compute_unit == "vector":
            return 0.0
        elif op_spec.compute_unit == "mixed":
            return 0.6
        return 0.8  # auto

    def _get_vector_ratio(self, op_spec: OpSpec) -> float:
        """获取 Vector 单元使用比例"""
        if op_spec.compute_unit == "vector":
            return 1.0
        elif op_spec.compute_unit == "cube":
            return 0.0
        elif op_spec.compute_unit == "mixed":
            return 0.4
        return 0.2  # auto

    def _calculate_shared_buffers(self, benchmark: Benchmark,
                                 group: List[str],
                                 tile_configs: Dict[str, TileConfig]) -> Dict[str, int]:
        """计算共享 buffer"""
        shared = {}

        # 分析数据依赖，找出可共享的中间结果
        for i, op_a in enumerate(group[:-1]):
            op_b = group[i + 1]

            # 假设相邻算子共享中间结果
            config_a = tile_configs[op_a]
            shared_size = sum(config_a.buffer_sizes.values()) // 2
            shared[f"{op_a}_{op_b}_shared"] = shared_size

        return shared

    def _determine_pipeline_depth(self, group: List[str],
                                 buffer_limit: int) -> int:
        """确定流水深度"""
        # 简单策略：组越大，流水越深
        if len(group) <= 2:
            return 2
        elif len(group) <= 4:
            return 3
        return 4

    def _generate_single_config(self, op_spec: OpSpec,
                               buffer_limit: int) -> TileConfig:
        """生成单个算子的配置"""
        tile_sizes = self._calculate_tile_size(op_spec.shapes, buffer_limit)

        return TileConfig(
            op_name=op_spec.name,
            tile_sizes=tile_sizes,
            loop_order=list(op_spec.shapes.keys()),
            parallel_axes=self._get_parallel_axes(op_spec),
            buffer_sizes=self._calculate_buffers(op_spec, tile_sizes),
            double_buffer=True,
            cube_ratio=self._get_cube_ratio(op_spec),
            vector_ratio=self._get_vector_ratio(op_spec),
        )

    def _estimate_total_cost(self, benchmark: Benchmark,
                            fusion_configs: List[FusionConfig],
                            unfused_configs: Dict[str, TileConfig],
                            buffer_limit: int) -> Tuple[int, int]:
        """估算总体代价"""
        total_cycles = 0
        max_memory = 0

        # 融合组
        for fc in fusion_configs:
            group_cycles = 0
            group_memory = fc.total_buffer_size()

            for op_name in fc.fused_ops:
                op_spec = benchmark.ops[op_name]
                config = fc.tile_configs[op_name]
                cycles = self._estimate_cycles(config, op_spec)
                group_cycles += cycles

            # 融合收益：减少访存
            fusion_benefit = 0.8  # 假设融合带来 20% 收益
            total_cycles += int(group_cycles * fusion_benefit)
            max_memory = max(max_memory, group_memory)

        # 未融合算子
        for op_name, config in unfused_configs.items():
            op_spec = benchmark.ops[op_name]
            cycles = self._estimate_cycles(config, op_spec)
            total_cycles += cycles
            max_memory = max(max_memory, config.total_buffer_size())

        return total_cycles, max_memory

    def _estimate_compute_utilization(self, benchmark: Benchmark,
                                      fusion_configs: List[FusionConfig],
                                      unfused_configs: Dict[str, TileConfig]) -> float:
        """估算计算利用率"""
        # 效率感知策略利用率较高
        if fusion_configs:
            return min(0.75, self.target_compute_util)
        return 0.5

    def _generate_breakdown(self, benchmark: Benchmark,
                           fusion_configs: List[FusionConfig],
                           unfused_configs: Dict[str, TileConfig],
                           buffer_limit: int) -> Dict:
        """生成详细分析"""
        breakdown = {
            "fusion_groups": [],
            "unfused_ops": {},
        }

        for fc in fusion_configs:
            group_info = {
                "ops": fc.fused_ops,
                "pipeline_depth": fc.pipeline_depth,
                "total_buffer": fc.total_buffer_size(),
            }
            breakdown["fusion_groups"].append(group_info)

        for op_name, config in unfused_configs.items():
            breakdown["unfused_ops"][op_name] = {
                "tile_sizes": config.tile_sizes,
                "buffer_size": config.total_buffer_size(),
            }

        return breakdown
