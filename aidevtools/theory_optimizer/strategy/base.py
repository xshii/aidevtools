"""
Tiling 策略基类

设计模式:
- Strategy 模式: 定义策略接口
- Template Method 模式: 基类定义算法骨架
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple, Any
from enum import Enum

from ..benchmark import Benchmark, OpSpec, FusePair


class TileAxis(Enum):
    """Tile 维度"""
    M = "M"
    N = "N"
    K = "K"
    BATCH = "batch"
    SEQ = "seq"
    HIDDEN = "hidden"


@dataclass
class TileConfig:
    """单个算子的 Tile 配置"""
    op_name: str
    tile_sizes: Dict[str, int] = field(default_factory=dict)  # axis -> size
    loop_order: List[str] = field(default_factory=list)       # 循环顺序
    parallel_axes: List[str] = field(default_factory=list)    # 并行维度

    # 内存分配
    buffer_sizes: Dict[str, int] = field(default_factory=dict)  # buffer_name -> size
    double_buffer: bool = False

    # 计算单元分配
    cube_ratio: float = 1.0   # Cube 单元使用比例
    vector_ratio: float = 0.0  # Vector 单元使用比例

    def total_buffer_size(self) -> int:
        """总 buffer 大小"""
        total = sum(self.buffer_sizes.values())
        return total * 2 if self.double_buffer else total

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "op_name": self.op_name,
            "tile_sizes": self.tile_sizes,
            "loop_order": self.loop_order,
            "parallel_axes": self.parallel_axes,
            "buffer_sizes": self.buffer_sizes,
            "double_buffer": self.double_buffer,
            "cube_ratio": self.cube_ratio,
            "vector_ratio": self.vector_ratio,
        }


@dataclass
class FusionConfig:
    """融合配置"""
    fused_ops: List[str]                    # 融合的算子列表
    tile_configs: Dict[str, TileConfig]     # 每个算子的 tile 配置
    shared_buffers: Dict[str, int] = field(default_factory=dict)  # 共享 buffer
    pipeline_depth: int = 1                 # 流水深度

    def total_buffer_size(self) -> int:
        """总 buffer 大小（含共享）"""
        individual = sum(tc.total_buffer_size() for tc in self.tile_configs.values())
        shared = sum(self.shared_buffers.values())
        return individual + shared


@dataclass
class TilingResult:
    """Tiling 结果"""
    benchmark: Benchmark
    fusion_configs: List[FusionConfig]
    unfused_configs: Dict[str, TileConfig]  # 未融合算子的配置

    # 评估指标
    estimated_cycles: int = 0
    memory_footprint: int = 0
    compute_utilization: float = 0.0
    memory_utilization: float = 0.0

    # 详细分析
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            f"Tiling Result Summary",
            f"=" * 40,
            f"Fusion groups: {len(self.fusion_configs)}",
            f"Unfused ops: {len(self.unfused_configs)}",
            f"Estimated cycles: {self.estimated_cycles:,}",
            f"Memory footprint: {self.memory_footprint:,} bytes",
            f"Compute utilization: {self.compute_utilization:.1%}",
            f"Memory utilization: {self.memory_utilization:.1%}",
        ]
        return "\n".join(lines)


class TilingStrategy(ABC):
    """
    Tiling 策略基类 (Strategy 模式)

    定义策略接口，子类实现具体算法
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass

    @property
    def description(self) -> str:
        """策略描述"""
        return ""

    @abstractmethod
    def tile(self, benchmark: Benchmark, constraints: Optional[Dict] = None) -> TilingResult:
        """
        执行 tiling

        Args:
            benchmark: 待优化的 benchmark
            constraints: 约束条件
                - max_buffer_size: 最大 buffer 大小
                - min_tile_size: 最小 tile 大小
                - target_utilization: 目标利用率

        Returns:
            TilingResult: tiling 结果
        """
        pass

    # Template Method: 定义算法骨架
    def optimize(self, benchmark: Benchmark,
                 constraints: Optional[Dict] = None,
                 iterations: int = 1) -> TilingResult:
        """
        优化入口 (Template Method)

        子类可以覆盖各个步骤，但保持整体流程
        """
        # 1. 预处理
        benchmark = self._preprocess(benchmark)

        # 2. 初始 tiling
        result = self.tile(benchmark, constraints)

        # 3. 迭代优化
        for i in range(iterations - 1):
            result = self._refine(result, constraints)

        # 4. 后处理
        result = self._postprocess(result)

        return result

    def _preprocess(self, benchmark: Benchmark) -> Benchmark:
        """预处理 hook，子类可覆盖"""
        return benchmark

    def _refine(self, result: TilingResult,
                constraints: Optional[Dict] = None) -> TilingResult:
        """优化迭代 hook，子类可覆盖"""
        return result

    def _postprocess(self, result: TilingResult) -> TilingResult:
        """后处理 hook，子类可覆盖"""
        return result

    # 辅助方法
    def _calculate_tile_size(self, shape: Dict[str, int],
                            buffer_limit: int,
                            dtype_size: int = 2) -> Dict[str, int]:
        """
        计算 tile 大小

        简单启发式：按比例缩放各维度
        """
        total_elements = 1
        for v in shape.values():
            total_elements *= v

        total_bytes = total_elements * dtype_size
        if total_bytes <= buffer_limit:
            return dict(shape)

        # 缩放因子
        scale = (buffer_limit / total_bytes) ** (1 / len(shape))

        tile_sizes = {}
        for axis, size in shape.items():
            tile_sizes[axis] = max(1, int(size * scale))

        return tile_sizes

    def _estimate_cycles(self, tile_config: TileConfig,
                        op_spec: OpSpec) -> int:
        """估算单个 tile 的执行周期"""
        # 简化估算
        compute = 1
        for axis, size in tile_config.tile_sizes.items():
            compute *= size

        # 考虑计算单元效率
        if op_spec.compute_unit == "cube":
            efficiency = tile_config.cube_ratio * 0.8
        elif op_spec.compute_unit == "vector":
            efficiency = tile_config.vector_ratio * 0.6
        else:
            efficiency = max(tile_config.cube_ratio * 0.8,
                           tile_config.vector_ratio * 0.6)

        if efficiency > 0:
            return int(compute / efficiency)
        return compute


class StrategyRegistry:
    """
    策略注册表 (Registry 模式)

    管理所有可用策略
    """
    _strategies: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """装饰器：注册策略"""
        def decorator(strategy_class: type):
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """获取策略类"""
        return cls._strategies.get(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[TilingStrategy]:
        """创建策略实例"""
        strategy_class = cls.get(name)
        if strategy_class:
            return strategy_class(**kwargs)
        return None

    @classmethod
    def list_strategies(cls) -> List[str]:
        """列出所有策略"""
        return list(cls._strategies.keys())
