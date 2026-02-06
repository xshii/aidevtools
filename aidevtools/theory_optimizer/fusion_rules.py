"""
融合规则配置

全局的算子融合规则库，支持:
- 两两算子融合规则 (FusionRule)
- 多算子融合模式 (FusionPattern)
- 基于 shape 的条件判断
- 融合收益计算

设计模式:
- Singleton 模式: 全局唯一配置
- Registry 模式: 规则注册和查找
- Composite 模式: 多算子模式组合
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Sequence
from enum import Enum


class FuseConstraint(Enum):
    """融合约束类型"""
    NONE = "none"                    # 无约束
    SAME_SHAPE = "same_shape"        # 要求形状相同
    CONTIGUOUS = "contiguous"        # 要求内存连续
    BROADCAST_OK = "broadcast_ok"    # 允许广播


# ============================================================
# 融合超参数 (支持 ML 校准)
# ============================================================


@dataclass
class FusionHyperParams:
    """
    融合相关的超参数

    这些参数可以通过机器学习校准来优化
    """
    # ==================== 组合加速比 ====================
    decay_base: float = 0.9          # 基础衰减因子
    decay_rate: float = 0.9          # 每层衰减率
    speedup_base: float = 1.0        # 基础加速比
    speedup_scale: float = 1.0       # 加速比缩放因子

    # ==================== 融合深度 ====================
    max_fusion_depth: int = 6        # 最大融合深度
    depth_penalty: float = 0.05      # 深度惩罚因子

    # ==================== 模式匹配 ====================
    pattern_bonus: float = 1.1       # 预定义模式奖励系数
    pairwise_ceiling: float = 2.5    # 两两组合加速比上限

    # ==================== IO 模型 ====================
    io_save_ratio: float = 0.3       # 融合节省的 IO 比例
    io_overlap_factor: float = 0.7   # IO 与计算重叠因子

    # ==================== Tile 相关 ====================
    tile_overhead_base: float = 0.02 # Tile 基础开销比例
    tile_size_scale: float = 1.0     # Tile 大小缩放
    double_buffer_benefit: float = 1.8  # 双缓冲收益

    # ==================== 流水线 ====================
    pipeline_efficiency: float = 0.85   # 流水线效率
    pipeline_fill_penalty: float = 0.1  # 流水填充惩罚
    pipeline_drain_penalty: float = 0.1 # 流水排空惩罚

    # ==================== 算子类型相关 ====================
    # 不同算子组合的基础收益修正
    matmul_elemwise_bonus: float = 1.2   # matmul + element-wise 奖励
    elemwise_chain_bonus: float = 1.3    # 连续 element-wise 奖励
    reduction_penalty: float = 0.9       # reduction 算子惩罚

    # ==================== Shape 相关 ====================
    small_shape_threshold: int = 1024    # 小 shape 阈值
    small_shape_penalty: float = 0.8     # 小 shape 融合惩罚
    large_shape_threshold: int = 65536   # 大 shape 阈值
    large_shape_bonus: float = 1.1       # 大 shape 融合奖励

    # ==================== 底噪/开销 (cycles) ====================
    # 算子提交底噪
    op_submit_base: float = 100.0        # 算子提交基础开销 (cycles)
    op_submit_per_arg: float = 10.0      # 每参数提交开销 (cycles)
    op_launch_latency: float = 50.0      # 算子启动延迟 (cycles)
    op_sync_overhead: float = 30.0       # 算子同步开销 (cycles)

    # DMA 底噪
    dma_submit_base: float = 80.0        # DMA 提交基础开销 (cycles)
    dma_setup_per_kb: float = 5.0        # 每 KB 的 DMA 设置开销 (cycles)
    dma_min_transfer: int = 256          # DMA 最小传输量 (bytes)
    dma_alignment_penalty: float = 1.2   # 非对齐传输惩罚因子

    # 融合后开销节省
    fuse_submit_save: float = 0.7        # 融合后提交开销节省比例
    fuse_sync_save: float = 0.8          # 融合后同步开销节省比例
    fuse_dma_save: float = 0.6           # 融合后 DMA 开销节省比例

    def to_vector(self) -> List[float]:
        """转换为向量（用于 ML 优化）"""
        return [
            # 组合加速比
            self.decay_base,
            self.decay_rate,
            self.speedup_base,
            self.speedup_scale,
            # 融合深度
            self.depth_penalty,
            # 模式匹配
            self.pattern_bonus,
            self.pairwise_ceiling,
            # IO 模型
            self.io_save_ratio,
            self.io_overlap_factor,
            # Tile 相关
            self.tile_overhead_base,
            self.tile_size_scale,
            self.double_buffer_benefit,
            # 流水线
            self.pipeline_efficiency,
            self.pipeline_fill_penalty,
            self.pipeline_drain_penalty,
            # 算子类型
            self.matmul_elemwise_bonus,
            self.elemwise_chain_bonus,
            self.reduction_penalty,
            # Shape 相关
            self.small_shape_penalty,
            self.large_shape_bonus,
            # 底噪/开销
            self.op_submit_base,
            self.op_submit_per_arg,
            self.op_launch_latency,
            self.op_sync_overhead,
            self.dma_submit_base,
            self.dma_setup_per_kb,
            self.dma_alignment_penalty,
            self.fuse_submit_save,
            self.fuse_sync_save,
            self.fuse_dma_save,
        ]

    @classmethod
    def from_vector(cls, vec: List[float]) -> "FusionHyperParams":
        """从向量创建"""
        return cls(
            decay_base=vec[0],
            decay_rate=vec[1],
            speedup_base=vec[2],
            speedup_scale=vec[3],
            depth_penalty=vec[4],
            pattern_bonus=vec[5],
            pairwise_ceiling=vec[6],
            io_save_ratio=vec[7],
            io_overlap_factor=vec[8],
            tile_overhead_base=vec[9],
            tile_size_scale=vec[10],
            double_buffer_benefit=vec[11],
            pipeline_efficiency=vec[12],
            pipeline_fill_penalty=vec[13],
            pipeline_drain_penalty=vec[14],
            matmul_elemwise_bonus=vec[15],
            elemwise_chain_bonus=vec[16],
            reduction_penalty=vec[17],
            small_shape_penalty=vec[18],
            large_shape_bonus=vec[19],
            # 底噪/开销
            op_submit_base=vec[20],
            op_submit_per_arg=vec[21],
            op_launch_latency=vec[22],
            op_sync_overhead=vec[23],
            dma_submit_base=vec[24],
            dma_setup_per_kb=vec[25],
            dma_alignment_penalty=vec[26],
            fuse_submit_save=vec[27],
            fuse_sync_save=vec[28],
            fuse_dma_save=vec[29],
        )

    @classmethod
    def param_names(cls) -> List[str]:
        """参数名列表"""
        return [
            # 组合加速比
            "decay_base", "decay_rate", "speedup_base", "speedup_scale",
            # 融合深度
            "depth_penalty",
            # 模式匹配
            "pattern_bonus", "pairwise_ceiling",
            # IO 模型
            "io_save_ratio", "io_overlap_factor",
            # Tile 相关
            "tile_overhead_base", "tile_size_scale", "double_buffer_benefit",
            # 流水线
            "pipeline_efficiency", "pipeline_fill_penalty", "pipeline_drain_penalty",
            # 算子类型
            "matmul_elemwise_bonus", "elemwise_chain_bonus", "reduction_penalty",
            # Shape 相关
            "small_shape_penalty", "large_shape_bonus",
            # 底噪/开销
            "op_submit_base", "op_submit_per_arg", "op_launch_latency", "op_sync_overhead",
            "dma_submit_base", "dma_setup_per_kb", "dma_alignment_penalty",
            "fuse_submit_save", "fuse_sync_save", "fuse_dma_save",
        ]

    @classmethod
    def param_bounds(cls) -> List[Tuple[float, float]]:
        """参数边界（用于优化）"""
        return [
            # 组合加速比
            (0.5, 1.0),    # decay_base
            (0.7, 1.0),    # decay_rate
            (0.8, 1.2),    # speedup_base
            (0.5, 2.0),    # speedup_scale
            # 融合深度
            (0.0, 0.2),    # depth_penalty
            # 模式匹配
            (1.0, 1.5),    # pattern_bonus
            (1.5, 4.0),    # pairwise_ceiling
            # IO 模型
            (0.1, 0.5),    # io_save_ratio
            (0.5, 1.0),    # io_overlap_factor
            # Tile 相关
            (0.01, 0.1),   # tile_overhead_base
            (0.5, 2.0),    # tile_size_scale
            (1.5, 2.0),    # double_buffer_benefit
            # 流水线
            (0.7, 0.95),   # pipeline_efficiency
            (0.05, 0.2),   # pipeline_fill_penalty
            (0.05, 0.2),   # pipeline_drain_penalty
            # 算子类型
            (1.0, 1.5),    # matmul_elemwise_bonus
            (1.1, 1.6),    # elemwise_chain_bonus
            (0.7, 1.0),    # reduction_penalty
            # Shape 相关
            (0.6, 1.0),    # small_shape_penalty
            (1.0, 1.3),    # large_shape_bonus
            # 底噪/开销 (cycles)
            (50, 200),     # op_submit_base
            (5, 20),       # op_submit_per_arg
            (20, 100),     # op_launch_latency
            (10, 60),      # op_sync_overhead
            (40, 150),     # dma_submit_base
            (2, 10),       # dma_setup_per_kb
            (1.0, 1.5),    # dma_alignment_penalty
            (0.5, 0.9),    # fuse_submit_save
            (0.6, 0.95),   # fuse_sync_save
            (0.4, 0.8),    # fuse_dma_save
        ]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            # 组合加速比
            "decay_base": self.decay_base,
            "decay_rate": self.decay_rate,
            "speedup_base": self.speedup_base,
            "speedup_scale": self.speedup_scale,
            # 融合深度
            "max_fusion_depth": self.max_fusion_depth,
            "depth_penalty": self.depth_penalty,
            # 模式匹配
            "pattern_bonus": self.pattern_bonus,
            "pairwise_ceiling": self.pairwise_ceiling,
            # IO 模型
            "io_save_ratio": self.io_save_ratio,
            "io_overlap_factor": self.io_overlap_factor,
            # Tile 相关
            "tile_overhead_base": self.tile_overhead_base,
            "tile_size_scale": self.tile_size_scale,
            "double_buffer_benefit": self.double_buffer_benefit,
            # 流水线
            "pipeline_efficiency": self.pipeline_efficiency,
            "pipeline_fill_penalty": self.pipeline_fill_penalty,
            "pipeline_drain_penalty": self.pipeline_drain_penalty,
            # 算子类型
            "matmul_elemwise_bonus": self.matmul_elemwise_bonus,
            "elemwise_chain_bonus": self.elemwise_chain_bonus,
            "reduction_penalty": self.reduction_penalty,
            # Shape 相关
            "small_shape_threshold": self.small_shape_threshold,
            "small_shape_penalty": self.small_shape_penalty,
            "large_shape_threshold": self.large_shape_threshold,
            "large_shape_bonus": self.large_shape_bonus,
            # 底噪/开销
            "op_submit_base": self.op_submit_base,
            "op_submit_per_arg": self.op_submit_per_arg,
            "op_launch_latency": self.op_launch_latency,
            "op_sync_overhead": self.op_sync_overhead,
            "dma_submit_base": self.dma_submit_base,
            "dma_setup_per_kb": self.dma_setup_per_kb,
            "dma_min_transfer": self.dma_min_transfer,
            "dma_alignment_penalty": self.dma_alignment_penalty,
            "fuse_submit_save": self.fuse_submit_save,
            "fuse_sync_save": self.fuse_sync_save,
            "fuse_dma_save": self.fuse_dma_save,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionHyperParams":
        """从字典创建"""
        return cls(
            # 组合加速比
            decay_base=data.get("decay_base", 0.9),
            decay_rate=data.get("decay_rate", 0.9),
            speedup_base=data.get("speedup_base", 1.0),
            speedup_scale=data.get("speedup_scale", 1.0),
            # 融合深度
            max_fusion_depth=int(data.get("max_fusion_depth", 6)),
            depth_penalty=data.get("depth_penalty", 0.05),
            # 模式匹配
            pattern_bonus=data.get("pattern_bonus", 1.1),
            pairwise_ceiling=data.get("pairwise_ceiling", 2.5),
            # IO 模型
            io_save_ratio=data.get("io_save_ratio", 0.3),
            io_overlap_factor=data.get("io_overlap_factor", 0.7),
            # Tile 相关
            tile_overhead_base=data.get("tile_overhead_base", 0.02),
            tile_size_scale=data.get("tile_size_scale", 1.0),
            double_buffer_benefit=data.get("double_buffer_benefit", 1.8),
            # 流水线
            pipeline_efficiency=data.get("pipeline_efficiency", 0.85),
            pipeline_fill_penalty=data.get("pipeline_fill_penalty", 0.1),
            pipeline_drain_penalty=data.get("pipeline_drain_penalty", 0.1),
            # 算子类型
            matmul_elemwise_bonus=data.get("matmul_elemwise_bonus", 1.2),
            elemwise_chain_bonus=data.get("elemwise_chain_bonus", 1.3),
            reduction_penalty=data.get("reduction_penalty", 0.9),
            # Shape 相关
            small_shape_threshold=int(data.get("small_shape_threshold", 1024)),
            small_shape_penalty=data.get("small_shape_penalty", 0.8),
            large_shape_threshold=int(data.get("large_shape_threshold", 65536)),
            large_shape_bonus=data.get("large_shape_bonus", 1.1),
            # 底噪/开销
            op_submit_base=data.get("op_submit_base", 100.0),
            op_submit_per_arg=data.get("op_submit_per_arg", 10.0),
            op_launch_latency=data.get("op_launch_latency", 50.0),
            op_sync_overhead=data.get("op_sync_overhead", 30.0),
            dma_submit_base=data.get("dma_submit_base", 80.0),
            dma_setup_per_kb=data.get("dma_setup_per_kb", 5.0),
            dma_min_transfer=int(data.get("dma_min_transfer", 256)),
            dma_alignment_penalty=data.get("dma_alignment_penalty", 1.2),
            fuse_submit_save=data.get("fuse_submit_save", 0.7),
            fuse_sync_save=data.get("fuse_sync_save", 0.8),
            fuse_dma_save=data.get("fuse_dma_save", 0.6),
        )


# ============================================================
# 多算子融合模式
# ============================================================


@dataclass
class FusionPattern:
    """
    多算子融合模式

    定义 3 个及以上算子的融合模式，例如:
    - matmul -> gelu -> matmul (FFN)
    - matmul -> softmax -> matmul (Attention)
    - add -> layernorm (Residual + LN)
    """
    name: str                              # 模式名称
    op_types: Tuple[str, ...]              # 算子类型序列
    fuse_speedup: float = 1.0              # 整体融合加速比
    priority: int = 0                      # 匹配优先级 (高优先)

    # 约束条件
    constraints: Dict[int, FuseConstraint] = field(default_factory=dict)

    # 条件函数: (shapes_list: List[Dict]) -> bool
    condition: Optional[Callable[[List[Dict]], bool]] = None

    # 收益计算函数: (shapes_list: List[Dict]) -> float
    speedup_func: Optional[Callable[[List[Dict]], float]] = None

    def __len__(self) -> int:
        return len(self.op_types)

    def matches(self, op_types: Sequence[str]) -> bool:
        """检查算子类型序列是否匹配此模式"""
        if len(op_types) != len(self.op_types):
            return False
        return all(a == b for a, b in zip(op_types, self.op_types))

    def matches_prefix(self, op_types: Sequence[str]) -> bool:
        """检查是否为此模式的前缀"""
        if len(op_types) > len(self.op_types):
            return False
        return all(a == b for a, b in zip(op_types, self.op_types))

    def can_fuse(self, shapes_list: List[Dict]) -> bool:
        """检查是否可以融合"""
        if self.condition:
            return self.condition(shapes_list)
        return True

    def get_speedup(self, shapes_list: List[Dict]) -> float:
        """获取融合加速比"""
        if self.speedup_func:
            return self.speedup_func(shapes_list)
        return self.fuse_speedup

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_types": list(self.op_types),
            "fuse_speedup": self.fuse_speedup,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionPattern":
        return cls(
            name=data["name"],
            op_types=tuple(data["op_types"]),
            fuse_speedup=data.get("fuse_speedup", 1.0),
            priority=data.get("priority", 0),
        )


# ============================================================
# 两两融合规则
# ============================================================


@dataclass
class FusionRule:
    """
    融合规则

    定义两类算子之间的融合特性
    """
    op_type_a: str                      # 第一个算子类型
    op_type_b: str                      # 第二个算子类型
    ratio: float = 1.0                  # 效率比 (融合后效率 / 分开效率)
    fuse_speedup: Optional[float] = None  # 融合加速比
    constraint: FuseConstraint = FuseConstraint.NONE
    bidirectional: bool = True          # 是否双向适用

    # 条件函数：基于 shape 判断是否可融合
    # 签名: (shapes_a: Dict, shapes_b: Dict) -> bool
    condition: Optional[Callable[[Dict, Dict], bool]] = None

    # 效率计算函数：基于 shape 计算实际效率比
    # 签名: (shapes_a: Dict, shapes_b: Dict) -> float
    ratio_func: Optional[Callable[[Dict, Dict], float]] = None

    def matches(self, type_a: str, type_b: str) -> bool:
        """检查是否匹配"""
        if self.op_type_a == type_a and self.op_type_b == type_b:
            return True
        if self.bidirectional and self.op_type_a == type_b and self.op_type_b == type_a:
            return True
        return False

    def get_ratio(self, shapes_a: Dict, shapes_b: Dict) -> float:
        """获取实际效率比"""
        if self.ratio_func:
            return self.ratio_func(shapes_a, shapes_b)
        return self.ratio

    def can_fuse(self, shapes_a: Dict, shapes_b: Dict) -> bool:
        """检查是否可以融合"""
        # 检查自定义条件
        if self.condition and not self.condition(shapes_a, shapes_b):
            return False

        # 检查约束
        if self.constraint == FuseConstraint.SAME_SHAPE:
            # 检查关键维度是否匹配
            common_keys = set(shapes_a.keys()) & set(shapes_b.keys())
            for key in common_keys:
                if shapes_a[key] != shapes_b[key]:
                    return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "op_type_a": self.op_type_a,
            "op_type_b": self.op_type_b,
            "ratio": self.ratio,
            "fuse_speedup": self.fuse_speedup,
            "constraint": self.constraint.value,
            "bidirectional": self.bidirectional,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FusionRule":
        """从字典创建"""
        return cls(
            op_type_a=data["op_type_a"],
            op_type_b=data["op_type_b"],
            ratio=data.get("ratio", 1.0),
            fuse_speedup=data.get("fuse_speedup"),
            constraint=FuseConstraint(data.get("constraint", "none")),
            bidirectional=data.get("bidirectional", True),
        )


class FusionRules:
    """
    全局融合规则库 (Singleton)

    管理所有算子融合规则，包括:
    - 两两融合规则 (FusionRule)
    - 多算子融合模式 (FusionPattern)
    - 融合超参数 (FusionHyperParams)
    """
    _instance: Optional["FusionRules"] = None
    _rules: List[FusionRule]
    _patterns: List[FusionPattern]
    _index: Dict[Tuple[str, str], FusionRule]
    _hyper_params: FusionHyperParams

    def __new__(cls) -> "FusionRules":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._rules = []
            cls._instance._patterns = []
            cls._instance._index = {}
            cls._instance._hyper_params = FusionHyperParams()
            cls._instance._load_defaults()
        return cls._instance

    @property
    def hyper_params(self) -> FusionHyperParams:
        """获取超参数"""
        return self._hyper_params

    @hyper_params.setter
    def hyper_params(self, params: FusionHyperParams) -> None:
        """设置超参数"""
        self._hyper_params = params

    @classmethod
    def get_instance(cls) -> "FusionRules":
        """获取单例实例"""
        return cls()

    @classmethod
    def reset(cls) -> None:
        """重置单例（主要用于测试）"""
        cls._instance = None

    def _load_defaults(self) -> None:
        """加载默认规则"""
        self._load_default_rules()
        self._load_default_patterns()

    def _load_default_rules(self) -> None:
        """加载默认两两规则"""
        # MatMul + Element-wise 融合规则
        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="relu",
            ratio=0.95,
            fuse_speedup=1.15,
            constraint=FuseConstraint.SAME_SHAPE,
        ))

        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="gelu",
            ratio=0.90,
            fuse_speedup=1.25,
            constraint=FuseConstraint.SAME_SHAPE,
        ))

        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="add",
            ratio=0.92,
            fuse_speedup=1.20,
            constraint=FuseConstraint.BROADCAST_OK,
        ))

        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="mul",
            ratio=0.93,
            fuse_speedup=1.18,
            constraint=FuseConstraint.BROADCAST_OK,
        ))

        # Element-wise 连续融合
        self.add_rule(FusionRule(
            op_type_a="add",
            op_type_b="relu",
            ratio=0.98,
            fuse_speedup=1.30,
        ))

        self.add_rule(FusionRule(
            op_type_a="mul",
            op_type_b="add",
            ratio=0.97,
            fuse_speedup=1.25,
        ))

        # GELU 分解融合
        self.add_rule(FusionRule(
            op_type_a="gelu",
            op_type_b="add",
            ratio=0.88,
            fuse_speedup=1.10,
        ))

        # Softmax 相关
        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="softmax",
            ratio=0.85,
            fuse_speedup=1.20,
        ))

        self.add_rule(FusionRule(
            op_type_a="softmax",
            op_type_b="matmul",
            ratio=0.85,
            fuse_speedup=1.15,
            bidirectional=False,  # 顺序敏感
        ))

        # LayerNorm 融合
        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="layernorm",
            ratio=0.80,
            fuse_speedup=1.10,
        ))

        self.add_rule(FusionRule(
            op_type_a="add",
            op_type_b="layernorm",
            ratio=0.90,
            fuse_speedup=1.20,
        ))

        # 连续 MatMul
        self.add_rule(FusionRule(
            op_type_a="matmul",
            op_type_b="matmul",
            ratio=0.75,  # 通常不如分开高效
            fuse_speedup=0.95,
            # 特殊条件：只有当中间维度匹配时才融合
            condition=lambda a, b: a.get("N") == b.get("M"),
        ))

    def _load_default_patterns(self) -> None:
        """加载默认多算子融合模式"""

        # FFN 模式: matmul -> activation -> matmul
        self.add_pattern(FusionPattern(
            name="ffn_gelu",
            op_types=("matmul", "gelu", "matmul"),
            fuse_speedup=1.8,  # 整体融合收益 > 两两融合叠加
            priority=10,
        ))

        self.add_pattern(FusionPattern(
            name="ffn_relu",
            op_types=("matmul", "relu", "matmul"),
            fuse_speedup=1.6,
            priority=10,
        ))

        self.add_pattern(FusionPattern(
            name="ffn_silu",
            op_types=("matmul", "silu", "matmul"),
            fuse_speedup=1.7,
            priority=10,
        ))

        # Attention 核心模式: matmul -> softmax -> matmul
        self.add_pattern(FusionPattern(
            name="attention_core",
            op_types=("matmul", "softmax", "matmul"),
            fuse_speedup=2.5,  # Flash Attention 级别收益
            priority=20,
        ))

        # SwiGLU 模式: matmul -> silu -> mul
        self.add_pattern(FusionPattern(
            name="swiglu",
            op_types=("matmul", "silu", "mul"),
            fuse_speedup=1.5,
            priority=15,
        ))

        # 完整 SwiGLU FFN: gate -> silu -> mul -> down
        self.add_pattern(FusionPattern(
            name="swiglu_ffn",
            op_types=("matmul", "silu", "mul", "matmul"),
            fuse_speedup=2.0,
            priority=25,
        ))

        # Residual + LayerNorm
        self.add_pattern(FusionPattern(
            name="residual_ln",
            op_types=("add", "layernorm"),
            fuse_speedup=1.4,
            priority=5,
        ))

        # MatMul + Bias + Activation
        self.add_pattern(FusionPattern(
            name="linear_bias_gelu",
            op_types=("matmul", "add", "gelu"),
            fuse_speedup=1.5,
            priority=8,
        ))

        self.add_pattern(FusionPattern(
            name="linear_bias_relu",
            op_types=("matmul", "add", "relu"),
            fuse_speedup=1.4,
            priority=8,
        ))

        # 四算子: matmul -> add -> activation -> matmul (带 bias 的 FFN)
        self.add_pattern(FusionPattern(
            name="ffn_with_bias",
            op_types=("matmul", "add", "gelu", "matmul"),
            fuse_speedup=2.0,
            priority=15,
        ))

    # ==================== 两两规则管理 ====================

    def add_rule(self, rule: FusionRule) -> None:
        """添加规则"""
        self._rules.append(rule)
        # 建立索引
        self._index[(rule.op_type_a, rule.op_type_b)] = rule
        if rule.bidirectional:
            self._index[(rule.op_type_b, rule.op_type_a)] = rule

    def get_rule(self, op_type_a: str, op_type_b: str) -> Optional[FusionRule]:
        """获取融合规则"""
        return self._index.get((op_type_a, op_type_b))

    def can_fuse(self, op_type_a: str, op_type_b: str,
                shapes_a: Optional[Dict] = None,
                shapes_b: Optional[Dict] = None) -> bool:
        """检查两个算子类型是否可融合"""
        rule = self.get_rule(op_type_a, op_type_b)
        if not rule:
            return False

        if shapes_a and shapes_b:
            return rule.can_fuse(shapes_a, shapes_b)

        return True

    def get_ratio(self, op_type_a: str, op_type_b: str,
                 shapes_a: Optional[Dict] = None,
                 shapes_b: Optional[Dict] = None) -> float:
        """获取效率比"""
        rule = self.get_rule(op_type_a, op_type_b)
        if not rule:
            return 1.0

        if shapes_a and shapes_b:
            return rule.get_ratio(shapes_a, shapes_b)

        return rule.ratio

    def get_speedup(self, op_type_a: str, op_type_b: str) -> Optional[float]:
        """获取融合加速比"""
        rule = self.get_rule(op_type_a, op_type_b)
        if rule:
            return rule.fuse_speedup
        return None

    def list_rules(self) -> List[FusionRule]:
        """列出所有两两规则"""
        return list(self._rules)

    def list_fusable_pairs(self) -> List[Tuple[str, str]]:
        """列出所有可融合的算子对"""
        pairs = set()
        for rule in self._rules:
            pairs.add((rule.op_type_a, rule.op_type_b))
            if rule.bidirectional:
                pairs.add((rule.op_type_b, rule.op_type_a))
        return list(pairs)

    # ==================== 多算子模式管理 ====================

    def add_pattern(self, pattern: FusionPattern) -> None:
        """添加多算子融合模式"""
        self._patterns.append(pattern)
        # 按优先级排序
        self._patterns.sort(key=lambda p: p.priority, reverse=True)

    def get_pattern(self, name: str) -> Optional[FusionPattern]:
        """根据名称获取模式"""
        return next((p for p in self._patterns if p.name == name), None)

    def match_pattern(self, op_types: Sequence[str],
                     shapes_list: Optional[List[Dict]] = None) -> Optional[FusionPattern]:
        """
        匹配多算子融合模式

        Args:
            op_types: 算子类型序列
            shapes_list: 对应的 shapes 列表

        Returns:
            匹配的最高优先级模式，或 None
        """
        for pattern in self._patterns:  # 已按优先级排序
            if pattern.matches(op_types):
                if shapes_list and not pattern.can_fuse(shapes_list):
                    continue
                return pattern
        return None

    def find_fusable_groups(self, op_types: Sequence[str],
                           shapes_list: Optional[List[Dict]] = None,
                           max_group_size: int = 6) -> List[Tuple[int, int, FusionPattern]]:
        """
        在算子序列中查找所有可融合的组

        使用贪心算法，优先匹配更长/更高优先级的模式

        Args:
            op_types: 算子类型序列
            shapes_list: 对应的 shapes 列表
            max_group_size: 最大融合组大小

        Returns:
            List of (start_idx, end_idx, pattern)
        """
        n = len(op_types)
        groups = []
        used = [False] * n

        # 按模式长度和优先级排序后匹配
        for pattern in self._patterns:
            pattern_len = len(pattern)
            if pattern_len > max_group_size:
                continue

            # 滑动窗口匹配
            for i in range(n - pattern_len + 1):
                # 检查是否已被使用
                if any(used[i:i + pattern_len]):
                    continue

                window_types = op_types[i:i + pattern_len]
                if not pattern.matches(window_types):
                    continue

                # 检查 shapes 条件
                if shapes_list:
                    window_shapes = shapes_list[i:i + pattern_len]
                    if not pattern.can_fuse(window_shapes):
                        continue

                # 匹配成功
                groups.append((i, i + pattern_len, pattern))
                for j in range(i, i + pattern_len):
                    used[j] = True

        # 对未匹配的连续算子，尝试用两两规则组成组
        groups.extend(self._find_pairwise_groups(op_types, shapes_list, used))

        # 按起始位置排序
        groups.sort(key=lambda x: x[0])
        return groups

    def _find_pairwise_groups(self, op_types: Sequence[str],
                              shapes_list: Optional[List[Dict]],
                              used: List[bool]) -> List[Tuple[int, int, FusionPattern]]:
        """使用两两规则查找融合组"""
        n = len(op_types)
        groups = []

        i = 0
        while i < n - 1:
            if used[i]:
                i += 1
                continue

            # 尝试扩展组
            group_start = i
            group_end = i + 1

            while group_end < n and not used[group_end]:
                type_a = op_types[group_end - 1]
                type_b = op_types[group_end]

                shapes_a = shapes_list[group_end - 1] if shapes_list else None
                shapes_b = shapes_list[group_end] if shapes_list else None

                if self.can_fuse(type_a, type_b, shapes_a, shapes_b):
                    group_end += 1
                else:
                    break

            # 如果组大小 >= 2，创建动态模式
            if group_end - group_start >= 2:
                group_types = tuple(op_types[group_start:group_end])
                group_shapes = shapes_list[group_start:group_end] if shapes_list else None

                # 计算组合加速比
                combined_speedup = self._calculate_combined_speedup(
                    group_types, group_shapes
                )

                dynamic_pattern = FusionPattern(
                    name=f"dynamic_{'_'.join(group_types)}",
                    op_types=group_types,
                    fuse_speedup=combined_speedup,
                    priority=-1,  # 低于预定义模式
                )

                groups.append((group_start, group_end, dynamic_pattern))
                for j in range(group_start, group_end):
                    used[j] = True

            i = group_end

        return groups

    def _calculate_combined_speedup(self, op_types: Tuple[str, ...],
                                    shapes_list: Optional[List[Dict]]) -> float:
        """
        计算两两规则组合的加速比

        使用超参数控制组合方式:
        - decay_base: 基础衰减因子
        - decay_rate: 每层衰减率
        - speedup_scale: 加速比缩放
        - depth_penalty: 深度惩罚
        - pairwise_ceiling: 加速比上限
        """
        hp = self._hyper_params

        if len(op_types) < 2:
            return hp.speedup_base

        # 累积两两加速比，带衰减
        combined = hp.speedup_base
        decay = hp.decay_base

        for i in range(len(op_types) - 1):
            type_a = op_types[i]
            type_b = op_types[i + 1]

            speedup = self.get_speedup(type_a, type_b)
            if speedup:
                # 加速比增益 = (speedup - 1) * scale * decay
                gain = (speedup - 1.0) * hp.speedup_scale * decay
                combined *= (1.0 + gain)
                decay *= hp.decay_rate

        # 深度惩罚
        depth = len(op_types)
        if depth > 2:
            combined *= (1.0 - hp.depth_penalty * (depth - 2))

        # 上限
        combined = min(combined, hp.pairwise_ceiling)

        return max(1.0, combined)  # 至少为 1.0

    def list_patterns(self) -> List[FusionPattern]:
        """列出所有多算子模式"""
        return list(self._patterns)

    def get_pattern_speedup(self, op_types: Sequence[str],
                           shapes_list: Optional[List[Dict]] = None) -> float:
        """获取多算子融合的加速比"""
        pattern = self.match_pattern(op_types, shapes_list)
        if pattern:
            if shapes_list:
                return pattern.get_speedup(shapes_list)
            return pattern.fuse_speedup

        # 回退到两两规则组合
        return self._calculate_combined_speedup(tuple(op_types), shapes_list)

    def save(self, path: str) -> None:
        """保存规则、模式和超参数到文件"""
        data = {
            "version": "2.0",
            "hyper_params": self._hyper_params.to_dict(),
            "rules": [rule.to_dict() for rule in self._rules],
            "patterns": [pattern.to_dict() for pattern in self._patterns],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str, merge: bool = True) -> Tuple[int, int]:
        """
        从文件加载规则、模式和超参数

        Args:
            path: 文件路径
            merge: 是否合并（True）还是替换（False）

        Returns:
            (加载的规则数量, 加载的模式数量)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not merge:
            self._rules.clear()
            self._patterns.clear()
            self._index.clear()

        # 加载超参数
        if "hyper_params" in data:
            self._hyper_params = FusionHyperParams.from_dict(data["hyper_params"])

        rule_count = 0
        for rule_data in data.get("rules", []):
            rule = FusionRule.from_dict(rule_data)
            self.add_rule(rule)
            rule_count += 1

        pattern_count = 0
        for pattern_data in data.get("patterns", []):
            pattern = FusionPattern.from_dict(pattern_data)
            self.add_pattern(pattern)
            pattern_count += 1

        return rule_count, pattern_count

    def summary(self) -> str:
        """生成摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append("Fusion Rules Summary")
        lines.append("=" * 60)
        lines.append(f"Pairwise rules: {len(self._rules)}")
        lines.append(f"Multi-op patterns: {len(self._patterns)}")
        lines.append("")

        # 多算子模式（按优先级）
        if self._patterns:
            lines.append("Multi-Operator Patterns:")
            lines.append("-" * 40)
            for pattern in self._patterns[:10]:  # 显示前 10 个
                ops_str = " → ".join(pattern.op_types)
                lines.append(f"  [{pattern.priority:2d}] {pattern.name}:")
                lines.append(f"       {ops_str}")
                lines.append(f"       speedup={pattern.fuse_speedup:.2f}x")
            if len(self._patterns) > 10:
                lines.append(f"  ... and {len(self._patterns) - 10} more patterns")
            lines.append("")

        # 两两规则
        lines.append("Pairwise Rules:")
        lines.append("-" * 40)
        by_type: Dict[str, List[FusionRule]] = {}
        for rule in self._rules:
            if rule.op_type_a not in by_type:
                by_type[rule.op_type_a] = []
            by_type[rule.op_type_a].append(rule)

        for op_type, rules in sorted(by_type.items()):
            lines.append(f"  {op_type}:")
            for rule in rules:
                speedup = f", speedup={rule.fuse_speedup:.2f}" if rule.fuse_speedup else ""
                bidir = " (↔)" if rule.bidirectional else " (→)"
                lines.append(f"    + {rule.op_type_b}: ratio={rule.ratio:.2f}{speedup}{bidir}")

        return "\n".join(lines)


# 便捷函数
def get_fusion_rules() -> FusionRules:
    """获取全局融合规则库"""
    return FusionRules.get_instance()


def can_fuse(op_type_a: str, op_type_b: str,
            shapes_a: Optional[Dict] = None,
            shapes_b: Optional[Dict] = None) -> bool:
    """检查两个算子类型是否可融合"""
    return get_fusion_rules().can_fuse(op_type_a, op_type_b, shapes_a, shapes_b)


def get_fuse_ratio(op_type_a: str, op_type_b: str,
                  shapes_a: Optional[Dict] = None,
                  shapes_b: Optional[Dict] = None) -> float:
    """获取融合效率比"""
    return get_fusion_rules().get_ratio(op_type_a, op_type_b, shapes_a, shapes_b)


def get_fuse_speedup(op_type_a: str, op_type_b: str) -> Optional[float]:
    """获取融合加速比"""
    return get_fusion_rules().get_speedup(op_type_a, op_type_b)
