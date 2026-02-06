"""
Benchmark 定义模块

设计模式:
- Builder 模式: Benchmark 链式构建
- Factory 模式: BenchmarkSuite 预置用例工厂
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from aidevtools.analysis.profile import OpProfile, dtype_bytes


# ============================================================
# 枚举定义
# ============================================================


class OpType(Enum):
    """算子类型枚举"""

    MATMUL = "matmul"
    CONV = "conv"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SOFTMAX = "softmax"
    LAYERNORM = "layernorm"
    RMSNORM = "rmsnorm"
    ADD = "add"
    MUL = "mul"
    REDUCE = "reduce"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    EMBEDDING = "embedding"
    ATTENTION = "attention"


class ComputeUnit(Enum):
    """计算单元类型"""

    CUBE = "cube"
    VECTOR = "vector"
    MIXED = "mixed"
    AUTO = "auto"


# ============================================================
# 计算单元配置
# ============================================================


@dataclass
class StageUnitCfg:
    """分阶段计算单元配置"""

    stage_name: str  # "compute" | "reduce" | "epilogue"
    unit: str  # "cube" | "vector"
    flops_ratio: float = 1.0  # 该阶段占总 FLOPs 比例


@dataclass
class ComputeUnitCfg:
    """
    计算单元配置

    支持:
    - 单一单元: primary="cube" 或 "vector"
    - 混合模式: primary + secondary 并行
    - 分阶段: 通过 stages 配置
    - 自动选择: primary="auto"
    """

    primary: str = "auto"  # "cube" | "vector" | "auto"
    secondary: Optional[str] = None  # 并行单元
    parallel_ratio: float = 1.0  # 主单元比例
    stages: List[StageUnitCfg] = field(default_factory=list)

    def is_mixed(self) -> bool:
        """是否为混合模式"""
        return self.secondary is not None or len(self.stages) > 0

    def get_effective_unit(self, op_type: OpType, shapes: Dict[str, int]) -> str:
        """获取实际使用的计算单元"""
        if self.primary != "auto":
            return self.primary

        # 自动选择规则
        auto_rules = {
            OpType.MATMUL: "cube",
            OpType.CONV: "cube",
            OpType.RELU: "vector",
            OpType.GELU: "vector",
            OpType.SILU: "vector",
            OpType.SOFTMAX: "vector",
            OpType.LAYERNORM: "vector",
            OpType.RMSNORM: "vector",
            OpType.ADD: "vector",
            OpType.MUL: "vector",
            OpType.REDUCE: "vector",
        }
        return auto_rules.get(op_type, "vector")

    def get_parallel_config(self) -> Dict[str, Any]:
        """获取并行配置"""
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "parallel_ratio": self.parallel_ratio,
            "stages": [
                {"name": s.stage_name, "unit": s.unit, "ratio": s.flops_ratio}
                for s in self.stages
            ],
        }


# ============================================================
# 算子规格
# ============================================================


@dataclass
class OpSpec:
    """
    算子规格定义

    包含形状、数据类型、计算单元配置等信息
    """

    name: str
    op_type: OpType
    shapes: Dict[str, int] = field(default_factory=dict)
    dtype: str = "fp16"
    compute_unit: ComputeUnitCfg = field(default_factory=ComputeUnitCfg)
    efficiency: float = 1.0  # 用户指定的效率系数

    def to_profile(self) -> OpProfile:
        """转换为 OpProfile"""
        flops, input_bytes, output_bytes = self._compute_cost()
        unit = self.compute_unit.get_effective_unit(self.op_type, self.shapes)

        return OpProfile(
            name=self.name,
            op_type=self.op_type.value,
            shapes=self.shapes,
            dtype=self.dtype,
            flops=flops,
            compute_unit=unit,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )

    def _compute_cost(self) -> Tuple[int, int, int]:
        """计算 FLOPs 和 IO bytes"""
        db = dtype_bytes(self.dtype)

        if self.op_type == OpType.MATMUL:
            M = self.shapes.get("M", 1)
            K = self.shapes.get("K", 1)
            N = self.shapes.get("N", 1)
            flops = 2 * M * K * N
            input_bytes = (M * K + K * N) * db
            output_bytes = M * N * db

        elif self.op_type == OpType.CONV:
            # 简化: 假设 NCHW 格式
            N = self.shapes.get("N", 1)
            C = self.shapes.get("C", 1)
            H = self.shapes.get("H", 1)
            W = self.shapes.get("W", 1)
            K = self.shapes.get("K", 1)
            R = self.shapes.get("R", 3)
            S = self.shapes.get("S", 3)
            flops = 2 * N * K * H * W * C * R * S
            input_bytes = N * C * H * W * db + K * C * R * S * db
            output_bytes = N * K * H * W * db

        elif self.op_type in (OpType.RELU, OpType.ADD, OpType.MUL):
            size = self._get_size()
            flops = size
            input_bytes = size * db
            output_bytes = size * db

        elif self.op_type in (OpType.GELU, OpType.SILU):
            size = self._get_size()
            flops = 8 * size  # GELU/SiLU 约 8 ops per element
            input_bytes = size * db
            output_bytes = size * db

        elif self.op_type == OpType.SOFTMAX:
            size = self._get_size()
            flops = 5 * size  # max, sub, exp, sum, div
            input_bytes = size * db
            output_bytes = size * db

        elif self.op_type in (OpType.LAYERNORM, OpType.RMSNORM):
            M = self.shapes.get("M", 1)
            N = self.shapes.get("N", 1)
            flops = 8 * M * N
            input_bytes = M * N * db + 2 * N * db  # + gamma/beta
            output_bytes = M * N * db

        elif self.op_type == OpType.REDUCE:
            size = self._get_size()
            flops = size
            input_bytes = size * db
            output_bytes = self.shapes.get("output_size", 1) * db

        else:
            flops = 0
            input_bytes = 0
            output_bytes = 0

        return int(flops), int(input_bytes), int(output_bytes)

    def _get_size(self) -> int:
        """获取元素总数"""
        if "size" in self.shapes:
            return self.shapes["size"]
        M = self.shapes.get("M", 1)
        N = self.shapes.get("N", 1)
        return M * N


# ============================================================
# 融合对 (用于 Benchmark 内部覆盖全局规则)
# ============================================================


@dataclass
class FusePair:
    """
    融合算子对

    可用于覆盖全局融合规则的特定配置
    """

    op_a: str  # 前算子名
    op_b: str  # 后算子名
    ratio: float  # 效率比 (覆盖全局规则)
    fuse_speedup: Optional[float] = None  # 融合加速比


# ============================================================
# Benchmark (Builder 模式)
# ============================================================


@dataclass
class Benchmark:
    """
    Benchmark 用例定义

    使用 Builder 模式支持链式构建。
    融合规则优先从全局 FusionRules 获取，可通过 override_pairs 覆盖。
    """

    name: str
    description: str = ""
    ops: List[OpSpec] = field(default_factory=list)
    override_pairs: List[FusePair] = field(default_factory=list)  # 覆盖全局规则
    category: str = ""  # "mlp" | "attention" | "conv"
    model_source: str = ""  # "bert-base" | "llama-7b"

    # Builder 方法
    def add_op(
        self,
        name: str,
        op_type: str,
        compute_unit: str = "auto",
        **shapes,
    ) -> "Benchmark":
        """添加算子 (Builder 模式)"""
        unit_cfg = ComputeUnitCfg(primary=compute_unit)
        self.ops.append(
            OpSpec(
                name=name,
                op_type=OpType(op_type),
                shapes=shapes,
                compute_unit=unit_cfg,
            )
        )
        return self

    def override_pair(
        self,
        op_a: str,
        op_b: str,
        ratio: float,
        fuse_speedup: Optional[float] = None,
    ) -> "Benchmark":
        """覆盖特定算子对的融合规则"""
        self.override_pairs.append(FusePair(op_a, op_b, ratio, fuse_speedup))
        return self

    def set_compute_unit(self, op_name: str, unit: str) -> "Benchmark":
        """设置算子计算单元"""
        op = self.get_op(op_name)
        if op:
            op.compute_unit = ComputeUnitCfg(primary=unit)
        return self

    def set_mixed_compute(
        self,
        op_name: str,
        primary: str,
        secondary: str,
        ratio: float = 0.8,
    ) -> "Benchmark":
        """设置混合计算单元"""
        op = self.get_op(op_name)
        if op:
            op.compute_unit = ComputeUnitCfg(
                primary=primary,
                secondary=secondary,
                parallel_ratio=ratio,
            )
        return self

    def get_op(self, name: str) -> Optional[OpSpec]:
        """根据名称获取算子"""
        return next((op for op in self.ops if op.name == name), None)

    def get_fusable_pairs(self) -> List[Tuple[str, str, float, Optional[float]]]:
        """
        获取所有可融合的算子对 (两两规则)

        优先级：override_pairs > 全局 FusionRules

        Returns:
            List of (op_a, op_b, ratio, fuse_speedup)
        """
        from .fusion_rules import get_fusion_rules

        rules = get_fusion_rules()
        result = []

        # 检查相邻算子对
        for i in range(len(self.ops) - 1):
            op_a = self.ops[i]
            op_b = self.ops[i + 1]

            # 检查是否有覆盖
            override = self._get_override(op_a.name, op_b.name)
            if override:
                result.append((
                    op_a.name,
                    op_b.name,
                    override.ratio,
                    override.fuse_speedup
                ))
                continue

            # 从全局规则获取
            type_a = op_a.op_type.value
            type_b = op_b.op_type.value

            if rules.can_fuse(type_a, type_b, op_a.shapes, op_b.shapes):
                ratio = rules.get_ratio(type_a, type_b, op_a.shapes, op_b.shapes)
                speedup = rules.get_speedup(type_a, type_b)
                result.append((op_a.name, op_b.name, ratio, speedup))

        return result

    def get_fusion_groups(self) -> List[Tuple[List[str], str, float]]:
        """
        获取所有可融合的算子组 (支持 3 个及以上算子)

        策略:
        1. 优先匹配预定义的多算子模式 (FusionPattern)
        2. 未匹配部分使用两两规则自动组合

        Returns:
            List of (op_names, pattern_name, speedup)
        """
        from .fusion_rules import get_fusion_rules

        rules = get_fusion_rules()

        # 获取算子类型和 shapes 列表
        op_types = [op.op_type.value for op in self.ops]
        shapes_list = [op.shapes for op in self.ops]

        # 查找所有融合组
        groups = rules.find_fusable_groups(op_types, shapes_list)

        result = []
        for start, end, pattern in groups:
            op_names = [self.ops[i].name for i in range(start, end)]
            speedup = pattern.get_speedup(shapes_list[start:end])
            result.append((op_names, pattern.name, speedup))

        return result

    def _get_override(self, op_a: str, op_b: str) -> Optional[FusePair]:
        """获取覆盖的融合对"""
        return next(
            (p for p in self.override_pairs if p.op_a == op_a and p.op_b == op_b),
            None,
        )

    def get_fuse_ratio(self, op_a: str, op_b: str) -> float:
        """获取两个算子间的效率比"""
        from .fusion_rules import get_fusion_rules

        # 优先检查覆盖
        override = self._get_override(op_a, op_b)
        if override:
            return override.ratio

        # 从全局规则获取
        spec_a = self.get_op(op_a)
        spec_b = self.get_op(op_b)
        if spec_a and spec_b:
            rules = get_fusion_rules()
            return rules.get_ratio(
                spec_a.op_type.value,
                spec_b.op_type.value,
                spec_a.shapes,
                spec_b.shapes
            )

        return 1.0

    def get_fuse_speedup(self, op_a: str, op_b: str) -> Optional[float]:
        """获取两个算子间的融合加速比"""
        from .fusion_rules import get_fusion_rules

        # 优先检查覆盖
        override = self._get_override(op_a, op_b)
        if override:
            return override.fuse_speedup

        # 从全局规则获取
        spec_a = self.get_op(op_a)
        spec_b = self.get_op(op_b)
        if spec_a and spec_b:
            rules = get_fusion_rules()
            return rules.get_speedup(spec_a.op_type.value, spec_b.op_type.value)

        return None

    def to_profiles(self) -> List[OpProfile]:
        """转换为 OpProfile 列表"""
        return [op.to_profile() for op in self.ops]

    def validate(self) -> List[str]:
        """验证 Benchmark 配置"""
        errors = []

        # 检查算子名唯一
        names = [op.name for op in self.ops]
        if len(names) != len(set(names)):
            errors.append("算子名不唯一")

        # 检查覆盖对引用的算子存在
        for pair in self.override_pairs:
            if pair.op_a not in names:
                errors.append(f"融合对引用了不存在的算子: {pair.op_a}")
            if pair.op_b not in names:
                errors.append(f"融合对引用了不存在的算子: {pair.op_b}")

        return errors

    def summary(self) -> str:
        """生成摘要"""
        lines = []
        lines.append(f"Benchmark: {self.name}")
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"Operators ({len(self.ops)}):")
        for op in self.ops:
            unit = op.compute_unit.get_effective_unit(op.op_type, op.shapes)
            lines.append(f"  - {op.name}: {op.op_type.value} [{unit}] {op.shapes}")

        # 显示融合组（多算子模式 + 两两组合）
        groups = self.get_fusion_groups()
        if groups:
            lines.append(f"Fusion Groups ({len(groups)}):")
            for op_names, pattern_name, speedup in groups:
                ops_str = " + ".join(op_names)
                lines.append(f"  - [{pattern_name}] {ops_str}: speedup={speedup:.2f}x")

        # 也显示基础的两两规则
        pairs = self.get_fusable_pairs()
        if pairs and len(groups) != len(pairs):
            lines.append(f"Pairwise Rules ({len(pairs)}):")
            for op_a, op_b, ratio, speedup in pairs:
                speedup_str = f", speedup={speedup:.2f}" if speedup else ""
                lines.append(f"  - {op_a} + {op_b}: ratio={ratio:.2f}{speedup_str}")

        return "\n".join(lines)


# ============================================================
# BenchmarkSuite (Factory 模式)
# ============================================================


class BenchmarkSuite:
    """
    预置 Benchmark 工厂

    使用 Factory 模式创建常用用例。
    融合规则从全局 FusionRules 自动获取，无需显式定义。
    """

    @staticmethod
    def bert_ffn(
        seq_len: int = 512,
        hidden: int = 768,
        intermediate: int = 3072,
    ) -> Benchmark:
        """BERT FFN: Linear -> GELU -> Linear"""
        return (
            Benchmark(
                name=f"bert_ffn_{hidden}_{intermediate}",
                description=f"BERT FFN: {hidden} -> {intermediate} -> {hidden}",
                category="mlp",
                model_source="bert-base",
            )
            .add_op("linear1", "matmul", "cube", M=seq_len, K=hidden, N=intermediate)
            .add_op("gelu", "gelu", "vector", M=seq_len, N=intermediate)
            .add_op("linear2", "matmul", "cube", M=seq_len, K=intermediate, N=hidden)
            # 融合规则从全局 FusionRules 获取
        )

    @staticmethod
    def llama_ffn(
        seq_len: int = 2048,
        hidden: int = 4096,
        intermediate: int = 11008,
    ) -> Benchmark:
        """LLaMA FFN: Gate + Up -> SiLU -> Mul -> Down"""
        return (
            Benchmark(
                name=f"llama_ffn_{hidden}_{intermediate}",
                description=f"LLaMA SwiGLU FFN: {hidden} -> {intermediate} -> {hidden}",
                category="mlp",
                model_source="llama-7b",
            )
            .add_op("gate", "matmul", "cube", M=seq_len, K=hidden, N=intermediate)
            .add_op("up", "matmul", "cube", M=seq_len, K=hidden, N=intermediate)
            .add_op("silu", "silu", "vector", M=seq_len, N=intermediate)
            .add_op("mul", "mul", "vector", M=seq_len, N=intermediate)
            .add_op("down", "matmul", "cube", M=seq_len, K=intermediate, N=hidden)
            # 融合规则从全局 FusionRules 获取
        )

    @staticmethod
    def attention(
        seq_len: int = 2048,
        head_dim: int = 64,
        num_heads: int = 32,
    ) -> Benchmark:
        """Multi-Head Attention: QK -> Softmax -> AV"""
        S, D = seq_len, head_dim
        return (
            Benchmark(
                name=f"attention_s{seq_len}_d{head_dim}",
                description=f"Attention: seq={seq_len}, head_dim={head_dim}",
                category="attention",
            )
            .add_op("qk", "matmul", "cube", M=S, K=D, N=S)
            .add_op("softmax", "softmax", "vector", M=S, N=S)
            .add_op("av", "matmul", "cube", M=S, K=S, N=D)
            # 融合规则从全局 FusionRules 获取
        )

    @staticmethod
    def flash_attention(
        seq_len: int = 2048,
        head_dim: int = 64,
    ) -> Benchmark:
        """Flash Attention (tiled)"""
        S, D = seq_len, head_dim
        block_size = 128

        return (
            Benchmark(
                name=f"flash_attn_s{seq_len}_d{head_dim}",
                description=f"Flash Attention: seq={seq_len}, head_dim={head_dim}, block={block_size}",
                category="attention",
            )
            .add_op("qk_block", "matmul", "cube", M=block_size, K=D, N=block_size)
            .add_op("softmax_block", "softmax", "vector", M=block_size, N=block_size)
            .add_op("av_block", "matmul", "cube", M=block_size, K=block_size, N=D)
            .add_op("rescale", "mul", "vector", M=block_size, N=D)
            # 融合规则从全局 FusionRules 获取
        )

    @staticmethod
    def layernorm(
        batch: int = 512,
        hidden: int = 768,
    ) -> Benchmark:
        """LayerNorm"""
        return Benchmark(
            name=f"layernorm_{batch}_{hidden}",
            description=f"LayerNorm: batch={batch}, hidden={hidden}",
            category="norm",
        ).add_op("layernorm", "layernorm", "vector", M=batch, N=hidden)

    @staticmethod
    def residual_add(
        batch: int = 512,
        hidden: int = 768,
    ) -> Benchmark:
        """Residual Add"""
        return Benchmark(
            name=f"residual_{batch}_{hidden}",
            description=f"Residual Add: batch={batch}, hidden={hidden}",
            category="elementwise",
        ).add_op("add", "add", "vector", M=batch, N=hidden)

    @staticmethod
    def transformer_block(
        seq_len: int = 512,
        hidden: int = 768,
        intermediate: int = 3072,
        num_heads: int = 12,
    ) -> Benchmark:
        """完整 Transformer Block: Attention + FFN"""
        head_dim = hidden // num_heads
        return (
            Benchmark(
                name=f"transformer_{hidden}",
                description=f"Transformer block: hidden={hidden}, heads={num_heads}",
                category="transformer",
            )
            # Attention
            .add_op("q_proj", "matmul", "cube", M=seq_len, K=hidden, N=hidden)
            .add_op("k_proj", "matmul", "cube", M=seq_len, K=hidden, N=hidden)
            .add_op("v_proj", "matmul", "cube", M=seq_len, K=hidden, N=hidden)
            .add_op("qk", "matmul", "cube", M=seq_len, K=head_dim, N=seq_len)
            .add_op("softmax", "softmax", "vector", M=seq_len, N=seq_len)
            .add_op("av", "matmul", "cube", M=seq_len, K=seq_len, N=head_dim)
            .add_op("o_proj", "matmul", "cube", M=seq_len, K=hidden, N=hidden)
            .add_op("add1", "add", "vector", M=seq_len, N=hidden)
            .add_op("ln1", "layernorm", "vector", M=seq_len, N=hidden)
            # FFN
            .add_op("ffn1", "matmul", "cube", M=seq_len, K=hidden, N=intermediate)
            .add_op("gelu", "gelu", "vector", M=seq_len, N=intermediate)
            .add_op("ffn2", "matmul", "cube", M=seq_len, K=intermediate, N=hidden)
            .add_op("add2", "add", "vector", M=seq_len, N=hidden)
            .add_op("ln2", "layernorm", "vector", M=seq_len, N=hidden)
        )

    @classmethod
    def all_benchmarks(cls) -> List[Benchmark]:
        """获取所有预置 benchmark"""
        return [
            cls.bert_ffn(512, 768, 3072),
            cls.bert_ffn(512, 1024, 4096),
            cls.llama_ffn(2048, 4096, 11008),
            cls.attention(512, 64),
            cls.attention(2048, 64),
            cls.flash_attention(2048, 64),
            cls.layernorm(512, 768),
            cls.transformer_block(512, 768, 3072, 12),
        ]

    @classmethod
    def register_custom(
        cls,
        name: str,
        factory_fn: Callable[..., Benchmark],
    ):
        """注册自定义 Benchmark 工厂"""
        setattr(cls, name, staticmethod(factory_fn))
