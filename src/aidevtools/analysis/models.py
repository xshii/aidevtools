"""Model Builder - 类似 ops 的简洁模型定义 API

提供两种使用方式:

1. ModelBuilder 链式调用:
    builder = ModelBuilder(batch=4, seq=512, hidden=768, dtype="fp16")
    builder.layernorm("attn_ln")
    builder.linear("q_proj", out_features=768)
    builder.linear("k_proj", out_features=768)
    builder.linear("v_proj", out_features=768)
    builder.attention("self_attn", num_heads=12)
    builder.linear("out_proj", out_features=768)
    builder.add("attn_residual")
    profiles = builder.build()

2. 预定义模型函数:
    profiles = transformer_layer(
        batch=4, seq=512, hidden=768,
        num_heads=12, ffn_hidden=3072
    )
"""

from typing import List, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np

from .profile import (
    OpProfile, dtype_bytes, MatMulDtypeConfig,
    profile_matmul, profile_layernorm, profile_softmax,
    profile_attention, profile_gelu, profile_add, profile_transpose,
)


@dataclass
class TensorShape:
    """张量形状追踪"""
    shape: Tuple[int, ...]
    dtype: str = "fp16"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return int(self.size * dtype_bytes(self.dtype))

    def __repr__(self):
        return f"TensorShape({self.shape}, {self.dtype})"


class ModelBuilder:
    """模型构建器 - 类似 ops 的链式 API

    Example:
        builder = ModelBuilder(batch=4, seq=512, hidden=768)
        builder.layernorm("attn_ln")
        builder.linear("q_proj", out_features=768)
        profiles = builder.build()
    """

    def __init__(
        self,
        batch: int = 1,
        seq: int = 512,
        hidden: int = 768,
        dtype: str = "fp16",
        num_heads: int = 12,
    ):
        """
        Args:
            batch: batch size
            seq: sequence length
            hidden: hidden dimension
            dtype: 数据类型
            num_heads: attention heads (用于 attention 算子)
        """
        self.batch = batch
        self.seq = seq
        self.hidden = hidden
        self.dtype = dtype
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads

        # 当前张量形状 (用于形状推导)
        self._current_shape = TensorShape((batch, seq, hidden), dtype)

        # 收集的 profiles
        self._profiles: List[OpProfile] = []

        # 命名计数器
        self._name_counters = {}

    def _auto_name(self, op_type: str, name: Optional[str] = None) -> str:
        """自动生成算子名称"""
        if name:
            return name
        count = self._name_counters.get(op_type, 0)
        self._name_counters[op_type] = count + 1
        return f"{op_type}_{count}"

    def _make_array(self, shape: Tuple[int, ...]) -> np.ndarray:
        """创建占位数组 (用于 profile 函数)"""
        return np.zeros(shape, dtype=np.float16)

    # ============================================================
    # 算子方法
    # ============================================================

    def layernorm(self, name: str = None) -> "ModelBuilder":
        """LayerNorm: [B, S, H] -> [B, S, H]"""
        name = self._auto_name("layernorm", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        gamma = self._make_array((shape[-1],))
        beta = self._make_array((shape[-1],))

        profile = profile_layernorm(x, gamma, beta)
        profile.name = name
        self._profiles.append(profile)

        return self

    def rmsnorm(self, name: str = None) -> "ModelBuilder":
        """RMSNorm: [B, S, H] -> [B, S, H] (与 LayerNorm 类似)"""
        name = self._auto_name("rmsnorm", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        gamma = self._make_array((shape[-1],))
        beta = self._make_array((shape[-1],))

        profile = profile_layernorm(x, gamma, beta)
        profile.name = name
        profile.op_type = "rmsnorm"
        self._profiles.append(profile)

        return self

    def linear(self, name: str = None, out_features: int = None,
               in_features: int = None) -> "ModelBuilder":
        """Linear: [B, S, in] @ [in, out] -> [B, S, out]"""
        name = self._auto_name("linear", name)
        shape = self._current_shape.shape

        in_f = in_features or shape[-1]
        out_f = out_features or shape[-1]

        # 构造输入和权重
        input_shape = shape[:-1] + (in_f,)
        weight_shape = (in_f, out_f)

        a = self._make_array(input_shape)
        b = self._make_array(weight_shape)

        profile = profile_matmul(a, b)
        profile.name = name
        profile.op_type = "linear"
        self._profiles.append(profile)

        # 更新当前形状
        self._current_shape = TensorShape(shape[:-1] + (out_f,), self.dtype)

        return self

    def matmul(self, name: str = None, weight_shape: Tuple[int, int] = None) -> "ModelBuilder":
        """MatMul: [B, S, K] @ [K, N] -> [B, S, N]"""
        name = self._auto_name("matmul", name)
        shape = self._current_shape.shape

        if weight_shape is None:
            weight_shape = (shape[-1], shape[-1])

        a = self._make_array(shape)
        b = self._make_array(weight_shape)

        profile = profile_matmul(a, b)
        profile.name = name
        self._profiles.append(profile)

        # 更新当前形状
        self._current_shape = TensorShape(shape[:-1] + (weight_shape[-1],), self.dtype)

        return self

    def attention(self, name: str = None, num_heads: int = None,
                  seq_kv: int = None) -> "ModelBuilder":
        """Multi-Head Attention

        输入: [B, S, H] (假设 Q/K/V 已投影)
        输出: [B, S, H]
        """
        name = self._auto_name("attention", name)
        heads = num_heads or self.num_heads
        head_dim = self.hidden // heads
        seq_q = self.seq
        seq_k = seq_kv or seq_q

        # Q, K, V shape: [B, heads, seq, head_dim]
        q = self._make_array((self.batch, heads, seq_q, head_dim))
        k = self._make_array((self.batch, heads, seq_k, head_dim))
        v = self._make_array((self.batch, heads, seq_k, head_dim))

        profile = profile_attention(q, k, v)
        profile.name = name
        self._profiles.append(profile)

        # 输出形状: [B, S, H]
        self._current_shape = TensorShape((self.batch, seq_q, self.hidden), self.dtype)

        return self

    def softmax(self, name: str = None, axis: int = -1) -> "ModelBuilder":
        """Softmax"""
        name = self._auto_name("softmax", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        profile = profile_softmax(x, axis)
        profile.name = name
        self._profiles.append(profile)

        return self

    def gelu(self, name: str = None) -> "ModelBuilder":
        """GELU 激活"""
        name = self._auto_name("gelu", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        profile = profile_gelu(x)
        profile.name = name
        self._profiles.append(profile)

        return self

    def relu(self, name: str = None) -> "ModelBuilder":
        """ReLU 激活 (FLOPs 与 GELU 类似但更少)"""
        name = self._auto_name("relu", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        profile = profile_gelu(x)  # 复用 gelu profile
        profile.name = name
        profile.op_type = "relu"
        profile.flops = int(x.size)  # ReLU 只有 1 op/element
        self._profiles.append(profile)

        return self

    def silu(self, name: str = None) -> "ModelBuilder":
        """SiLU/Swish 激活"""
        name = self._auto_name("silu", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        profile = profile_gelu(x)
        profile.name = name
        profile.op_type = "silu"
        profile.flops = int(x.size * 5)  # sigmoid + mul
        self._profiles.append(profile)

        return self

    def add(self, name: str = None) -> "ModelBuilder":
        """残差连接 Add"""
        name = self._auto_name("add", name)
        shape = self._current_shape.shape

        a = self._make_array(shape)
        b = self._make_array(shape)

        profile = profile_add(a, b)
        profile.name = name
        self._profiles.append(profile)

        return self

    def mul(self, name: str = None) -> "ModelBuilder":
        """Element-wise Mul"""
        name = self._auto_name("mul", name)
        shape = self._current_shape.shape

        a = self._make_array(shape)
        b = self._make_array(shape)

        profile = profile_add(a, b)  # 复用 add profile
        profile.name = name
        profile.op_type = "mul"
        self._profiles.append(profile)

        return self

    def transpose(self, name: str = None, axes: Tuple[int, ...] = None) -> "ModelBuilder":
        """Transpose"""
        name = self._auto_name("transpose", name)
        shape = self._current_shape.shape

        x = self._make_array(shape)
        profile = profile_transpose(x, axes)
        profile.name = name
        self._profiles.append(profile)

        # 更新形状
        if axes:
            new_shape = tuple(shape[i] for i in axes)
            self._current_shape = TensorShape(new_shape, self.dtype)

        return self

    def embedding(self, name: str = None, vocab_size: int = 32000) -> "ModelBuilder":
        """Embedding lookup"""
        name = self._auto_name("embedding", name)

        # Embedding: [B, S] -> [B, S, H]
        # 访存为主，计算几乎为 0
        db = dtype_bytes(self.dtype)
        output_bytes = self.batch * self.seq * self.hidden * db
        weight_bytes = vocab_size * self.hidden * db

        profile = OpProfile(
            name=name,
            op_type="embedding",
            shapes={"batch": self.batch, "seq": self.seq, "vocab": vocab_size, "hidden": self.hidden},
            dtype=self.dtype,
            flops=0,
            compute_unit="vector",
            input_bytes=self.batch * self.seq * 4,  # int32 indices
            weight_bytes=int(weight_bytes),
            output_bytes=int(output_bytes),
            memory_pattern="random",  # embedding 是随机访问
        )
        self._profiles.append(profile)

        self._current_shape = TensorShape((self.batch, self.seq, self.hidden), self.dtype)

        return self

    def set_shape(self, shape: Tuple[int, ...]) -> "ModelBuilder":
        """手动设置当前形状 (用于分支合并等场景)"""
        self._current_shape = TensorShape(shape, self.dtype)
        return self

    def reset_shape(self) -> "ModelBuilder":
        """重置为初始形状 [B, S, H]"""
        self._current_shape = TensorShape((self.batch, self.seq, self.hidden), self.dtype)
        return self

    # ============================================================
    # 复合模块
    # ============================================================

    def qkv_proj(self, name_prefix: str = "") -> "ModelBuilder":
        """Q/K/V 投影 (3 个 Linear)"""
        prefix = f"{name_prefix}_" if name_prefix else ""
        self.linear(f"{prefix}q_proj", out_features=self.hidden)
        self.reset_shape()
        self.linear(f"{prefix}k_proj", out_features=self.hidden)
        self.reset_shape()
        self.linear(f"{prefix}v_proj", out_features=self.hidden)
        return self

    def ffn(self, name_prefix: str = "", ffn_hidden: int = None,
            activation: str = "gelu") -> "ModelBuilder":
        """Feed-Forward Network

        FFN = Linear -> Activation -> Linear
        """
        prefix = f"{name_prefix}_" if name_prefix else ""
        ffn_h = ffn_hidden or self.hidden * 4

        self.linear(f"{prefix}ffn1", out_features=ffn_h)

        if activation == "gelu":
            self.gelu(f"{prefix}ffn_act")
        elif activation == "relu":
            self.relu(f"{prefix}ffn_act")
        elif activation == "silu":
            self.silu(f"{prefix}ffn_act")

        self.linear(f"{prefix}ffn2", out_features=self.hidden)

        return self

    def self_attention_block(self, name_prefix: str = "",
                             with_residual: bool = True) -> "ModelBuilder":
        """Self-Attention Block

        LN -> Q/K/V Proj -> Attention -> Out Proj -> Add
        """
        prefix = f"{name_prefix}_" if name_prefix else ""

        self.layernorm(f"{prefix}attn_ln")
        self.qkv_proj(prefix + "attn")
        self.attention(f"{prefix}self_attn")
        self.linear(f"{prefix}out_proj", out_features=self.hidden)

        if with_residual:
            self.add(f"{prefix}attn_residual")

        return self

    def ffn_block(self, name_prefix: str = "", ffn_hidden: int = None,
                  activation: str = "gelu", with_residual: bool = True) -> "ModelBuilder":
        """FFN Block

        LN -> FFN1 -> Act -> FFN2 -> Add
        """
        prefix = f"{name_prefix}_" if name_prefix else ""

        self.layernorm(f"{prefix}ffn_ln")
        self.ffn(prefix, ffn_hidden=ffn_hidden, activation=activation)

        if with_residual:
            self.add(f"{prefix}ffn_residual")

        return self

    def transformer_layer(self, layer_idx: int = 0, ffn_hidden: int = None,
                          activation: str = "gelu") -> "ModelBuilder":
        """完整 Transformer Layer

        Self-Attention Block + FFN Block
        """
        prefix = f"layer{layer_idx}"
        self.self_attention_block(prefix)
        self.ffn_block(prefix, ffn_hidden=ffn_hidden, activation=activation)
        return self

    # ============================================================
    # 构建
    # ============================================================

    def build(self) -> List[OpProfile]:
        """返回收集的 profiles"""
        return self._profiles

    def clear(self) -> "ModelBuilder":
        """清空已收集的 profiles"""
        self._profiles = []
        self._name_counters = {}
        self._current_shape = TensorShape((self.batch, self.seq, self.hidden), self.dtype)
        return self

    def __len__(self) -> int:
        return len(self._profiles)

    def __repr__(self):
        return f"ModelBuilder(batch={self.batch}, seq={self.seq}, hidden={self.hidden}, ops={len(self._profiles)})"


# ============================================================
# 预定义模型函数
# ============================================================

def transformer_layer(
    batch: int = 4,
    seq: int = 512,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
    activation: str = "gelu",
) -> List[OpProfile]:
    """创建 Transformer Layer 的 profiles

    Args:
        batch: batch size
        seq: sequence length
        hidden: hidden dimension
        num_heads: attention heads
        ffn_hidden: FFN hidden dimension (default: 4 * hidden)
        dtype: 数据类型
        activation: FFN 激活函数 ("gelu", "relu", "silu")

    Returns:
        List[OpProfile]

    Example:
        profiles = transformer_layer(batch=4, seq=512, hidden=768, num_heads=12)
    """
    ffn_h = ffn_hidden or hidden * 4

    builder = ModelBuilder(batch=batch, seq=seq, hidden=hidden,
                           dtype=dtype, num_heads=num_heads)

    # Self-Attention
    builder.layernorm("attn_ln")
    builder.linear("q_proj", out_features=hidden)
    builder.reset_shape()
    builder.linear("k_proj", out_features=hidden)
    builder.reset_shape()
    builder.linear("v_proj", out_features=hidden)
    builder.attention("self_attn", num_heads=num_heads)
    builder.linear("out_proj", out_features=hidden)
    builder.add("attn_residual")

    # FFN
    builder.layernorm("ffn_ln")
    builder.linear("ffn1", out_features=ffn_h)

    if activation == "gelu":
        builder.gelu("ffn_gelu")
    elif activation == "relu":
        builder.relu("ffn_relu")
    elif activation == "silu":
        builder.silu("ffn_silu")

    builder.linear("ffn2", out_features=hidden)
    builder.add("ffn_residual")

    return builder.build()


def llama_layer(
    batch: int = 1,
    seq: int = 2048,
    hidden: int = 4096,
    num_heads: int = 32,
    ffn_hidden: int = None,
    num_kv_heads: int = None,
    dtype: str = "fp16",
) -> List[OpProfile]:
    """创建 LLaMA-style Layer 的 profiles

    特点:
    - RMSNorm 而非 LayerNorm
    - SiLU 激活
    - GQA (Grouped Query Attention) 支持

    Args:
        batch: batch size
        seq: sequence length
        hidden: hidden dimension
        num_heads: attention heads
        ffn_hidden: FFN hidden dimension (default: 8/3 * hidden, rounded)
        num_kv_heads: KV heads for GQA (default: same as num_heads)
        dtype: 数据类型

    Returns:
        List[OpProfile]
    """
    ffn_h = ffn_hidden or int(hidden * 8 / 3 / 256) * 256  # LLaMA style
    kv_heads = num_kv_heads or num_heads
    head_dim = hidden // num_heads

    builder = ModelBuilder(batch=batch, seq=seq, hidden=hidden,
                           dtype=dtype, num_heads=num_heads)

    # Self-Attention with RMSNorm
    builder.rmsnorm("attn_norm")
    builder.linear("q_proj", out_features=hidden)
    builder.reset_shape()
    builder.linear("k_proj", out_features=kv_heads * head_dim)
    builder.reset_shape()
    builder.linear("v_proj", out_features=kv_heads * head_dim)
    builder.attention("self_attn", num_heads=num_heads)
    builder.linear("o_proj", out_features=hidden)
    builder.add("attn_residual")

    # FFN with SiLU (gate + up + down)
    builder.rmsnorm("ffn_norm")
    builder.linear("gate_proj", out_features=ffn_h)
    builder.silu("gate_act")
    builder.reset_shape().linear("up_proj", out_features=ffn_h)
    builder.mul("gate_up_mul")  # gate * up
    builder.linear("down_proj", out_features=hidden)
    builder.add("ffn_residual")

    return builder.build()


def gpt2_layer(
    batch: int = 4,
    seq: int = 1024,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
) -> List[OpProfile]:
    """创建 GPT-2 style Layer 的 profiles

    特点:
    - Pre-LayerNorm
    - GELU 激活
    - 标准 4x FFN

    Args:
        batch: batch size
        seq: sequence length
        hidden: hidden dimension
        num_heads: attention heads
        ffn_hidden: FFN hidden dimension (default: 4 * hidden)
        dtype: 数据类型

    Returns:
        List[OpProfile]
    """
    return transformer_layer(
        batch=batch, seq=seq, hidden=hidden,
        num_heads=num_heads, ffn_hidden=ffn_hidden or hidden * 4,
        dtype=dtype, activation="gelu"
    )


def bert_layer(
    batch: int = 8,
    seq: int = 512,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
) -> List[OpProfile]:
    """创建 BERT style Layer 的 profiles

    与 GPT-2 类似，但通常 batch size 更大
    """
    return transformer_layer(
        batch=batch, seq=seq, hidden=hidden,
        num_heads=num_heads, ffn_hidden=ffn_hidden or hidden * 4,
        dtype=dtype, activation="gelu"
    )


def vit_layer(
    batch: int = 32,
    seq: int = 197,  # 14*14 + 1 (cls token)
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
) -> List[OpProfile]:
    """创建 ViT (Vision Transformer) Layer 的 profiles

    Args:
        batch: batch size
        seq: number of patches + 1 (default: 14*14 + 1 = 197 for 224x224 image)
        hidden: hidden dimension
        num_heads: attention heads
        ffn_hidden: FFN hidden dimension (default: 4 * hidden)
        dtype: 数据类型

    Returns:
        List[OpProfile]
    """
    return transformer_layer(
        batch=batch, seq=seq, hidden=hidden,
        num_heads=num_heads, ffn_hidden=ffn_hidden or hidden * 4,
        dtype=dtype, activation="gelu"
    )


# 模型配置预设
MODEL_CONFIGS = {
    # GPT-2 系列
    "gpt2": {"hidden": 768, "num_heads": 12, "ffn_hidden": 3072, "seq": 1024},
    "gpt2-medium": {"hidden": 1024, "num_heads": 16, "ffn_hidden": 4096, "seq": 1024},
    "gpt2-large": {"hidden": 1280, "num_heads": 20, "ffn_hidden": 5120, "seq": 1024},
    "gpt2-xl": {"hidden": 1600, "num_heads": 25, "ffn_hidden": 6400, "seq": 1024},

    # BERT 系列
    "bert-base": {"hidden": 768, "num_heads": 12, "ffn_hidden": 3072, "seq": 512},
    "bert-large": {"hidden": 1024, "num_heads": 16, "ffn_hidden": 4096, "seq": 512},

    # LLaMA 系列
    "llama-7b": {"hidden": 4096, "num_heads": 32, "ffn_hidden": 11008, "seq": 2048},
    "llama-13b": {"hidden": 5120, "num_heads": 40, "ffn_hidden": 13824, "seq": 2048},
    "llama-70b": {"hidden": 8192, "num_heads": 64, "ffn_hidden": 28672, "seq": 4096, "num_kv_heads": 8},

    # ViT 系列
    "vit-base": {"hidden": 768, "num_heads": 12, "ffn_hidden": 3072, "seq": 197},
    "vit-large": {"hidden": 1024, "num_heads": 16, "ffn_hidden": 4096, "seq": 197},
    "vit-huge": {"hidden": 1280, "num_heads": 16, "ffn_hidden": 5120, "seq": 197},
}


def from_preset(
    model_name: str,
    batch: int = 1,
    seq: int = None,
    dtype: str = "fp16",
) -> List[OpProfile]:
    """从预设配置创建模型 profiles

    Args:
        model_name: 模型名称 (如 "gpt2", "llama-7b", "bert-base")
        batch: batch size
        seq: sequence length (覆盖预设值)
        dtype: 数据类型

    Returns:
        List[OpProfile]

    Example:
        profiles = from_preset("llama-7b", batch=1)
    """
    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    config = MODEL_CONFIGS[model_name].copy()
    if seq is not None:
        config["seq"] = seq

    # 根据模型类型选择构建函数
    if model_name.startswith("llama"):
        return llama_layer(batch=batch, dtype=dtype, **config)
    elif model_name.startswith("vit"):
        return vit_layer(batch=batch, dtype=dtype, **config)
    else:
        return transformer_layer(batch=batch, dtype=dtype, **config)


def list_presets() -> List[str]:
    """列出所有可用的模型预设"""
    return list(MODEL_CONFIGS.keys())
