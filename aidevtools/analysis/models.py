"""预定义模型 Profile 生成

使用 ops 模块定义模型，自动生成 OpProfile 用于 Paper Analysis。

Usage:
    from aidevtools.analysis import transformer_layer, from_preset

    # 方式 1: 预定义模型函数
    profiles = transformer_layer(batch=4, seq=512, hidden=768, num_heads=12)

    # 方式 2: 从预设配置
    profiles = from_preset("llama-7b", batch=1)

    # 用于分析
    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profiles(profiles)
    result = analyzer.analyze()
"""

from typing import List

import numpy as np


def _run_ops_model(model_fn):
    """运行模型函数，返回自动生成的 profiles"""
    from aidevtools.ops import clear, get_profiles

    clear()
    model_fn()
    return get_profiles()


def transformer_layer(
    batch: int = 4,
    seq: int = 512,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
    activation: str = "gelu",
) -> List:
    """创建 Transformer Layer 的 profiles

    使用 ops 模块定义模型，自动生成 profiles。

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
    from aidevtools.ops._functional import add, attention, gelu, layernorm, linear, silu
    from aidevtools.ops._functional import relu as relu_op

    ffn_h = ffn_hidden or hidden * 4
    head_dim = hidden // num_heads
    np_dtype = np.float16 if dtype == "fp16" else np.float32

    def model():
        # 创建输入张量
        x = np.zeros((batch, seq, hidden), dtype=np_dtype)
        gamma = np.ones((hidden,), dtype=np_dtype)
        beta = np.zeros((hidden,), dtype=np_dtype)
        w = np.zeros((hidden, hidden), dtype=np_dtype)

        # Self-Attention
        layernorm(x, gamma, beta)  # attn_ln
        linear(x, w)  # q_proj
        linear(x, w)  # k_proj
        linear(x, w)  # v_proj

        q = np.zeros((batch, num_heads, seq, head_dim), dtype=np_dtype)
        k = np.zeros((batch, num_heads, seq, head_dim), dtype=np_dtype)
        v = np.zeros((batch, num_heads, seq, head_dim), dtype=np_dtype)
        attention(q, k, v)  # self_attn

        linear(x, w)  # out_proj
        add(x, x)  # attn_residual

        # FFN
        layernorm(x, gamma, beta)  # ffn_ln

        w_ffn1 = np.zeros((hidden, ffn_h), dtype=np_dtype)
        h = np.zeros((batch, seq, ffn_h), dtype=np_dtype)
        linear(x, w_ffn1)  # ffn1

        if activation == "gelu":
            gelu(h)
        elif activation == "relu":
            relu_op(h)
        elif activation == "silu":
            silu(h)

        w_ffn2 = np.zeros((ffn_h, hidden), dtype=np_dtype)
        linear(h, w_ffn2)  # ffn2
        add(x, x)  # ffn_residual

    return _run_ops_model(model)


def llama_layer(
    batch: int = 1,
    seq: int = 2048,
    hidden: int = 4096,
    num_heads: int = 32,
    ffn_hidden: int = None,
    num_kv_heads: int = None,
    dtype: str = "fp16",
) -> List:
    """创建 LLaMA-style Layer 的 profiles

    特点: RMSNorm, SiLU 激活, GQA 支持

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
    from aidevtools.ops._functional import add, attention, linear, mul, rmsnorm, silu

    ffn_h = ffn_hidden or int(hidden * 8 / 3 / 256) * 256
    kv_heads = num_kv_heads or num_heads
    head_dim = hidden // num_heads
    np_dtype = np.float16 if dtype == "fp16" else np.float32

    def model():
        x = np.zeros((batch, seq, hidden), dtype=np_dtype)
        gamma = np.ones((hidden,), dtype=np_dtype)

        # Self-Attention with RMSNorm
        rmsnorm(x, gamma)  # attn_norm

        w_q = np.zeros((hidden, hidden), dtype=np_dtype)
        w_kv = np.zeros((hidden, kv_heads * head_dim), dtype=np_dtype)
        linear(x, w_q)  # q_proj
        linear(x, w_kv)  # k_proj
        linear(x, w_kv)  # v_proj

        q = np.zeros((batch, num_heads, seq, head_dim), dtype=np_dtype)
        k = np.zeros((batch, kv_heads, seq, head_dim), dtype=np_dtype)
        v = np.zeros((batch, kv_heads, seq, head_dim), dtype=np_dtype)
        attention(q, k, v)  # self_attn

        linear(x, w_q)  # o_proj
        add(x, x)  # attn_residual

        # FFN with SiLU (gate + up + down)
        rmsnorm(x, gamma)  # ffn_norm

        w_up = np.zeros((hidden, ffn_h), dtype=np_dtype)
        h = np.zeros((batch, seq, ffn_h), dtype=np_dtype)

        linear(x, w_up)  # gate_proj
        silu(h)  # gate_act
        linear(x, w_up)  # up_proj
        mul(h, h)  # gate_up_mul

        w_down = np.zeros((ffn_h, hidden), dtype=np_dtype)
        linear(h, w_down)  # down_proj
        add(x, x)  # ffn_residual

    return _run_ops_model(model)


def gpt2_layer(
    batch: int = 4,
    seq: int = 1024,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
) -> List:
    """创建 GPT-2 style Layer 的 profiles"""
    return transformer_layer(
        batch=batch,
        seq=seq,
        hidden=hidden,
        num_heads=num_heads,
        ffn_hidden=ffn_hidden or hidden * 4,
        dtype=dtype,
        activation="gelu",
    )


def bert_layer(
    batch: int = 8,
    seq: int = 512,
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
) -> List:
    """创建 BERT style Layer 的 profiles"""
    return transformer_layer(
        batch=batch,
        seq=seq,
        hidden=hidden,
        num_heads=num_heads,
        ffn_hidden=ffn_hidden or hidden * 4,
        dtype=dtype,
        activation="gelu",
    )


def vit_layer(
    batch: int = 32,
    seq: int = 197,  # 14*14 + 1 (cls token)
    hidden: int = 768,
    num_heads: int = 12,
    ffn_hidden: int = None,
    dtype: str = "fp16",
) -> List:
    """创建 ViT (Vision Transformer) Layer 的 profiles"""
    return transformer_layer(
        batch=batch,
        seq=seq,
        hidden=hidden,
        num_heads=num_heads,
        ffn_hidden=ffn_hidden or hidden * 4,
        dtype=dtype,
        activation="gelu",
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
    "llama-70b": {
        "hidden": 8192,
        "num_heads": 64,
        "ffn_hidden": 28672,
        "seq": 4096,
        "num_kv_heads": 8,
    },
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
) -> List:
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

    if model_name.startswith("llama"):
        return llama_layer(batch=batch, dtype=dtype, **config)
    if model_name.startswith("vit"):
        return vit_layer(batch=batch, dtype=dtype, **config)
    return transformer_layer(batch=batch, dtype=dtype, **config)


def list_presets() -> List[str]:
    """列出所有可用的模型预设"""
    return list(MODEL_CONFIGS.keys())
