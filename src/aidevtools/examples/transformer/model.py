"""Transformer 模型组合 (fp32 Golden 实现)"""
import numpy as np

from aidevtools.trace import trace
from aidevtools.examples.transformer.operators import (
    linear, relu, gelu, softmax_safe, layernorm, attention, embedding
)


class TransformerConfig:
    """Transformer 配置"""
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_heads: int = 12,
        intermediate_size: int = 3072,
        num_layers: int = 12,
        max_seq_len: int = 512,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len


def init_weights(config: TransformerConfig) -> dict:
    """初始化模型权重 (随机)"""
    np.random.seed(42)
    h = config.hidden_size
    i = config.intermediate_size
    v = config.vocab_size

    weights = {
        # Embedding
        "embed": np.random.randn(v, h).astype(np.float32) * 0.02,

        # 每层权重
        "layers": []
    }

    for _ in range(config.num_layers):
        layer = {
            # Self-Attention
            "q_proj": {"w": np.random.randn(h, h).astype(np.float32) * 0.02,
                       "b": np.zeros(h, dtype=np.float32)},
            "k_proj": {"w": np.random.randn(h, h).astype(np.float32) * 0.02,
                       "b": np.zeros(h, dtype=np.float32)},
            "v_proj": {"w": np.random.randn(h, h).astype(np.float32) * 0.02,
                       "b": np.zeros(h, dtype=np.float32)},
            "o_proj": {"w": np.random.randn(h, h).astype(np.float32) * 0.02,
                       "b": np.zeros(h, dtype=np.float32)},

            # LayerNorm 1
            "ln1": {"gamma": np.ones(h, dtype=np.float32),
                    "beta": np.zeros(h, dtype=np.float32)},

            # FFN
            "ffn_up": {"w": np.random.randn(h, i).astype(np.float32) * 0.02,
                       "b": np.zeros(i, dtype=np.float32)},
            "ffn_down": {"w": np.random.randn(i, h).astype(np.float32) * 0.02,
                         "b": np.zeros(h, dtype=np.float32)},

            # LayerNorm 2
            "ln2": {"gamma": np.ones(h, dtype=np.float32),
                    "beta": np.zeros(h, dtype=np.float32)},
        }
        weights["layers"].append(layer)

    return weights


@trace(name="self_attention")
def self_attention_block(x: np.ndarray, layer_weights: dict, config: TransformerConfig) -> np.ndarray:
    """
    Multi-Head Self-Attention

    Args:
        x: 输入 (batch, seq_len, hidden_size)
        layer_weights: 层权重
        config: 模型配置
    """
    batch, seq_len, _ = x.shape

    # Q, K, V 投影
    q = linear.__wrapped__(x, layer_weights["q_proj"]["w"], layer_weights["q_proj"]["b"])
    k = linear.__wrapped__(x, layer_weights["k_proj"]["w"], layer_weights["k_proj"]["b"])
    v = linear.__wrapped__(x, layer_weights["v_proj"]["w"], layer_weights["v_proj"]["b"])

    # 拆分多头
    q = q.reshape(batch, seq_len, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)

    # Attention
    d_k = config.head_dim
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = softmax_safe.__wrapped__(scores, axis=-1)
    attn_out = np.matmul(attn_weights, v)

    # 合并多头
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    # 输出投影
    out = linear.__wrapped__(attn_out, layer_weights["o_proj"]["w"], layer_weights["o_proj"]["b"])
    return out


@trace(name="ffn")
def ffn_block(x: np.ndarray, layer_weights: dict) -> np.ndarray:
    """
    Feed-Forward Network

    Args:
        x: 输入 (batch, seq_len, hidden_size)
        layer_weights: 层权重
    """
    # Up projection
    h = linear.__wrapped__(x, layer_weights["ffn_up"]["w"], layer_weights["ffn_up"]["b"])

    # Activation (GELU)
    h = gelu.__wrapped__(h)

    # Down projection
    out = linear.__wrapped__(h, layer_weights["ffn_down"]["w"], layer_weights["ffn_down"]["b"])
    return out


@trace(name="transformer_layer")
def transformer_layer(x: np.ndarray, layer_weights: dict, config: TransformerConfig) -> np.ndarray:
    """
    单层 Transformer

    Args:
        x: 输入 (batch, seq_len, hidden_size)
        layer_weights: 层权重
        config: 模型配置
    """
    # Self-Attention + Residual
    attn_out = self_attention_block.__wrapped__(x, layer_weights, config)
    x = x + attn_out

    # LayerNorm 1
    x = layernorm.__wrapped__(x, layer_weights["ln1"]["gamma"], layer_weights["ln1"]["beta"])

    # FFN + Residual
    ffn_out = ffn_block.__wrapped__(x, layer_weights)
    x = x + ffn_out

    # LayerNorm 2
    x = layernorm.__wrapped__(x, layer_weights["ln2"]["gamma"], layer_weights["ln2"]["beta"])

    return x


@trace(name="transformer")
def transformer_forward(input_ids: np.ndarray, weights: dict, config: TransformerConfig) -> np.ndarray:
    """
    Transformer 前向传播

    Args:
        input_ids: 输入 token ID (batch, seq_len)
        weights: 模型权重
        config: 模型配置
    """
    # Embedding
    x = embedding.__wrapped__(input_ids, weights["embed"])

    # Transformer Layers
    for i, layer_weights in enumerate(weights["layers"]):
        x = transformer_layer.__wrapped__(x, layer_weights, config)

    return x
