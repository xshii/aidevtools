"""Transformer 模型组合 (BFP 量化版本)

量化策略:
- matmul 操作: bfp4 (2-bit mantissa, 极端量化)
- 其他操作: bfp8 (4-bit mantissa, 保持精度)

使用全局量化函数 simulate_quantize 模拟量化精度损失。
"""
import numpy as np
import sys
from pathlib import Path
from functools import partial

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aidevtools.trace import trace
from aidevtools.formats.quantize import simulate_quantize
from operators import (
    linear, relu, gelu, softmax_safe, layernorm, attention, embedding
)

# Op 实例直接调用（使用配置的 golden 模式 - cpp 或 python）
# 这里调用 _get_golden 方法，跳过 trace 记录
def _softmax(x, axis=-1):
    return softmax_safe._get_golden(x, axis)

def _gelu(x):
    return gelu._get_golden(x)

def _layernorm(x, gamma, beta, eps=1e-5):
    return layernorm._get_golden(x, gamma, beta, eps)

def _embedding(input_ids, embed_table):
    return embedding._get_golden(input_ids, embed_table)


# ==================== 量化快捷函数 ====================
# 使用全局 simulate_quantize，直接指定 qtype 即可

bfp4 = partial(simulate_quantize, qtype="bfp4")  # matmul 用，极端量化
bfp8 = partial(simulate_quantize, qtype="bfp8")  # 其他操作用


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
    Multi-Head Self-Attention (BFP 量化版本)

    Args:
        x: 输入 (batch, seq_len, hidden_size)
        layer_weights: 层权重
        config: 模型配置

    量化策略:
        - Q/K/V 投影 (matmul): bfp4
        - Attention scores (matmul): bfp4
        - Softmax: bfp8
        - 输出投影 (matmul): bfp4
    """
    batch, seq_len, _ = x.shape

    # Q, K, V 投影 - bfp4 量化输入和权重
    q = np.matmul(bfp4(x), bfp4(layer_weights["q_proj"]["w"])) + layer_weights["q_proj"]["b"]
    k = np.matmul(bfp4(x), bfp4(layer_weights["k_proj"]["w"])) + layer_weights["k_proj"]["b"]
    v = np.matmul(bfp4(x), bfp4(layer_weights["v_proj"]["w"])) + layer_weights["v_proj"]["b"]

    # 拆分多头
    q = q.reshape(batch, seq_len, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, config.num_heads, config.head_dim).transpose(0, 2, 1, 3)

    # Attention scores - bfp4 量化 Q, K
    d_k = config.head_dim
    scores = np.matmul(bfp4(q), bfp4(k.transpose(0, 1, 3, 2))) / np.sqrt(d_k)

    # Softmax - bfp8 量化
    attn_weights = _softmax(bfp8(scores), axis=-1)

    # Attention output - bfp4 量化 weights 和 V
    attn_out = np.matmul(bfp4(attn_weights), bfp4(v))

    # 合并多头
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

    # 输出投影 - bfp4 量化
    out = np.matmul(bfp4(attn_out), bfp4(layer_weights["o_proj"]["w"])) + layer_weights["o_proj"]["b"]
    return out


@trace(name="ffn")
def ffn_block(x: np.ndarray, layer_weights: dict) -> np.ndarray:
    """
    Feed-Forward Network (BFP 量化版本)

    Args:
        x: 输入 (batch, seq_len, hidden_size)
        layer_weights: 层权重

    量化策略:
        - Up/Down projection (matmul): bfp4
        - GELU 激活: bfp8
    """
    # Up projection - bfp4 量化
    h = np.matmul(bfp4(x), bfp4(layer_weights["ffn_up"]["w"])) + layer_weights["ffn_up"]["b"]

    # Activation (GELU) - bfp8 量化
    h = _gelu(bfp8(h))

    # Down projection - bfp4 量化
    out = np.matmul(bfp4(h), bfp4(layer_weights["ffn_down"]["w"])) + layer_weights["ffn_down"]["b"]
    return out


@trace(name="transformer_layer")
def transformer_layer(x: np.ndarray, layer_weights: dict, config: TransformerConfig) -> np.ndarray:
    """
    单层 Transformer (BFP 量化版本)

    Args:
        x: 输入 (batch, seq_len, hidden_size)
        layer_weights: 层权重
        config: 模型配置

    量化策略:
        - LayerNorm: bfp8
        - 残差连接: bfp8
    """
    # Self-Attention + Residual - bfp8 量化
    attn_out = self_attention_block.__wrapped__(x, layer_weights, config)
    x = bfp8(x + attn_out)

    # LayerNorm 1 - bfp8 量化
    x = _layernorm(bfp8(x), layer_weights["ln1"]["gamma"], layer_weights["ln1"]["beta"])

    # FFN + Residual - bfp8 量化
    ffn_out = ffn_block.__wrapped__(x, layer_weights)
    x = bfp8(x + ffn_out)

    # LayerNorm 2 - bfp8 量化
    x = _layernorm(bfp8(x), layer_weights["ln2"]["gamma"], layer_weights["ln2"]["beta"])

    return x


@trace(name="transformer")
def transformer_forward(input_ids: np.ndarray, weights: dict, config: TransformerConfig) -> np.ndarray:
    """
    Transformer 前向传播 (BFP 量化版本)

    Args:
        input_ids: 输入 token ID (batch, seq_len)
        weights: 模型权重
        config: 模型配置

    量化策略:
        - Embedding 输出: bfp8
    """
    # Embedding - bfp8 量化输出
    x = bfp8(_embedding(input_ids, weights["embed"]))

    # Transformer Layers
    for i, layer_weights in enumerate(weights["layers"]):
        x = transformer_layer.__wrapped__(x, layer_weights, config)

    return x
