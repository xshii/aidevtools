"""Transformer 常用算子 (fp32 Golden 实现)"""
import numpy as np

from aidevtools.trace import trace


@trace
def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
    """
    线性层: y = x @ W + b

    Args:
        x: 输入 (..., in_features)
        weight: 权重 (in_features, out_features)
        bias: 偏置 (out_features,)
    """
    y = np.matmul(x, weight)
    if bias is not None:
        y = y + bias
    return y


@trace
def relu(x: np.ndarray) -> np.ndarray:
    """ReLU: y = max(0, x)"""
    return np.maximum(0, x)


@trace
def gelu(x: np.ndarray) -> np.ndarray:
    """GELU 近似: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


@trace
def softmax_safe(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    安全 Softmax (防止溢出)

    y = exp(x - max(x)) / sum(exp(x - max(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


@trace
def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Layer Normalization

    Args:
        x: 输入 (..., hidden_size)
        gamma: 缩放参数 (hidden_size,)
        beta: 偏移参数 (hidden_size,)
        eps: 防止除零
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


@trace
def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """
    Scaled Dot-Product Attention

    Args:
        q: Query (batch, seq_len, d_k)
        k: Key (batch, seq_len, d_k)
        v: Value (batch, seq_len, d_v)
        mask: 注意力掩码 (可选)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask * (-1e9)

    attn_weights = softmax_safe.__wrapped__(scores, axis=-1)  # 调用原函数避免重复 trace
    return np.matmul(attn_weights, v)


@trace
def embedding(input_ids: np.ndarray, embed_table: np.ndarray) -> np.ndarray:
    """
    Embedding 查表

    Args:
        input_ids: 输入 token ID (batch, seq_len)
        embed_table: 嵌入表 (vocab_size, hidden_size)
    """
    return embed_table[input_ids]
