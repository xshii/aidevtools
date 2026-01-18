"""神经网络算子"""
import numpy as np

from aidevtools.ops.base import Op


class Linear(Op):
    """线性层: y = x @ W + b"""
    name = "linear"

    def reference(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
        y = np.matmul(x, weight)
        if bias is not None:
            y = y + bias
        return y


class ReLU(Op):
    """ReLU: y = max(0, x)"""
    name = "relu"

    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class GELU(Op):
    """GELU 近似"""
    name = "gelu"

    def reference(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class Sigmoid(Op):
    """Sigmoid: y = 1 / (1 + exp(-x))"""
    name = "sigmoid"

    def reference(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class Tanh(Op):
    """Tanh"""
    name = "tanh"

    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


class Softmax(Op):
    """安全 Softmax（防溢出）"""
    name = "softmax"

    def reference(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


class LayerNorm(Op):
    """Layer Normalization"""
    name = "layernorm"

    def reference(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


class BatchNorm(Op):
    """Batch Normalization"""
    name = "batchnorm"

    def reference(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  mean: np.ndarray = None, var: np.ndarray = None, eps: float = 1e-5) -> np.ndarray:
        if mean is None:
            mean = np.mean(x, axis=0, keepdims=True)
        if var is None:
            var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


class Embedding(Op):
    """Embedding 查表"""
    name = "embedding"

    def reference(self, input_ids: np.ndarray, embed_table: np.ndarray) -> np.ndarray:
        return embed_table[input_ids]


class MatMul(Op):
    """矩阵乘法"""
    name = "matmul"

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)


class Add(Op):
    """加法"""
    name = "add"

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b


class Mul(Op):
    """乘法"""
    name = "mul"

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b


class Div(Op):
    """除法"""
    name = "div"

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a / b


class Attention(Op):
    """Scaled Dot-Product Attention"""
    name = "attention"

    def reference(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        d_k = q.shape[-1]
        scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask * (-1e9)
        # softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        return np.matmul(attn_weights, v)


# 实例化算子（方便直接调用）
linear = Linear()
relu = ReLU()
gelu = GELU()
sigmoid = Sigmoid()
tanh = Tanh()
softmax = Softmax()
layernorm = LayerNorm()
batchnorm = BatchNorm()
embedding = Embedding()
matmul = MatMul()
add = Add()
mul = Mul()
div = Div()
attention = Attention()
