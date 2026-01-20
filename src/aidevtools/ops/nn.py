"""神经网络算子

每个算子包含三种实现：
- golden_python: Python Golden 实现（fp32，精确实现）
- cpu_golden: C++ Golden 实现（通过 subprocess 调用）
- reference: 高精度参考实现（fp64，用于 fuzzy 比对）

使用 @register_op 装饰器自动注册算子元信息。
"""
import numpy as np

from aidevtools.ops.base import Op
from aidevtools.ops.registry import register_op
from aidevtools.ops.cpu_golden import (
    run_cpu_golden,
    get_cpu_golden_dtype,
    get_matmul_dtypes,
)


@register_op(
    inputs=["x", "weight"],
    optional=["bias"],
    description="线性变换 y = x @ weight + bias",
)
class Linear(Op):
    """线性层: y = x @ W + b"""
    name = "linear"

    def golden_python(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
        y = np.matmul(x, weight)
        if bias is not None:
            y = y + bias
        return y.astype(np.float32)

    def reference(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
        # 使用 fp64 计算高精度参考
        x64 = x.astype(np.float64)
        w64 = weight.astype(np.float64)
        y = np.matmul(x64, w64)
        if bias is not None:
            y = y + bias.astype(np.float64)
        return y.astype(np.float32)


@register_op(
    inputs=["x"],
    description="ReLU 激活 y = max(0, x)",
)
class ReLU(Op):
    """ReLU: y = max(0, x)"""
    name = "relu"

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x).astype(np.float32)

    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["x"],
    description="GELU 激活 (近似)",
)
class GELU(Op):
    """GELU 近似"""
    name = "gelu"

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)

    def reference(self, x: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64)
        return (0.5 * x64 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x64 + 0.044715 * x64 ** 3)))).astype(np.float32)


@register_op(
    inputs=["x"],
    description="Sigmoid 激活 y = 1 / (1 + exp(-x))",
)
class Sigmoid(Op):
    """Sigmoid: y = 1 / (1 + exp(-x))"""
    name = "sigmoid"

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-x))).astype(np.float32)

    def reference(self, x: np.ndarray) -> np.ndarray:
        x64 = x.astype(np.float64)
        return (1 / (1 + np.exp(-x64))).astype(np.float32)


@register_op(
    inputs=["x"],
    description="Tanh 激活",
)
class Tanh(Op):
    """Tanh"""
    name = "tanh"

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x).astype(np.float32)

    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["x"],
    optional=["axis"],
    description="Softmax 激活 (防溢出)",
    has_cpp_golden=True,
)
class Softmax(Op):
    """安全 Softmax（防溢出）"""
    name = "softmax"

    def golden_python(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return (x_exp / np.sum(x_exp, axis=axis, keepdims=True)).astype(np.float32)

    def cpu_golden(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """C++ Golden 实现"""
        dtype = get_cpu_golden_dtype()
        x = np.asarray(x, dtype=np.float32)
        original_shape = x.shape

        # flatten 到 2D: [batch, seq]
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

        batch, seq = x.shape

        y = run_cpu_golden(
            op_name="softmax",
            cmd_args=["softmax", dtype, "@input.bin", "@output", str(batch), str(seq)],
            inputs={"input.bin": (x, dtype)},
            output_name="output.bin",
            output_dtype=dtype,
            output_size=batch * seq,
            output_shape=(batch, seq),
        )

        return y.reshape(original_shape)

    def reference(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x64 = x.astype(np.float64)
        x_max = np.max(x64, axis=axis, keepdims=True)
        x_exp = np.exp(x64 - x_max)
        return (x_exp / np.sum(x_exp, axis=axis, keepdims=True)).astype(np.float32)


@register_op(
    inputs=["x", "gamma", "beta"],
    optional=["eps"],
    description="Layer Normalization",
    has_cpp_golden=True,
)
class LayerNorm(Op):
    """Layer Normalization"""
    name = "layernorm"

    def golden_python(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return (gamma * x_norm + beta).astype(np.float32)

    def cpu_golden(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """C++ Golden 实现"""
        dtype = get_cpu_golden_dtype()
        x = np.asarray(x, dtype=np.float32)
        gamma = np.asarray(gamma, dtype=np.float32)
        beta = np.asarray(beta, dtype=np.float32)

        original_shape = x.shape
        hidden = x.shape[-1]

        # flatten 到 2D: [batch, hidden]
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(-1, hidden)

        batch = x.shape[0]

        assert gamma.shape == (hidden,), f"gamma shape mismatch: {gamma.shape} vs ({hidden},)"
        assert beta.shape == (hidden,), f"beta shape mismatch: {beta.shape} vs ({hidden},)"

        y = run_cpu_golden(
            op_name="layernorm",
            cmd_args=["layernorm", dtype, "@x.bin", "@gamma.bin", "@beta.bin", "@output", str(batch), str(hidden)],
            inputs={
                "x.bin": (x, dtype),
                "gamma.bin": (gamma, dtype),
                "beta.bin": (beta, dtype),
            },
            output_name="y.bin",
            output_dtype=dtype,
            output_size=batch * hidden,
            output_shape=(batch, hidden),
        )

        return y.reshape(original_shape)

    def reference(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        x64 = x.astype(np.float64)
        mean = np.mean(x64, axis=-1, keepdims=True)
        var = np.var(x64, axis=-1, keepdims=True)
        x_norm = (x64 - mean) / np.sqrt(var + eps)
        return (gamma.astype(np.float64) * x_norm + beta.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["x", "gamma", "beta"],
    optional=["mean", "var", "eps"],
    description="Batch Normalization",
)
class BatchNorm(Op):
    """Batch Normalization"""
    name = "batchnorm"

    def golden_python(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      mean: np.ndarray = None, var: np.ndarray = None, eps: float = 1e-5) -> np.ndarray:
        if mean is None:
            mean = np.mean(x, axis=0, keepdims=True)
        if var is None:
            var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return (gamma * x_norm + beta).astype(np.float32)

    def reference(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  mean: np.ndarray = None, var: np.ndarray = None, eps: float = 1e-5) -> np.ndarray:
        x64 = x.astype(np.float64)
        if mean is None:
            mean = np.mean(x64, axis=0, keepdims=True)
        else:
            mean = mean.astype(np.float64)
        if var is None:
            var = np.var(x64, axis=0, keepdims=True)
        else:
            var = var.astype(np.float64)
        x_norm = (x64 - mean) / np.sqrt(var + eps)
        return (gamma.astype(np.float64) * x_norm + beta.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["input_ids", "embed_table"],
    description="Embedding 查表",
)
class Embedding(Op):
    """Embedding 查表"""
    name = "embedding"

    def golden_python(self, input_ids: np.ndarray, embed_table: np.ndarray) -> np.ndarray:
        return embed_table[input_ids].astype(np.float32)

    def reference(self, input_ids: np.ndarray, embed_table: np.ndarray) -> np.ndarray:
        # Embedding 是查表操作，fp64 无额外精度收益
        return embed_table[input_ids].astype(np.float32)


@register_op(
    inputs=["a", "b"],
    description="矩阵乘法 c = a @ b",
    has_cpp_golden=True,
)
class MatMul(Op):
    """矩阵乘法"""
    name = "matmul"

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b).astype(np.float32)

    def cpu_golden(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """C++ Golden 实现 (支持 batch 和混合精度)"""
        dtype_a, dtype_b, dtype_out = get_matmul_dtypes()
        is_mixed = (dtype_a != dtype_b) or (dtype_a != dtype_out)

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        # 处理 batch 维度
        a_batch_shape = a.shape[:-2] if a.ndim > 2 else ()

        # 获取 M, K, N
        M, K = a.shape[-2:]
        if b.ndim == 1:
            K2, N = b.shape[0], 1
            b = b.reshape(K2, N)
        else:
            K2, N = b.shape[-2:]

        assert K == K2, f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"

        # 处理 2D 情况 (快速路径)
        if a.ndim == 2 and b.ndim == 2:
            return self._matmul_2d(a, b, M, K, N, dtype_a, dtype_b, dtype_out, is_mixed)

        # 处理 batch: flatten batch dims, 循环调用 2D matmul
        if a.ndim > 2:
            batch_size = int(np.prod(a_batch_shape))
            a_flat = a.reshape(batch_size, M, K)
        else:
            batch_size = 1
            a_flat = a.reshape(1, M, K)

        # b 可能没有 batch (广播)
        if b.ndim == 2:
            b_batched = False
            b_flat = b.reshape(1, K, N)
        else:
            b_batched = True
            b_flat = b.reshape(batch_size, K, N)

        # 逐 batch 计算
        c_flat = np.zeros((batch_size, M, N), dtype=np.float32)
        for i in range(batch_size):
            a_i = a_flat[i]
            b_i = b_flat[i] if b_batched else b_flat[0]
            c_flat[i] = self._matmul_2d(a_i, b_i, M, K, N, dtype_a, dtype_b, dtype_out, is_mixed)

        # 恢复 batch shape
        output_shape = a_batch_shape + (M, N)
        return c_flat.reshape(output_shape)

    def _matmul_2d(self, a, b, M, K, N, dtype_a, dtype_b, dtype_out, is_mixed):
        """2D 矩阵乘法 (内部函数)"""
        if is_mixed:
            return run_cpu_golden(
                op_name="matmul_mixed",
                cmd_args=["matmul_mixed", dtype_a, dtype_b, "@a.bin", "@b.bin", "@output", str(M), str(K), str(N), dtype_out],
                inputs={"a.bin": (a, dtype_a), "b.bin": (b, dtype_b)},
                output_name="c.bin",
                output_dtype=dtype_out,
                output_size=M * N,
                output_shape=(M, N),
            )
        else:
            return run_cpu_golden(
                op_name="matmul",
                cmd_args=["matmul", dtype_a, "@a.bin", "@b.bin", "@output", str(M), str(K), str(N)],
                inputs={"a.bin": (a, dtype_a), "b.bin": (b, dtype_a)},
                output_name="c.bin",
                output_dtype=dtype_a,
                output_size=M * N,
                output_shape=(M, N),
            )

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a.astype(np.float64), b.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["a", "b"],
    description="逐元素加法",
)
class Add(Op):
    """加法"""
    name = "add"

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b).astype(np.float32)

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.float64) + b.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["a", "b"],
    description="逐元素乘法",
)
class Mul(Op):
    """乘法"""
    name = "mul"

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * b).astype(np.float32)

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.float64) * b.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["a", "b"],
    description="逐元素除法",
)
class Div(Op):
    """除法"""
    name = "div"

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a / b).astype(np.float32)

    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a.astype(np.float64) / b.astype(np.float64)).astype(np.float32)


@register_op(
    inputs=["x"],
    optional=["axes"],
    description="转置 (交换最后两个维度或指定轴)",
    has_cpp_golden=True,
)
class Transpose(Op):
    """Transpose: 支持任意维度转置"""
    name = "transpose"

    def golden_python(self, x: np.ndarray, axes: tuple = None) -> np.ndarray:
        """
        转置

        Args:
            x: 输入
            axes: 轴顺序，如 (0, 1, 3, 2)。None 则交换最后两个维度
        """
        if axes is None:
            return np.swapaxes(x, -2, -1).astype(np.float32)
        return np.transpose(x, axes).astype(np.float32)

    def cpu_golden(self, x: np.ndarray, axes: tuple = None) -> np.ndarray:
        """C++ Golden 实现 (仅支持 4D 交换最后两个维度)"""
        dtype = get_cpu_golden_dtype()
        x = np.asarray(x, dtype=np.float32)

        if x.ndim != 4:
            raise ValueError(f"cpu_golden transpose requires 4D input, got {x.ndim}D")

        d0, d1, d2, d3 = x.shape

        # 输出 shape: [d0, d1, d3, d2]
        return run_cpu_golden(
            op_name="transpose",
            cmd_args=["transpose", dtype, "@x.bin", "@output", str(d0), str(d1), str(d2), str(d3)],
            inputs={"x.bin": (x, dtype)},
            output_name="y.bin",
            output_dtype=dtype,
            output_size=d0 * d1 * d2 * d3,
            output_shape=(d0, d1, d3, d2),
        )

    def reference(self, x: np.ndarray, axes: tuple = None) -> np.ndarray:
        if axes is None:
            return np.swapaxes(x.astype(np.float64), -2, -1).astype(np.float32)
        return np.transpose(x.astype(np.float64), axes).astype(np.float32)


@register_op(
    inputs=["q", "k", "v"],
    optional=["mask", "scale"],
    description="Scaled Dot-Product Attention",
)
class Attention(Op):
    """Scaled Dot-Product Attention"""
    name = "attention"

    def golden_python(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        d_k = q.shape[-1]
        scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask * (-1e9)
        # softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        return np.matmul(attn_weights, v).astype(np.float32)

    def reference(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        q64 = q.astype(np.float64)
        k64 = k.astype(np.float64)
        v64 = v.astype(np.float64)
        d_k = q64.shape[-1]
        scores = np.matmul(q64, k64.swapaxes(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask.astype(np.float64) * (-1e9)
        # softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        return np.matmul(attn_weights, v64).astype(np.float32)


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
transpose = Transpose()
attention = Attention()
