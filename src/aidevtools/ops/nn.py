# pylint: disable=redefined-builtin
# 使用 `input` 作为参数名以兼容 PyTorch API
"""神经网络算子

每个算子包含三种实现：
- golden_python: Python Golden 实现（fp32，精确实现）
- cpu_golden: C++ Golden 实现（通过 subprocess 调用）
- reference: 参考实现（fp32，用于 fuzzy 比对）

使用 @register_op 装饰器自动注册算子元信息。
"""
import numpy as np

from aidevtools.ops.base import Op, fp32_reference
from aidevtools.ops.cpu_golden import (
    get_cpu_golden_dtype,
    get_matmul_dtypes,
    run_cpu_golden,
)
from aidevtools.ops.registry import register_op


@register_op(
    inputs=["x", "weight"],
    optional=["bias"],
    description="线性变换 y = x @ weight + bias",
    auto_gen={
        "x": "input",
        "weight": "xavier",
        "bias": "uniform",
    },
    compute_unit="cube",
    weight_params=["weight", "bias"],
)
class Linear(Op):
    """线性层: y = x @ W + b"""
    name = "linear"

    @staticmethod
    def compute_flops(s):
        """FLOPs: 2 * batch * M * K * N"""
        x_shape = s.get("x_shape", (1, 1))
        weight_shape = s.get("weight_shape", (1, 1))
        if len(x_shape) >= 2 and len(weight_shape) >= 2:
            batch = int(np.prod(x_shape[:-2])) if len(x_shape) > 2 else 1
            M = x_shape[-2] if len(x_shape) >= 2 else 1
            K = x_shape[-1]
            N = weight_shape[-1]
            return batch * 2 * M * K * N
        return 0

    def golden_python(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
        y = np.matmul(x, weight)
        if bias is not None:
            y = y + bias
        return y.astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
        y = np.matmul(x, weight)
        if bias is not None:
            y = y + bias
        return y

    def cpu_golden(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray = None) -> np.ndarray:
        """C++ Golden 实现 (复用 MatMul cpu_golden)"""
        # linear = matmul + bias
        y = MatMul().cpu_golden(x, weight)
        if bias is not None:
            y = y + bias.astype(np.float32)
        return y


@register_op(
    inputs=["x"],
    description="ReLU 激活 y = max(0, x)",
    compute_unit="vector",
)
class ReLU(Op):
    """ReLU: y = max(0, x)"""
    name = "relu"

    @staticmethod
    def compute_flops(s):
        """1 op/element"""
        return s.get("x_size", 0)

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x).astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


@register_op(
    inputs=["x"],
    description="GELU 激活 (近似)",
    compute_unit="vector",
)
class GELU(Op):
    """GELU 近似"""
    name = "gelu"

    @staticmethod
    def compute_flops(s):
        """~10 ops/element (tanh expensive)"""
        return 10 * s.get("x_size", 0)

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return (0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))).astype(np.float32)

    def cpu_golden(self, x: np.ndarray) -> np.ndarray:
        """C++ Golden 未实现，使用 Python 实现"""
        return self.golden_python(x)

    @fp32_reference
    def reference(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


@register_op(
    inputs=["x"],
    description="Sigmoid 激活 y = 1 / (1 + exp(-x))",
    compute_unit="vector",
)
class Sigmoid(Op):
    """Sigmoid: y = 1 / (1 + exp(-x))"""
    name = "sigmoid"

    @staticmethod
    def compute_flops(s):
        """exp + div + add + neg = 4 ops/element"""
        return 4 * s.get("x_size", 0)

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return (1 / (1 + np.exp(-x))).astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


@register_op(
    inputs=["x"],
    description="Tanh 激活",
    compute_unit="vector",
)
class Tanh(Op):
    """Tanh"""
    name = "tanh"

    @staticmethod
    def compute_flops(s):
        """~6 ops/element (tanh expensive)"""
        return 6 * s.get("x_size", 0)

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x).astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)


@register_op(
    inputs=["x"],
    description="SiLU/Swish 激活 y = x * sigmoid(x) (LLaMA FFN)",
    compute_unit="vector",
)
class SiLU(Op):
    """SiLU (Swish): y = x * sigmoid(x)

    用于 LLaMA, Mistral 等现代 LLM 的 FFN。
    """
    name = "silu"

    @staticmethod
    def compute_flops(s):
        """sigmoid + mul = 5 ops/element"""
        return 5 * s.get("x_size", 0)

    def golden_python(self, x: np.ndarray) -> np.ndarray:
        return (x * (1 / (1 + np.exp(-x)))).astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray) -> np.ndarray:
        return x * (1 / (1 + np.exp(-x)))


@register_op(
    inputs=["x"],
    optional=["axis"],
    description="Softmax 激活 (防溢出)",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Softmax(Op):
    """安全 Softmax（防溢出）"""
    name = "softmax"

    @staticmethod
    def compute_flops(s):
        """max + sub + exp + sum + div = 5 ops/element"""
        return 5 * s.get("x_size", 0)

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

    @fp32_reference
    def reference(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


@register_op(
    inputs=["x", "gamma", "beta"],
    optional=["eps"],
    description="Layer Normalization",
    has_cpp_golden=True,
    auto_gen={
        "x": "input",
        "gamma": "ones:-1",
        "beta": "zeros:-1",
    },
    compute_unit="vector",
    weight_params=["gamma", "beta"],
)
class LayerNorm(Op):
    """Layer Normalization"""
    name = "layernorm"

    @staticmethod
    def compute_flops(s):
        """mean + var + norm + scale + shift = ~8 ops/element"""
        return 8 * s.get("x_size", 0)

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

    @fp32_reference
    def reference(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


@register_op(
    inputs=["x", "gamma"],
    optional=["eps"],
    description="RMS Normalization (LLaMA/Mistral)",
    auto_gen={
        "x": "input",
        "gamma": "ones:-1",
    },
    compute_unit="vector",
    weight_params=["gamma"],
)
class RMSNorm(Op):
    """RMS Normalization: y = x / rms(x) * gamma

    用于 LLaMA, Mistral, Qwen 等现代 LLM。
    比 LayerNorm 更高效（无需计算 mean）。
    """
    name = "rmsnorm"

    @staticmethod
    def compute_flops(s):
        """square + mean + sqrt + div + mul = 6 ops/element"""
        return 6 * s.get("x_size", 0)

    def golden_python(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Python Golden 实现 (fp32)"""
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return (x / rms * gamma).astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        return x / rms * gamma


@register_op(
    inputs=["x", "gamma", "beta"],
    optional=["mean", "var", "eps"],
    description="Batch Normalization",
    auto_gen={
        "x": "input",
        "gamma": "ones:-1",
        "beta": "zeros:-1",
    },
    compute_unit="vector",
    weight_params=["gamma", "beta"],
)
class BatchNorm(Op):
    """Batch Normalization"""
    name = "batchnorm"

    @staticmethod
    def compute_flops(s):
        """~8 ops/element"""
        return 8 * s.get("x_size", 0)

    def golden_python(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                      mean: np.ndarray = None, var: np.ndarray = None, eps: float = 1e-5) -> np.ndarray:
        if mean is None:
            mean = np.mean(x, axis=0, keepdims=True)
        if var is None:
            var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return (gamma * x_norm + beta).astype(np.float32)

    @fp32_reference
    def reference(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                  mean: np.ndarray = None, var: np.ndarray = None, eps: float = 1e-5) -> np.ndarray:
        if mean is None:
            mean = np.mean(x, axis=0, keepdims=True)
        if var is None:
            var = np.var(x, axis=0, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


@register_op(
    inputs=["input_ids", "embed_table"],
    description="Embedding 查表",
    compute_unit="vector",
    memory_pattern="random",
    weight_params=["embed_table"],
)
class Embedding(Op):
    """Embedding 查表"""
    name = "embedding"

    @staticmethod
    def compute_flops(s):
        """纯内存操作，无计算"""
        return 0

    def golden_python(self, input_ids: np.ndarray, embed_table: np.ndarray) -> np.ndarray:
        return embed_table[input_ids].astype(np.float32)

    def reference(self, input_ids: np.ndarray, embed_table: np.ndarray) -> np.ndarray:
        # Embedding 是查表操作，直接返回 fp32
        return embed_table[input_ids].astype(np.float32)


@register_op(
    inputs=["a", "b"],
    description="矩阵乘法 c = a @ b",
    has_cpp_golden=True,
    compute_unit="cube",
    weight_params=["b"],
)
class MatMul(Op):
    """矩阵乘法"""
    name = "matmul"

    @staticmethod
    def compute_flops(s):
        """FLOPs: batch * 2 * M * K * N"""
        a_shape = s.get("a_shape", (1, 1))
        b_shape = s.get("b_shape", (1, 1))
        if len(a_shape) >= 2:
            batch = int(np.prod(a_shape[:-2])) if len(a_shape) > 2 else 1
            M, K = a_shape[-2:]
            N = b_shape[-1] if len(b_shape) >= 1 else 1
            return batch * 2 * M * K * N
        return 0

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
        return run_cpu_golden(
            op_name="matmul",
            cmd_args=["matmul", dtype_a, "@a.bin", "@b.bin", "@output", str(M), str(K), str(N)],
            inputs={"a.bin": (a, dtype_a), "b.bin": (b, dtype_a)},
            output_name="c.bin",
            output_dtype=dtype_a,
            output_size=M * N,
            output_shape=(M, N),
        )

    @fp32_reference
    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.matmul(a, b)


@register_op(
    inputs=["a", "b"],
    description="逐元素加法",
    compute_unit="vector",
)
class Add(Op):
    """加法"""
    name = "add"

    @staticmethod
    def compute_flops(s):
        """1 op/element"""
        return s.get("a_size", 0)

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b).astype(np.float32)

    @fp32_reference
    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b


@register_op(
    inputs=["a", "b"],
    description="逐元素乘法",
    compute_unit="vector",
)
class Mul(Op):
    """乘法"""
    name = "mul"

    @staticmethod
    def compute_flops(s):
        """1 op/element"""
        return s.get("a_size", 0)

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a * b).astype(np.float32)

    @fp32_reference
    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b


@register_op(
    inputs=["a", "b"],
    description="逐元素除法",
    compute_unit="vector",
)
class Div(Op):
    """除法"""
    name = "div"

    @staticmethod
    def compute_flops(s):
        """1 op/element"""
        return s.get("a_size", 0)

    def golden_python(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a / b).astype(np.float32)

    @fp32_reference
    def reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a / b


@register_op(
    inputs=["x"],
    optional=["axes"],
    description="转置 (交换最后两个维度或指定轴)",
    has_cpp_golden=True,
    compute_unit="vector",
    memory_pattern="strided",
)
class Transpose(Op):
    """Transpose: 支持任意维度转置"""
    name = "transpose"

    @staticmethod
    def compute_flops(s):
        """纯内存操作，无计算"""
        return 0

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
        """C++ Golden 实现 (支持 2D/3D/4D，交换最后两个维度)"""
        dtype = get_cpu_golden_dtype()
        x = np.asarray(x, dtype=np.float32)
        original_ndim = x.ndim

        if x.ndim < 2 or x.ndim > 4:
            raise ValueError(f"cpu_golden transpose requires 2D-4D input, got {x.ndim}D")

        # 将 2D/3D 扩展到 4D
        if x.ndim == 2:
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
        elif x.ndim == 3:
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])

        d0, d1, d2, d3 = x.shape

        # 输出 shape: [d0, d1, d3, d2]
        result = run_cpu_golden(
            op_name="transpose",
            cmd_args=["transpose", dtype, "@x.bin", "@output", str(d0), str(d1), str(d2), str(d3)],
            inputs={"x.bin": (x, dtype)},
            output_name="y.bin",
            output_dtype=dtype,
            output_size=d0 * d1 * d2 * d3,
            output_shape=(d0, d1, d3, d2),
        )

        # 恢复原始维度
        if original_ndim == 2:
            result = result.reshape(d3, d2)
        elif original_ndim == 3:
            result = result.reshape(d1, d3, d2)

        return result

    @fp32_reference
    def reference(self, x: np.ndarray, axes: tuple = None) -> np.ndarray:
        if axes is None:
            return np.swapaxes(x, -2, -1)
        return np.transpose(x, axes)


@register_op(
    inputs=["q", "k", "v"],
    optional=["mask", "scale"],
    description="Scaled Dot-Product Attention",
    compute_unit="cube",  # dominated by matmul
)
class Attention(Op):
    """Scaled Dot-Product Attention"""
    name = "attention"

    @staticmethod
    def compute_flops(s):
        """计算 Attention FLOPs: QK + softmax + SV"""
        q_shape = s.get("q_shape", (1, 1, 1, 1))
        k_shape = s.get("k_shape", (1, 1, 1, 1))
        if len(q_shape) == 4:
            batch, heads, seq_q, head_dim = q_shape
            seq_kv = k_shape[-2]
        elif len(q_shape) == 3:
            batch, seq_q, head_dim = q_shape
            heads = 1
            seq_kv = k_shape[-2]
        else:
            return 0
        # QK: 2 * batch * heads * seq_q * head_dim * seq_kv
        qk_flops = batch * heads * 2 * seq_q * head_dim * seq_kv
        # Softmax: 5 * batch * heads * seq_q * seq_kv
        soft_flops = batch * heads * 5 * seq_q * seq_kv
        # SV: 2 * batch * heads * seq_q * seq_kv * head_dim
        sv_flops = batch * heads * 2 * seq_q * seq_kv * head_dim
        return qk_flops + soft_flops + sv_flops

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

    @fp32_reference
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


# ============================================================
# 损失函数
# ============================================================

@register_op(
    inputs=["input", "target"],
    optional=["weight", "reduction", "label_smoothing"],
    description="交叉熵损失 (含 log_softmax)",
    compute_unit="vector",
)
class CrossEntropyLoss(Op):
    """交叉熵损失函数

    与 torch.nn.functional.cross_entropy 兼容。
    内部实现 log_softmax + nll_loss。
    """
    name = "cross_entropy"

    @staticmethod
    def compute_flops(s):
        """CrossEntropy FLOPs: log_softmax + nll_loss"""
        input_shape = s.get("input_shape", (1, 1))
        # log_softmax: 5 * size (max, sub, exp, sum, log)
        # nll_loss: 1 * batch
        size = int(np.prod(input_shape))
        batch = input_shape[0] if len(input_shape) > 0 else 1
        return 5 * size + batch

    def golden_python(
        self,
        input: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> np.ndarray:
        """
        Args:
            input: [N, C] 或 [N, C, ...] logits (未经 softmax)
            target: [N] 或 [N, ...] 类别索引 (0 ~ C-1)
            weight: [C] 类别权重 (可选)
            reduction: "none" | "mean" | "sum"
            label_smoothing: 标签平滑 (0.0 ~ 1.0)
        """
        # 处理多维情况: [N, C, H, W] -> [N, C, H*W] -> [N*H*W, C]
        if input.ndim > 2:
            N, C = input.shape[:2]
            input.shape[2:]
            input = input.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
            target = target.reshape(-1)

        N, C = input.shape

        # log_softmax (数值稳定)
        x_max = np.max(input, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(input - x_max), axis=-1, keepdims=True)) + x_max
        log_probs = input - log_sum_exp  # [N, C]

        # 标签平滑
        if label_smoothing > 0:
            # smoothed target: (1 - smoothing) * one_hot + smoothing / C
            smooth_target = np.full((N, C), label_smoothing / C, dtype=np.float32)
            smooth_target[np.arange(N), target.astype(int)] += (1.0 - label_smoothing)
            loss = -np.sum(smooth_target * log_probs, axis=-1)
        else:
            # 标准 nll_loss
            loss = -log_probs[np.arange(N), target.astype(int)]

        # 应用类别权重
        if weight is not None:
            loss = loss * weight[target.astype(int)]

        # reduction
        if reduction == "none":
            return loss.astype(np.float32)
        if reduction == "sum":
            return np.array(np.sum(loss), dtype=np.float32)
        # mean
        if weight is not None:
            return np.array(np.sum(loss) / np.sum(weight[target.astype(int)]), dtype=np.float32)
        return np.array(np.mean(loss), dtype=np.float32)

    @fp32_reference
    def reference(
        self,
        input: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> np.ndarray:
        return self.golden_python(input, target, weight, reduction, label_smoothing)


@register_op(
    inputs=["input", "target"],
    optional=["reduction"],
    description="均方误差损失",
    compute_unit="vector",
)
class MSELoss(Op):
    """均方误差损失函数

    与 torch.nn.functional.mse_loss 兼容。
    """
    name = "mse_loss"

    @staticmethod
    def compute_flops(s):
        """MSE FLOPs: sub + square + mean"""
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 3 * size  # sub, square, sum/mean

    def golden_python(
        self,
        input: np.ndarray,
        target: np.ndarray,
        reduction: str = "mean",
    ) -> np.ndarray:
        """
        Args:
            input: 预测值
            target: 目标值
            reduction: "none" | "mean" | "sum"
        """
        diff = input - target
        loss = diff ** 2

        if reduction == "none":
            return loss.astype(np.float32)
        if reduction == "sum":
            return np.array(np.sum(loss), dtype=np.float32)
        # mean
        return np.array(np.mean(loss), dtype=np.float32)

    @fp32_reference
    def reference(
        self,
        input: np.ndarray,  # pylint: disable=redefined-builtin
        target: np.ndarray,
        reduction: str = "mean",
    ) -> np.ndarray:
        return self.golden_python(input, target, reduction)


@register_op(
    inputs=["input", "target"],
    optional=["reduction"],
    description="L1 损失 (MAE)",
    compute_unit="vector",
)
class L1Loss(Op):
    """L1 损失函数 (Mean Absolute Error)

    与 torch.nn.functional.l1_loss 兼容。
    """
    name = "l1_loss"

    @staticmethod
    def compute_flops(s):
        """L1 FLOPs: sub + abs + mean"""
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 3 * size

    def golden_python(
        self,
        input: np.ndarray,  # pylint: disable=redefined-builtin
        target: np.ndarray,
        reduction: str = "mean",
    ) -> np.ndarray:
        loss = np.abs(input - target)

        if reduction == "none":
            return loss.astype(np.float32)
        if reduction == "sum":
            return np.array(np.sum(loss), dtype=np.float32)
        # mean
        return np.array(np.mean(loss), dtype=np.float32)

    @fp32_reference
    def reference(
        self,
        input: np.ndarray,  # pylint: disable=redefined-builtin
        target: np.ndarray,
        reduction: str = "mean",
    ) -> np.ndarray:
        return self.golden_python(input, target, reduction)


@register_op(
    inputs=["input", "target"],
    optional=["reduction", "beta"],
    description="Smooth L1 损失 (Huber Loss)",
    compute_unit="vector",
)
class SmoothL1Loss(Op):
    """Smooth L1 损失 (Huber Loss)

    与 torch.nn.functional.smooth_l1_loss 兼容。
    """
    name = "smooth_l1_loss"

    @staticmethod
    def compute_flops(s):
        """Smooth L1 FLOPs: sub + abs + compare + compute"""
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 4 * size  # sub, abs, where, compute

    def golden_python(
        self,
        input: np.ndarray,  # pylint: disable=redefined-builtin
        target: np.ndarray,
        reduction: str = "mean",
        beta: float = 1.0,
    ) -> np.ndarray:
        diff = np.abs(input - target)
        loss = np.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )

        if reduction == "none":
            return loss.astype(np.float32)
        if reduction == "sum":
            return np.array(np.sum(loss), dtype=np.float32)
        # mean
        return np.array(np.mean(loss), dtype=np.float32)

    @fp32_reference
    def reference(
        self,
        input: np.ndarray,  # pylint: disable=redefined-builtin
        target: np.ndarray,
        reduction: str = "mean",
        beta: float = 1.0,
    ) -> np.ndarray:
        return self.golden_python(input, target, reduction, beta)


@register_op(
    inputs=["input", "target"],
    optional=["weight", "reduction", "pos_weight"],
    description="二元交叉熵 (带 logits)",
    compute_unit="vector",
)
class BCEWithLogitsLoss(Op):
    """二元交叉熵损失 (带 logits)

    与 torch.nn.functional.binary_cross_entropy_with_logits 兼容。
    """
    name = "bce_with_logits"

    @staticmethod
    def compute_flops(s):
        """BCE with logits FLOPs: sigmoid + log + mul + sum"""
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 6 * size  # max, exp, log, mul, add, etc.

    def golden_python(
        self,
        input: np.ndarray,
        target: np.ndarray,
        weight: np.ndarray = None,
        reduction: str = "mean",
        pos_weight: np.ndarray = None,
    ) -> np.ndarray:
        # 数值稳定的 BCE: max(x, 0) - x * t + log(1 + exp(-|x|))
        max_val = np.maximum(-input, 0)
        if pos_weight is not None:
            log_weight = (pos_weight - 1) * target + 1
            loss = (1 - target) * input + log_weight * (max_val + np.log(np.exp(-max_val) + np.exp(-input - max_val)))
        else:
            loss = (1 - target) * input + max_val + np.log(np.exp(-max_val) + np.exp(-input - max_val))

        if weight is not None:
            loss = loss * weight

        if reduction == "none":
            return loss.astype(np.float32)
        if reduction == "sum":
            return np.array(np.sum(loss), dtype=np.float32)
        # mean
        return np.array(np.mean(loss), dtype=np.float32)

    @fp32_reference
    def reference(
        self,
        input: np.ndarray,  # pylint: disable=redefined-builtin
        target: np.ndarray,
        weight: np.ndarray = None,
        reduction: str = "mean",
        pos_weight: np.ndarray = None,
    ) -> np.ndarray:
        return self.golden_python(input, target, weight, reduction, pos_weight)


# 实例化算子（方便直接调用）
linear = Linear()
relu = ReLU()
gelu = GELU()
sigmoid = Sigmoid()
tanh = Tanh()
silu = SiLU()
softmax = Softmax()
layernorm = LayerNorm()
rmsnorm = RMSNorm()
batchnorm = BatchNorm()
embedding = Embedding()
matmul = MatMul()
add = Add()
mul = Mul()
div = Div()
transpose = Transpose()
attention = Attention()

# 损失函数
cross_entropy_loss = CrossEntropyLoss()
mse_loss = MSELoss()
l1_loss = L1Loss()
smooth_l1_loss = SmoothL1Loss()
bce_with_logits_loss = BCEWithLogitsLoss()
