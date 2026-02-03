# pylint: disable=redefined-builtin,arguments-differ,arguments-renamed,unused-argument
# redefined-builtin: 使用 `input` 作为参数名以兼容 PyTorch API
# arguments-differ/renamed: 子类重写方法时使用具体参数签名而非 *args, **kwargs
# unused-argument: 保留 API 兼容参数 (axis, eps 等) 和基类 compute_flops(s) 签名
"""神经网络算子

自定义 API，torch 对外部不可见。

工作流：
1. 用户调用 F.matmul, F.softmax 等自定义 API
2. cpu_golden/gpu_golden 计算 golden（C++ 实现）
3. torch_reference 计算 reference（torch fp32，内部使用）
4. 比对 golden 和 reference

使用 @register_op 装饰器自动注册算子元信息。

注意:
- cpu_golden 通过 C++ 实现 (src/aidevtools/golden/cpp/)
- torch_reference 内部使用 torch 计算（对外不可见）
- 目前有 C++ cpu_golden 的算子: matmul, softmax, layernorm, transpose
"""

import numpy as np

from aidevtools.ops.base import Op
from aidevtools.ops.cpu_golden import (
    get_cpu_golden_dtype,
    get_matmul_dtypes,
    run_cpu_golden,
)
from aidevtools.ops.traced_tensor import TracedTensor
from aidevtools.ops.registry import register_op


# ============================================================
# Torch 内部导入（对外不可见）
# ============================================================

def _import_torch():
    """延迟导入 torch"""
    try:
        import torch
        return torch
    except ImportError:
        return None


def _to_torch(x: np.ndarray):
    """numpy -> torch tensor (fp32)"""
    torch = _import_torch()
    if torch is None:
        return None
    return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _to_numpy(t) -> np.ndarray:
    """torch tensor -> numpy (fp32)"""
    if t is None:
        return None
    return t.detach().cpu().numpy().astype(np.float32)


# ============================================================
# 通用 cpu_golden 辅助函数
# ============================================================


def _extract_data(x, target_dtype: str) -> np.ndarray:
    """从输入提取数据，支持 TracedTensor 和 TracedTensor

    如果输入是 TracedTensor 且精度匹配，直接返回内部数据（跳过转换）。
    否则转换为 fp32。

    Args:
        x: 输入数据 (np.ndarray, TracedTensor, 或 TracedTensor)
        target_dtype: 目标精度 (gfp4/gfp8/gfp16)

    Returns:
        fp32 numpy array
    """
    if isinstance(x, TracedTensor):
        # TracedTensor: 检查精度是否匹配
        if x.dtype == target_dtype:
            # 精度匹配，直接使用（已量化的 fp32 表示）
            return x.data
        # 精度不匹配，使用原始数据
        return np.asarray(x.data, dtype=np.float32)
    # 普通数组，转为 fp32
    return np.asarray(x, dtype=np.float32)


def _unary_cpu_golden(op_name: str, x) -> np.ndarray:
    """单目运算的通用 cpu_golden 实现

    适用于: relu, gelu, sigmoid, tanh, silu 等激活函数

    Args:
        op_name: 算子名称 (对应 C++ golden 命令)
        x: 输入数组 (np.ndarray 或 TracedTensor)

    Returns:
        输出数组 (shape 与输入相同)
    """
    dtype = get_cpu_golden_dtype()
    data = _extract_data(x, dtype)
    original_shape = data.shape
    size = data.size

    y = run_cpu_golden(
        op_name=op_name,
        cmd_args=[op_name, dtype, "@input.bin", "@output", str(size)],
        inputs={"input.bin": (data.flatten(), dtype)},
        output_name="output.bin",
        output_dtype=dtype,
        output_size=size,
        output_shape=(size,),
    )

    return y.reshape(original_shape)


def _binary_cpu_golden(op_name: str, a, b) -> np.ndarray:
    """双目运算的通用 cpu_golden 实现

    适用于: add, mul, div 等逐元素运算

    Args:
        op_name: 算子名称 (对应 C++ golden 命令)
        a: 第一个输入数组 (np.ndarray 或 TracedTensor)
        b: 第二个输入数组 (np.ndarray 或 TracedTensor)

    Returns:
        输出数组 (shape 与 a 相同)
    """
    dtype = get_cpu_golden_dtype()
    a_data = _extract_data(a, dtype)
    b_data = _extract_data(b, dtype)
    original_shape = a_data.shape
    size = a_data.size

    c = run_cpu_golden(
        op_name=op_name,
        cmd_args=[op_name, dtype, "@a.bin", "@b.bin", "@output", str(size)],
        inputs={"a.bin": (a_data.flatten(), dtype), "b.bin": (b_data.flatten(), dtype)},
        output_name="output.bin",
        output_dtype=dtype,
        output_size=size,
        output_shape=(size,),
    )

    return c.reshape(original_shape)


# ============================================================
# 线性层
# ============================================================


@register_op(
    inputs=["input", "weight"],
    optional=["bias"],
    description="线性变换 y = input @ weight.T + bias (PyTorch 格式)",
    auto_gen={
        "input": "input",
        "weight": "xavier",
        "bias": "uniform",
    },
    compute_unit="cube",
    weight_params=["weight", "bias"],
)
class Linear(Op):
    """线性层 (PyTorch 格式)

    Args:
        input: [..., in_features]
        weight: [out_features, in_features] (PyTorch 格式)
        bias: [out_features] 可选

    Returns:
        [..., out_features]
    """

    name = "linear"

    @staticmethod
    def compute_flops(s):
        """FLOPs: 2 * batch * M * K * N"""
        input_shape = s.get("input_shape", (1, 1))
        weight_shape = s.get("weight_shape", (1, 1))
        if len(input_shape) >= 2 and len(weight_shape) >= 2:
            batch = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1
            K = input_shape[-1]  # in_features
            N = weight_shape[0]  # out_features
            return batch * 2 * K * N
        return 0

    def cpu_golden(
        self, input: np.ndarray, weight: np.ndarray, bias: np.ndarray = None
    ) -> np.ndarray:
        """C++ Golden 实现

        Linear = MatMul + Add，拆成两个量化操作
        """
        y = MatMul().cpu_golden(input, weight.T)
        if bias is not None:
            # bias 需要 broadcast 到 y 的 shape，用量化 add
            # 注意：_binary_cpu_golden 要求 shape 相同，需要先 broadcast
            bias_broadcast = np.broadcast_to(bias, y.shape)
            y = _binary_cpu_golden("add", y, bias_broadcast)
        return y

    def torch_reference(
        self, input: np.ndarray, weight: np.ndarray, bias: np.ndarray = None
    ) -> np.ndarray:
        """Torch Reference: y = input @ weight.T + bias"""
        torch = _import_torch()
        if torch is None:
            return None
        import torch.nn.functional as torch_F
        input_t = _to_torch(input)
        weight_t = _to_torch(weight)
        bias_t = _to_torch(bias) if bias is not None else None
        y_t = torch_F.linear(input_t, weight_t, bias_t)
        return _to_numpy(y_t)


# ============================================================
# 激活函数
# ============================================================


@register_op(
    inputs=["x"],
    description="ReLU 激活 y = max(0, x)",
    has_cpp_golden=True,
    compute_unit="vector",
)
class ReLU(Op):
    """ReLU: y = max(0, x)"""

    name = "relu"

    @staticmethod
    def compute_flops(s):
        return s.get("x_size", 0)

    def cpu_golden(self, x: np.ndarray, inplace: bool = False) -> np.ndarray:  # pylint: disable=unused-argument
        return _unary_cpu_golden("relu", x)

    def torch_reference(self, x: np.ndarray, inplace: bool = False) -> np.ndarray:
        """Torch Reference: relu"""
        torch = _import_torch()
        if torch is None:
            return None
        x_t = _to_torch(x)
        y_t = torch.nn.functional.relu(x_t)
        return _to_numpy(y_t)


@register_op(
    inputs=["x"],
    description="GELU 激活 (近似)",
    has_cpp_golden=True,
    compute_unit="vector",
)
class GELU(Op):
    """GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""

    name = "gelu"

    @staticmethod
    def compute_flops(s):
        return 10 * s.get("x_size", 0)

    def cpu_golden(self, x: np.ndarray) -> np.ndarray:
        return _unary_cpu_golden("gelu", x)

    def torch_reference(self, x: np.ndarray) -> np.ndarray:
        """Torch Reference: gelu"""
        torch = _import_torch()
        if torch is None:
            return None
        x_t = _to_torch(x)
        y_t = torch.nn.functional.gelu(x_t)
        return _to_numpy(y_t)


@register_op(
    inputs=["x"],
    description="Sigmoid 激活 y = 1 / (1 + exp(-x))",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Sigmoid(Op):
    """Sigmoid: y = 1 / (1 + exp(-x))"""

    name = "sigmoid"

    @staticmethod
    def compute_flops(s):
        return 4 * s.get("x_size", 0)

    def cpu_golden(self, x: np.ndarray) -> np.ndarray:
        return _unary_cpu_golden("sigmoid", x)

    def torch_reference(self, x: np.ndarray) -> np.ndarray:
        """Torch Reference: sigmoid"""
        torch = _import_torch()
        if torch is None:
            return None
        x_t = _to_torch(x)
        y_t = torch.sigmoid(x_t)
        return _to_numpy(y_t)


@register_op(
    inputs=["x"],
    description="Tanh 激活",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Tanh(Op):
    """Tanh"""

    name = "tanh"

    @staticmethod
    def compute_flops(s):
        return 6 * s.get("x_size", 0)

    def cpu_golden(self, x: np.ndarray) -> np.ndarray:
        return _unary_cpu_golden("tanh", x)

    def torch_reference(self, x: np.ndarray) -> np.ndarray:
        """Torch Reference: tanh"""
        torch = _import_torch()
        if torch is None:
            return None
        x_t = _to_torch(x)
        y_t = torch.tanh(x_t)
        return _to_numpy(y_t)


@register_op(
    inputs=["x"],
    description="SiLU/Swish 激活 y = x * sigmoid(x) (LLaMA FFN)",
    has_cpp_golden=True,
    compute_unit="vector",
)
class SiLU(Op):
    """SiLU (Swish): y = x * sigmoid(x)"""

    name = "silu"

    @staticmethod
    def compute_flops(s):
        return 5 * s.get("x_size", 0)

    def cpu_golden(self, x: np.ndarray) -> np.ndarray:
        return _unary_cpu_golden("silu", x)

    def torch_reference(self, x: np.ndarray) -> np.ndarray:
        """Torch Reference: silu"""
        torch = _import_torch()
        if torch is None:
            return None
        x_t = _to_torch(x)
        y_t = torch.nn.functional.silu(x_t)
        return _to_numpy(y_t)


@register_op(
    inputs=["input"],
    optional=["dim"],
    description="Softmax 激活 (PyTorch 格式)",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Softmax(Op):
    """Softmax (PyTorch 格式)"""

    name = "softmax"

    @staticmethod
    def compute_flops(s):
        return 5 * s.get("input_size", 0)

    def cpu_golden(self, input: np.ndarray, dim: int = -1) -> np.ndarray:  # pylint: disable=unused-argument
        # NOTE: C++ golden 只支持 dim=-1
        dtype = get_cpu_golden_dtype()
        x = np.asarray(input, dtype=np.float32)
        original_shape = x.shape

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

    def torch_reference(self, input: np.ndarray, dim: int = -1) -> np.ndarray:
        """Torch Reference: softmax(input, dim)"""
        torch = _import_torch()
        if torch is None:
            return None
        import torch.nn.functional as torch_F
        input_t = _to_torch(input)
        y_t = torch_F.softmax(input_t, dim=dim)
        return _to_numpy(y_t)


# ============================================================
# Normalization
# ============================================================


@register_op(
    inputs=["input", "normalized_shape"],
    optional=["weight", "bias", "eps"],
    description="Layer Normalization",
    has_cpp_golden=True,
    auto_gen={
        "input": "input",
        "weight": "ones:-1",
        "bias": "zeros:-1",
    },
    compute_unit="vector",
    weight_params=["weight", "bias"],
)
class LayerNorm(Op):
    """Layer Normalization (PyTorch 兼容)

    签名: layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)
    """

    name = "layernorm"

    @staticmethod
    def compute_flops(s):
        return 8 * s.get("input_size", 0)

    def cpu_golden(
        self, input: np.ndarray, normalized_shape: tuple,
        weight: np.ndarray = None, bias: np.ndarray = None, eps: float = 1e-5  # pylint: disable=unused-argument
    ) -> np.ndarray:
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        hidden = normalized_shape[-1]
        if weight is None:
            weight = np.ones(hidden, dtype=np.float32)
        if bias is None:
            bias = np.zeros(hidden, dtype=np.float32)

        dtype = get_cpu_golden_dtype()
        x = np.asarray(input, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)

        original_shape = x.shape

        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim > 2:
            x = x.reshape(-1, hidden)

        batch = x.shape[0]

        assert weight.shape == (hidden,), f"weight shape mismatch: {weight.shape} vs ({hidden},)"
        assert bias.shape == (hidden,), f"bias shape mismatch: {bias.shape} vs ({hidden},)"

        y = run_cpu_golden(
            op_name="layernorm",
            cmd_args=[
                "layernorm",
                dtype,
                "@x.bin",
                "@gamma.bin",
                "@beta.bin",
                "@output",
                str(batch),
                str(hidden),
            ],
            inputs={
                "x.bin": (x, dtype),
                "gamma.bin": (weight, dtype),
                "beta.bin": (bias, dtype),
            },
            output_name="y.bin",
            output_dtype=dtype,
            output_size=batch * hidden,
            output_shape=(batch, hidden),
        )

        return y.reshape(original_shape)

    def torch_reference(
        self, input: np.ndarray, normalized_shape: tuple,
        weight: np.ndarray = None, bias: np.ndarray = None, eps: float = 1e-5
    ) -> np.ndarray:
        """Torch Reference: layer_norm(input, normalized_shape, weight, bias, eps)"""
        torch = _import_torch()
        if torch is None:
            return None
        import torch.nn.functional as torch_F
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        input_t = _to_torch(input)
        weight_t = _to_torch(weight) if weight is not None else None
        bias_t = _to_torch(bias) if bias is not None else None
        y_t = torch_F.layer_norm(input_t, normalized_shape, weight_t, bias_t, eps)
        return _to_numpy(y_t)


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
    """RMS Normalization: y = x / rms(x) * gamma"""

    name = "rmsnorm"

    @staticmethod
    def compute_flops(s):
        return 6 * s.get("x_size", 0)


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
        return 8 * s.get("x_size", 0)


# ============================================================
# Embedding
# ============================================================


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
    def compute_flops(_s):
        return 0


# ============================================================
# 矩阵运算
# ============================================================


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
        a_shape = s.get("a_shape", (1, 1))
        b_shape = s.get("b_shape", (1, 1))
        if len(a_shape) >= 2:
            batch = int(np.prod(a_shape[:-2])) if len(a_shape) > 2 else 1
            M, K = a_shape[-2:]
            N = b_shape[-1] if len(b_shape) >= 1 else 1
            return batch * 2 * M * K * N
        return 0

    def cpu_golden(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dtype_a, dtype_b, dtype_out = get_matmul_dtypes()
        is_mixed = (dtype_a != dtype_b) or (dtype_a != dtype_out)

        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        a_batch_shape = a.shape[:-2] if a.ndim > 2 else ()

        M, K = a.shape[-2:]
        if b.ndim == 1:
            K2, N = b.shape[0], 1
            b = b.reshape(K2, N)
        else:
            K2, N = b.shape[-2:]

        assert K == K2, f"Shape mismatch: a.shape={a.shape}, b.shape={b.shape}"

        if a.ndim == 2 and b.ndim == 2:
            return self._matmul_2d(a, b, M, K, N, dtype_a, dtype_b, dtype_out, is_mixed)

        if a.ndim > 2:
            batch_size = int(np.prod(a_batch_shape))
            a_flat = a.reshape(batch_size, M, K)
        else:
            batch_size = 1
            a_flat = a.reshape(1, M, K)

        if b.ndim == 2:
            b_batched = False
            b_flat = b.reshape(1, K, N)
        else:
            b_batched = True
            b_flat = b.reshape(batch_size, K, N)

        c_flat = np.zeros((batch_size, M, N), dtype=np.float32)
        for i in range(batch_size):
            a_i = a_flat[i]
            b_i = b_flat[i] if b_batched else b_flat[0]
            c_flat[i] = self._matmul_2d(a_i, b_i, M, K, N, dtype_a, dtype_b, dtype_out, is_mixed)

        output_shape = a_batch_shape + (M, N)
        return c_flat.reshape(output_shape)

    def _matmul_2d(self, a, b, M, K, N, dtype_a, dtype_b, dtype_out, is_mixed):
        if is_mixed:
            return run_cpu_golden(
                op_name="matmul_mixed",
                cmd_args=[
                    "matmul_mixed",
                    dtype_a,
                    dtype_b,
                    "@a.bin",
                    "@b.bin",
                    "@output",
                    str(M),
                    str(K),
                    str(N),
                    dtype_out,
                ],
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

    def torch_reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Torch Reference: c = a @ b"""
        torch = _import_torch()
        if torch is None:
            return None
        a_t = _to_torch(a)
        b_t = _to_torch(b)
        c_t = torch.matmul(a_t, b_t)
        return _to_numpy(c_t)


@register_op(
    inputs=["a", "b"],
    description="逐元素加法",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Add(Op):
    """加法"""

    name = "add"

    @staticmethod
    def compute_flops(s):
        return s.get("a_size", 0)

    def cpu_golden(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _binary_cpu_golden("add", a, b)

    def torch_reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Torch Reference: a + b"""
        torch = _import_torch()
        if torch is None:
            return None
        a_t = _to_torch(a)
        b_t = _to_torch(b)
        c_t = a_t + b_t
        return _to_numpy(c_t)


@register_op(
    inputs=["a", "b"],
    description="逐元素乘法",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Mul(Op):
    """乘法"""

    name = "mul"

    @staticmethod
    def compute_flops(s):
        return s.get("a_size", 0)

    def cpu_golden(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _binary_cpu_golden("mul", a, b)

    def torch_reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Torch Reference: a * b"""
        torch = _import_torch()
        if torch is None:
            return None
        a_t = _to_torch(a)
        b_t = _to_torch(b)
        c_t = a_t * b_t
        return _to_numpy(c_t)


@register_op(
    inputs=["a", "b"],
    description="逐元素除法",
    has_cpp_golden=True,
    compute_unit="vector",
)
class Div(Op):
    """除法"""

    name = "div"

    @staticmethod
    def compute_flops(s):
        return s.get("a_size", 0)

    def cpu_golden(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _binary_cpu_golden("div", a, b)

    def torch_reference(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Torch Reference: a / b"""
        torch = _import_torch()
        if torch is None:
            return None
        a_t = _to_torch(a)
        b_t = _to_torch(b)
        c_t = a_t / b_t
        return _to_numpy(c_t)


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
    def compute_flops(_s):
        return 0

    def cpu_golden(self, x: np.ndarray, axes: tuple = None) -> np.ndarray:  # pylint: disable=unused-argument
        dtype = get_cpu_golden_dtype()
        x = np.asarray(x, dtype=np.float32)
        original_ndim = x.ndim

        if x.ndim < 2 or x.ndim > 4:
            raise ValueError(f"cpu_golden transpose requires 2D-4D input, got {x.ndim}D")

        if x.ndim == 2:
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
        elif x.ndim == 3:
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])

        d0, d1, d2, d3 = x.shape

        result = run_cpu_golden(
            op_name="transpose",
            cmd_args=["transpose", dtype, "@x.bin", "@output", str(d0), str(d1), str(d2), str(d3)],
            inputs={"x.bin": (x, dtype)},
            output_name="y.bin",
            output_dtype=dtype,
            output_size=d0 * d1 * d2 * d3,
            output_shape=(d0, d1, d3, d2),
        )

        if original_ndim == 2:
            result = result.reshape(d3, d2)
        elif original_ndim == 3:
            result = result.reshape(d1, d3, d2)

        return result

    def torch_reference(self, x: np.ndarray, axes: tuple = None) -> np.ndarray:
        """Torch Reference: transpose"""
        torch = _import_torch()
        if torch is None:
            return None
        x_t = _to_torch(x)
        if axes is not None:
            y_t = x_t.permute(*axes)
        else:
            # 默认交换最后两个维度
            y_t = x_t.transpose(-2, -1)
        return _to_numpy(y_t)


# ============================================================
# Attention
# ============================================================


@register_op(
    inputs=["q", "k", "v"],
    optional=["mask", "scale"],
    description="Scaled Dot-Product Attention",
    compute_unit="cube",
)
class Attention(Op):
    """Scaled Dot-Product Attention"""

    name = "attention"

    @staticmethod
    def compute_flops(s):
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
        qk_flops = batch * heads * 2 * seq_q * head_dim * seq_kv
        soft_flops = batch * heads * 5 * seq_q * seq_kv
        sv_flops = batch * heads * 2 * seq_q * seq_kv * head_dim
        return qk_flops + soft_flops + sv_flops


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
    """交叉熵损失函数"""

    name = "cross_entropy"

    @staticmethod
    def compute_flops(s):
        input_shape = s.get("input_shape", (1, 1))
        size = int(np.prod(input_shape))
        batch = input_shape[0] if len(input_shape) > 0 else 1
        return 5 * size + batch


@register_op(
    inputs=["input", "target"],
    optional=["reduction"],
    description="均方误差损失",
    compute_unit="vector",
)
class MSELoss(Op):
    """均方误差损失函数"""

    name = "mse_loss"

    @staticmethod
    def compute_flops(s):
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 3 * size


@register_op(
    inputs=["input", "target"],
    optional=["reduction"],
    description="L1 损失 (MAE)",
    compute_unit="vector",
)
class L1Loss(Op):
    """L1 损失函数 (Mean Absolute Error)"""

    name = "l1_loss"

    @staticmethod
    def compute_flops(s):
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 3 * size


@register_op(
    inputs=["input", "target"],
    optional=["reduction", "beta"],
    description="Smooth L1 损失 (Huber Loss)",
    compute_unit="vector",
)
class SmoothL1Loss(Op):
    """Smooth L1 损失 (Huber Loss)"""

    name = "smooth_l1_loss"

    @staticmethod
    def compute_flops(s):
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 4 * size


@register_op(
    inputs=["input", "target"],
    optional=["weight", "reduction", "pos_weight"],
    description="二元交叉熵 (带 logits)",
    compute_unit="vector",
)
class BCEWithLogitsLoss(Op):
    """二元交叉熵损失 (带 logits)"""

    name = "bce_with_logits"

    @staticmethod
    def compute_flops(s):
        input_shape = s.get("input_shape", (1,))
        size = int(np.prod(input_shape))
        return 6 * size


# ============================================================
# 实例化算子（方便直接调用）
# ============================================================

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

# 损失函数实例
cross_entropy_loss = CrossEntropyLoss()
mse_loss = MSELoss()
l1_loss = L1Loss()
smooth_l1_loss = SmoothL1Loss()
bce_with_logits_loss = BCEWithLogitsLoss()


# ============================================================
# 独立函数 (无对应类)
# ============================================================


def dropout(input: np.ndarray, _p: float = 0.5, _training: bool = False) -> np.ndarray:
    """Dropout (推理模式直接返回)"""
    return input


# ============================================================
# PyTorch 风格别名
# ============================================================

layer_norm = layernorm
batch_norm = batchnorm
rms_norm = rmsnorm

scaled_dot_product_attention = attention
sdpa = attention

cross_entropy = cross_entropy_loss
bce_with_logits = bce_with_logits_loss
binary_cross_entropy_with_logits = bce_with_logits_loss

bmm = matmul
