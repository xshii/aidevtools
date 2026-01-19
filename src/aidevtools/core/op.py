"""统一的算子注册系统"""
from typing import Callable, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from aidevtools.core.tensor import Tensor, generate_random, generate_weight
from aidevtools.core.config import get_config
from aidevtools.core.log import logger


@dataclass
class OpSpec:
    """算子规格"""
    name: str
    python_golden: Optional[Callable] = None
    cpp_golden: Optional[Callable] = None

    # 输入输出配置
    num_inputs: int = 1
    num_weights: int = 1
    weight_shapes: List[str] = field(default_factory=list)  # e.g., ["K,N"] for linear

    # 支持的量化类型
    supported_qtypes: List[str] = field(default_factory=lambda: [
        "float32", "float16", "bfloat16",
        "bfp16", "bfp8",
        "gfloat16", "gfloat8",
    ])

    def has_golden(self, mode: str) -> bool:
        """检查是否有指定模式的 golden"""
        if mode == "python":
            return self.python_golden is not None
        elif mode == "cpp":
            return self.cpp_golden is not None
        return False

    def get_golden(self, mode: str) -> Optional[Callable]:
        """获取指定模式的 golden"""
        if mode == "python":
            return self.python_golden
        elif mode == "cpp":
            return self.cpp_golden
        return None


# 算子注册表
_op_registry: Dict[str, OpSpec] = {}


def register_op(name: str):
    """
    算子注册装饰器

    用法:
        @register_op("linear")
        class LinearOp:
            def python_golden(self, x, w, b=None):
                return np.matmul(x, w) + (b if b is not None else 0)

            def cpp_golden(self, x, w, b=None):
                from some_cpp_lib import cpp_linear
                return cpp_linear(x, w, b)

            num_inputs = 1
            num_weights = 2  # weight + bias
            supported_qtypes = ["float32", "bfp16", "bfp8"]
    """
    def decorator(cls):
        spec = OpSpec(name=name)

        # 提取 golden 实现
        if hasattr(cls, 'python_golden'):
            instance = cls()
            spec.python_golden = instance.python_golden

        if hasattr(cls, 'cpp_golden'):
            instance = cls() if not hasattr(cls, '_instance') else cls._instance
            spec.cpp_golden = instance.cpp_golden

        # 提取配置
        if hasattr(cls, 'num_inputs'):
            spec.num_inputs = cls.num_inputs
        if hasattr(cls, 'num_weights'):
            spec.num_weights = cls.num_weights
        if hasattr(cls, 'weight_shapes'):
            spec.weight_shapes = cls.weight_shapes
        if hasattr(cls, 'supported_qtypes'):
            spec.supported_qtypes = cls.supported_qtypes

        _op_registry[name] = spec
        logger.debug(f"注册算子: {name}")

        return cls

    return decorator


def get_op(name: str) -> Optional[OpSpec]:
    """获取算子规格"""
    return _op_registry.get(name)


def list_ops() -> List[str]:
    """列出所有已注册的算子"""
    return list(_op_registry.keys())


def run_golden(op_name: str, *args, mode: str = None, **kwargs) -> np.ndarray:
    """
    执行 golden 计算

    Args:
        op_name: 算子名称
        *args: 输入参数
        mode: golden 模式 (python/cpp)，None 则使用全局配置
        **kwargs: 其他参数

    Returns:
        golden 输出
    """
    spec = get_op(op_name)
    if spec is None:
        raise ValueError(f"未注册的算子: {op_name}")

    if mode is None:
        mode = get_config().golden_mode

    golden_fn = spec.get_golden(mode)
    if golden_fn is None:
        # 尝试回退到另一个模式
        fallback_mode = "cpp" if mode == "python" else "python"
        golden_fn = spec.get_golden(fallback_mode)
        if golden_fn is not None:
            logger.debug(f"{op_name}: {mode} golden 不存在，回退到 {fallback_mode}")
        else:
            raise ValueError(f"{op_name}: 没有可用的 golden 实现")

    return golden_fn(*args, **kwargs)


# ==================== 内置算子定义 ====================

@register_op("linear")
class LinearOp:
    """线性层: y = x @ W + b"""
    num_inputs = 1
    num_weights = 2
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "bfp4", "gfloat16", "gfloat8"]

    def python_golden(self, x, w, b=None):
        y = np.matmul(x, w)
        if b is not None:
            y = y + b
        return y


@register_op("matmul")
class MatMulOp:
    """矩阵乘法: y = a @ b"""
    num_inputs = 2
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "bfp4", "gfloat16", "gfloat8"]

    def python_golden(self, a, b):
        return np.matmul(a, b)


@register_op("relu")
class ReLUOp:
    """ReLU: y = max(0, x)"""
    num_inputs = 1
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, x):
        return np.maximum(0, x)


@register_op("gelu")
class GELUOp:
    """GELU 近似"""
    num_inputs = 1
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


@register_op("softmax")
class SoftmaxOp:
    """Softmax (数值稳定版)"""
    num_inputs = 1
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


@register_op("layernorm")
class LayerNormOp:
    """Layer Normalization"""
    num_inputs = 1
    num_weights = 2  # gamma, beta
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, x, gamma, beta, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta


@register_op("attention")
class AttentionOp:
    """Scaled Dot-Product Attention"""
    num_inputs = 3  # q, k, v
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "bfp4", "gfloat16", "gfloat8"]

    def python_golden(self, q, k, v, mask=None):
        d_k = q.shape[-1]
        scores = np.matmul(q, k.swapaxes(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores + mask * (-1e9)
        # softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn_weights = scores_exp / np.sum(scores_exp, axis=-1, keepdims=True)
        return np.matmul(attn_weights, v)


@register_op("add")
class AddOp:
    """加法: y = a + b"""
    num_inputs = 2
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, a, b):
        return a + b


@register_op("mul")
class MulOp:
    """乘法: y = a * b"""
    num_inputs = 2
    num_weights = 0
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, a, b):
        return a * b


@register_op("embedding")
class EmbeddingOp:
    """Embedding 查表"""
    num_inputs = 1  # input_ids
    num_weights = 1  # embed_table
    supported_qtypes = ["float32", "float16", "bfloat16", "bfp16", "bfp8", "gfloat16", "gfloat8"]

    def python_golden(self, input_ids, embed_table):
        return embed_table[input_ids]
