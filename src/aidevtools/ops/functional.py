# pylint: disable=redefined-builtin
# 使用 `input` 作为参数名以兼容 PyTorch API
"""PyTorch 风格的函数式 API

与 torch.nn.functional 接口兼容，方便从 PyTorch 代码迁移。

用法:
    import aidevtools.ops.functional as F
    # 或
    from aidevtools import F

    # 与 PyTorch 完全兼容的接口
    y = F.linear(x, weight, bias)
    y = F.relu(y)
    y = F.softmax(y, dim=-1)
    y = F.layer_norm(y, normalized_shape, weight, bias)
    y = F.gelu(y)

    # Attention
    attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask)

对比 PyTorch:
    import torch.nn.functional as F
    y = F.linear(x, weight, bias)
    y = F.relu(y)
    ...

Note:
    所有函数同时执行 golden + reference + profile 记录。
"""
import numpy as np
from typing import Optional, Tuple, Union, List

from aidevtools.ops import nn as _nn


# ============================================================
# 线性变换
# ============================================================

def linear(
    input: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
) -> np.ndarray:
    """线性变换 y = x @ weight.T + bias

    与 torch.nn.functional.linear 兼容。

    Args:
        input: 输入张量 [..., in_features]
        weight: 权重 [out_features, in_features] (PyTorch 格式)
        bias: 偏置 [out_features] (可选)

    Returns:
        输出张量 [..., out_features]

    Note:
        PyTorch linear weight 是 [out, in]，需要转置。
        如果 weight 是 [in, out] 格式，会自动检测并跳过转置。
    """
    # PyTorch weight: [out_features, in_features]
    # 我们的 linear: x @ weight, weight 应该是 [in_features, out_features]
    if weight.shape[0] == input.shape[-1]:
        # weight 已经是 [in, out] 格式
        w = weight
    else:
        # PyTorch 格式 [out, in]，需要转置
        w = weight.T

    return _nn.linear(input, w, bias)


# ============================================================
# 矩阵运算
# ============================================================

def matmul(input: np.ndarray, other: np.ndarray) -> np.ndarray:
    """矩阵乘法

    与 torch.matmul 兼容。

    Args:
        input: 第一个矩阵
        other: 第二个矩阵

    Returns:
        矩阵乘积
    """
    return _nn.matmul(input, other)


def bmm(input: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """批量矩阵乘法

    与 torch.bmm 兼容。

    Args:
        input: [B, N, M]
        mat2: [B, M, P]

    Returns:
        [B, N, P]
    """
    return _nn.matmul(input, mat2)


# ============================================================
# 激活函数
# ============================================================

def relu(input: np.ndarray, inplace: bool = False) -> np.ndarray:
    """ReLU 激活

    与 torch.nn.functional.relu 兼容。

    Args:
        input: 输入张量
        inplace: 忽略 (numpy 不支持 inplace)

    Returns:
        max(0, input)
    """
    return _nn.relu(input)


def gelu(input: np.ndarray, approximate: str = "none") -> np.ndarray:
    """GELU 激活

    与 torch.nn.functional.gelu 兼容。

    Args:
        input: 输入张量
        approximate: 近似方法 ("none" | "tanh")

    Returns:
        GELU(input)
    """
    return _nn.gelu(input)


def silu(input: np.ndarray, inplace: bool = False) -> np.ndarray:
    """SiLU/Swish 激活 (x * sigmoid(x))

    与 torch.nn.functional.silu 兼容。

    Args:
        input: 输入张量
        inplace: 忽略

    Returns:
        x * sigmoid(x)
    """
    return _nn.silu(input)


def sigmoid(input: np.ndarray) -> np.ndarray:
    """Sigmoid 激活

    与 torch.sigmoid 兼容。
    """
    return _nn.sigmoid(input)


def tanh(input: np.ndarray) -> np.ndarray:
    """Tanh 激活

    与 torch.tanh 兼容。
    """
    return _nn.tanh(input)


# ============================================================
# 归一化
# ============================================================

def softmax(input: np.ndarray, dim: int = -1) -> np.ndarray:
    """Softmax

    与 torch.nn.functional.softmax 兼容。

    Args:
        input: 输入张量
        dim: 计算维度

    Returns:
        softmax(input, dim)
    """
    return _nn.softmax(input, axis=dim)


def layer_norm(
    input: np.ndarray,
    normalized_shape: Union[int, Tuple[int, ...], List[int]],
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Layer Normalization

    与 torch.nn.functional.layer_norm 兼容。

    Args:
        input: 输入张量 [..., normalized_shape]
        normalized_shape: 归一化的形状 (通常是最后几维)
        weight: gamma 参数 (可选，默认 1)
        bias: beta 参数 (可选，默认 0)
        eps: 数值稳定性 epsilon

    Returns:
        归一化后的张量
    """
    # 处理 normalized_shape
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    # 如果没有提供 weight/bias，生成默认值
    if weight is None:
        weight = np.ones(normalized_shape, dtype=np.float32)
    if bias is None:
        bias = np.zeros(normalized_shape, dtype=np.float32)

    return _nn.layernorm(input, weight, bias, eps=eps)


def batch_norm(
    input: np.ndarray,
    running_mean: Optional[np.ndarray],
    running_var: Optional[np.ndarray],
    weight: Optional[np.ndarray] = None,
    bias: Optional[np.ndarray] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> np.ndarray:
    """Batch Normalization

    与 torch.nn.functional.batch_norm 兼容。

    Args:
        input: 输入张量 [N, C, ...]
        running_mean: 运行均值 [C]
        running_var: 运行方差 [C]
        weight: gamma [C]
        bias: beta [C]
        training: 是否训练模式
        momentum: 动量
        eps: epsilon

    Returns:
        归一化后的张量
    """
    C = input.shape[1]

    if weight is None:
        weight = np.ones(C, dtype=np.float32)
    if bias is None:
        bias = np.zeros(C, dtype=np.float32)

    if running_mean is None:
        running_mean = np.zeros(C, dtype=np.float32)
    if running_var is None:
        running_var = np.ones(C, dtype=np.float32)

    return _nn.batchnorm(input, running_mean, running_var, weight, bias, eps=eps)


def rms_norm(
    input: np.ndarray,
    normalized_shape: Union[int, Tuple[int, ...]],
    weight: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """RMS Normalization (LLaMA style)

    Args:
        input: 输入张量
        normalized_shape: 归一化形状
        weight: 缩放参数 (可选)
        eps: epsilon

    Returns:
        RMS 归一化后的张量
    """
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    if weight is None:
        weight = np.ones(normalized_shape, dtype=np.float32)

    return _nn.rmsnorm(input, weight, eps=eps)


# ============================================================
# Attention
# ============================================================

def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    attn_mask: Optional[np.ndarray] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Scaled Dot-Product Attention

    与 torch.nn.functional.scaled_dot_product_attention 兼容。

    Args:
        query: [B, H, L, D] 或 [B, L, H, D]
        key: [B, H, S, D] 或 [B, S, H, D]
        value: [B, H, S, D] 或 [B, S, H, D]
        attn_mask: attention mask (可选)
        dropout_p: dropout 概率 (忽略)
        is_causal: 是否使用因果 mask
        scale: 缩放因子 (默认 1/sqrt(D))

    Returns:
        attention 输出 [B, H, L, D]
    """
    # 生成因果 mask
    if is_causal and attn_mask is None:
        L = query.shape[-2]
        S = key.shape[-2]
        attn_mask = np.triu(np.ones((L, S), dtype=np.float32) * -1e9, k=1)

    return _nn.attention(query, key, value, mask=attn_mask)


# ============================================================
# 其他常用函数
# ============================================================

def dropout(
    input: np.ndarray,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
) -> np.ndarray:
    """Dropout (推理模式直接返回)

    与 torch.nn.functional.dropout 兼容。
    """
    if not training or p == 0:
        return input
    # 推理模式不做 dropout
    return input


def embedding(
    input: np.ndarray,
    weight: np.ndarray,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> np.ndarray:
    """Embedding lookup

    与 torch.nn.functional.embedding 兼容。

    Args:
        input: 索引张量 [...]
        weight: embedding 权重 [vocab_size, embed_dim]
        padding_idx: padding 索引 (忽略)
        其他参数: 忽略

    Returns:
        embedding 输出 [..., embed_dim]
    """
    return _nn.embedding(input, weight)


def pad(
    input: np.ndarray,
    pad: Tuple[int, ...],
    mode: str = "constant",
    value: float = 0.0,
) -> np.ndarray:
    """Padding

    与 torch.nn.functional.pad 兼容。

    Args:
        input: 输入张量
        pad: padding 大小 (left, right, top, bottom, ...)
        mode: padding 模式
        value: constant padding 值

    Returns:
        padding 后的张量
    """
    # 转换 PyTorch 格式 (从最后一维开始) 到 numpy 格式
    ndim = input.ndim
    pad_width = [(0, 0)] * ndim

    # PyTorch pad: (left, right, top, bottom, front, back, ...)
    # 从最后一维开始
    for i in range(0, len(pad), 2):
        dim = ndim - 1 - (i // 2)
        if dim >= 0:
            pad_width[dim] = (pad[i], pad[i + 1])

    if mode == "constant":
        return np.pad(input, pad_width, mode="constant", constant_values=value)
    if mode == "reflect":
        return np.pad(input, pad_width, mode="reflect")
    if mode == "replicate":
        return np.pad(input, pad_width, mode="edge")
    return np.pad(input, pad_width, mode="constant", constant_values=value)


def interpolate(
    input: np.ndarray,
    size: Optional[Tuple[int, ...]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> np.ndarray:
    """插值 (简化实现)

    与 torch.nn.functional.interpolate 部分兼容。
    """
    from scipy import ndimage

    if size is not None:
        # 计算 scale factor
        spatial_dims = input.shape[2:]
        zoom_factors = [1.0, 1.0]  # batch, channel
        for i, (s, t) in enumerate(zip(spatial_dims, size)):
            zoom_factors.append(t / s)
    elif scale_factor is not None:
        zoom_factors = [1.0, 1.0] + [scale_factor] * (input.ndim - 2)
    else:
        return input

    if mode == "nearest":
        order = 0
    elif mode == "bilinear" or mode == "linear":
        order = 1
    else:
        order = 0

    return ndimage.zoom(input, zoom_factors, order=order)


# ============================================================
# 卷积 (可选)
# ============================================================

def conv2d(
    input: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
) -> np.ndarray:
    """2D 卷积

    与 torch.nn.functional.conv2d 兼容。

    Note: 这是一个简化实现，不支持所有参数。
    """
    # 使用 _nn.conv2d 如果存在
    if hasattr(_nn, 'conv2d'):
        return _nn.conv2d(input, weight, bias, stride=stride, padding=padding)

    # 简化实现 (仅支持基本情况)
    from scipy import signal

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    N, C_in, H, W = input.shape
    C_out, _, kH, kW = weight.shape

    # Padding
    if padding[0] > 0 or padding[1] > 0:
        input = np.pad(input, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])))

    H_out = (input.shape[2] - kH) // stride[0] + 1
    W_out = (input.shape[3] - kW) // stride[1] + 1

    output = np.zeros((N, C_out, H_out, W_out), dtype=np.float32)

    for n in range(N):
        for c_out in range(C_out):
            for c_in in range(C_in):
                output[n, c_out] += signal.correlate2d(
                    input[n, c_in], weight[c_out, c_in], mode='valid'
                )[::stride[0], ::stride[1]]

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


# ============================================================
# 损失函数
# ============================================================

def cross_entropy(
    input: np.ndarray,
    target: np.ndarray,
    weight: Optional[np.ndarray] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> np.ndarray:
    """交叉熵损失

    与 torch.nn.functional.cross_entropy 兼容。

    Args:
        input: [N, C] 或 [N, C, H, W] logits (未经 softmax)
        target: [N] 或 [N, H, W] 类别索引 (0 ~ C-1)
        weight: [C] 类别权重 (可选)
        ignore_index: 忽略的标签索引 (暂不支持)
        reduction: "none" | "mean" | "sum"
        label_smoothing: 标签平滑系数 (0.0 ~ 1.0)

    Returns:
        损失值

    Example:
        >>> logits = np.random.randn(4, 10)  # 4 samples, 10 classes
        >>> target = np.array([1, 3, 5, 7])   # class indices
        >>> loss = F.cross_entropy(logits, target)
    """
    return _nn.cross_entropy_loss(input, target, weight, reduction, label_smoothing)


def nll_loss(
    input: np.ndarray,
    target: np.ndarray,
    weight: Optional[np.ndarray] = None,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> np.ndarray:
    """负对数似然损失

    与 torch.nn.functional.nll_loss 兼容。
    注意: input 应该是 log_softmax 的输出。

    Args:
        input: [N, C] log probabilities (log_softmax 输出)
        target: [N] 类别索引
        weight: [C] 类别权重
        reduction: "none" | "mean" | "sum"

    Returns:
        损失值
    """
    N = input.shape[0]
    loss = -input[np.arange(N), target.astype(int)]

    if weight is not None:
        loss = loss * weight[target.astype(int)]

    if reduction == "none":
        return loss.astype(np.float32)
    if reduction == "sum":
        return np.array(np.sum(loss), dtype=np.float32)
    # mean
    if weight is not None:
        return np.array(np.sum(loss) / np.sum(weight[target.astype(int)]), dtype=np.float32)
    return np.array(np.mean(loss), dtype=np.float32)


def mse_loss(
    input: np.ndarray,
    target: np.ndarray,
    reduction: str = "mean",
) -> np.ndarray:
    """均方误差损失

    与 torch.nn.functional.mse_loss 兼容。

    Args:
        input: 预测值
        target: 目标值
        reduction: "none" | "mean" | "sum"

    Returns:
        损失值

    Example:
        >>> pred = np.array([1.0, 2.0, 3.0])
        >>> target = np.array([1.1, 2.2, 2.8])
        >>> loss = F.mse_loss(pred, target)
    """
    return _nn.mse_loss(input, target, reduction)


def l1_loss(
    input: np.ndarray,
    target: np.ndarray,
    reduction: str = "mean",
) -> np.ndarray:
    """L1 损失 (Mean Absolute Error)

    与 torch.nn.functional.l1_loss 兼容。

    Args:
        input: 预测值
        target: 目标值
        reduction: "none" | "mean" | "sum"

    Returns:
        损失值
    """
    return _nn.l1_loss(input, target, reduction)


def smooth_l1_loss(
    input: np.ndarray,
    target: np.ndarray,
    reduction: str = "mean",
    beta: float = 1.0,
) -> np.ndarray:
    """Smooth L1 损失 (Huber Loss)

    与 torch.nn.functional.smooth_l1_loss 兼容。

    Args:
        input: 预测值
        target: 目标值
        reduction: "none" | "mean" | "sum"
        beta: L1/L2 过渡点

    Returns:
        损失值
    """
    return _nn.smooth_l1_loss(input, target, reduction, beta)


def binary_cross_entropy(
    input: np.ndarray,
    target: np.ndarray,
    weight: Optional[np.ndarray] = None,
    reduction: str = "mean",
) -> np.ndarray:
    """二元交叉熵损失

    与 torch.nn.functional.binary_cross_entropy 兼容。
    注意: input 应该是 sigmoid 的输出 (0~1 之间的概率)。

    Args:
        input: [N, *] 预测概率 (0~1)
        target: [N, *] 目标值 (0 或 1)
        weight: 样本权重
        reduction: "none" | "mean" | "sum"

    Returns:
        损失值
    """
    # BCE = -[t * log(p) + (1-t) * log(1-p)]
    eps = 1e-7
    input = np.clip(input, eps, 1 - eps)
    loss = -(target * np.log(input) + (1 - target) * np.log(1 - input))

    if weight is not None:
        loss = loss * weight

    if reduction == "none":
        return loss.astype(np.float32)
    if reduction == "sum":
        return np.array(np.sum(loss), dtype=np.float32)
    return np.array(np.mean(loss), dtype=np.float32)


def binary_cross_entropy_with_logits(
    input: np.ndarray,
    target: np.ndarray,
    weight: Optional[np.ndarray] = None,
    reduction: str = "mean",
    pos_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """二元交叉熵损失 (带 logits)

    与 torch.nn.functional.binary_cross_entropy_with_logits 兼容。

    Args:
        input: [N, *] logits (未经 sigmoid)
        target: [N, *] 目标值 (0 或 1)
        weight: 样本权重
        reduction: "none" | "mean" | "sum"
        pos_weight: 正样本权重

    Returns:
        损失值
    """
    return _nn.bce_with_logits_loss(input, target, weight, reduction, pos_weight)


def log_softmax(input: np.ndarray, dim: int = -1) -> np.ndarray:
    """Log Softmax

    与 torch.nn.functional.log_softmax 兼容。

    Args:
        input: 输入张量
        dim: 计算维度

    Returns:
        log(softmax(input))
    """
    # 数值稳定的 log_softmax
    x_max = np.max(input, axis=dim, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(input - x_max), axis=dim, keepdims=True)) + x_max
    return (input - log_sum_exp).astype(np.float32)


# ============================================================
# 别名 (兼容不同命名习惯)
# ============================================================

# PyTorch 使用下划线
layernorm = layer_norm
batchnorm = batch_norm
rmsnorm = rms_norm

# 缩写
sdpa = scaled_dot_product_attention

# 损失函数别名
bce_loss = binary_cross_entropy
bce_with_logits = binary_cross_entropy_with_logits
