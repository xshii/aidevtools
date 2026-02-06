"""公共随机数生成器

将 datagen / ops.datagen / frontend.datagen 中重复的随机数生成逻辑
提取为一个无状态的工具类，只负责：给定 shape、dtype、method、seed → 返回 ndarray。

用法:
    from aidevtools.core.random import RandomGenerator

    rng = RandomGenerator(seed=42)

    # 基本生成
    x = rng.generate((2, 3), method="normal")
    w = rng.generate((64, 128), method="xavier")
    b = rng.generate((64,), method="uniform", low=-0.1, high=0.1)

    # 指定输出 dtype
    x_fp16 = rng.generate((2, 3), method="normal", dtype="float16")
"""

from enum import Enum
from typing import Optional, Tuple, Union

import numpy as np


class Method(Enum):
    """随机数生成方法"""

    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"
    ONES = "ones"
    XAVIER = "xavier"
    KAIMING = "kaiming"


# numpy dtype 别名映射
_DTYPE_MAP = {
    "fp32": np.float32,
    "float32": np.float32,
    "fp16": np.float16,
    "float16": np.float16,
    "fp64": np.float64,
    "float64": np.float64,
}


def _resolve_dtype(dtype: Union[str, np.dtype, type, None]) -> np.dtype:
    """将各种 dtype 表示统一转为 np.dtype。"""
    if dtype is None:
        return np.dtype(np.float32)
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, str):
        if dtype in _DTYPE_MAP:
            return np.dtype(_DTYPE_MAP[dtype])
        return np.dtype(dtype)
    return np.dtype(dtype)


def _resolve_method(method: Union[str, Method]) -> Method:
    """将字符串转为 Method 枚举。"""
    if isinstance(method, Method):
        return method
    try:
        return Method(method.lower())
    except ValueError:
        raise ValueError(
            f"Unknown distribution / 未知的随机方法: {method!r}, "
            f"supported: {[m.value for m in Method]}"
        )


class RandomGenerator:
    """公共随机数生成器

    只关注「给定参数 → 生成 ndarray」，不涉及 L2 内存布局、tensor 命名
    等上层业务逻辑。各 datagen 模块在自身的 _add_tensor / _gen_from_strategy
    里调用本类即可。

    Args:
        seed: 随机种子，None 表示不固定种子。
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 种子管理
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        """重置 RNG 状态。seed=None 时使用上次设定的种子。"""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def generate(
        self,
        shape: Tuple[int, ...],
        method: Union[str, Method] = "normal",
        dtype: Union[str, np.dtype, type, None] = np.float32,
        **kwargs,
    ) -> np.ndarray:
        """生成随机数组。

        Args:
            shape: 输出形状。
            method: 生成方法 — "normal" | "uniform" | "zeros" | "ones"
                    | "xavier" | "kaiming"。
            dtype: 输出 numpy dtype，支持字符串别名如 "fp32"、"float16"。
            **kwargs: 方法相关参数：
                normal  — mean (float, 默认 0), std (float, 默认 1)
                uniform — low (float, 默认 -1), high (float, 默认 1)
                xavier  — (自动由 shape 推导 fan_in / fan_out)
                kaiming — (自动由 shape 推导 fan_in)

        Returns:
            np.ndarray，形状为 shape，dtype 为指定类型。
        """
        m = _resolve_method(method)
        out_dtype = _resolve_dtype(dtype)
        data = self._dispatch(m, shape, **kwargs)
        return data.astype(out_dtype)

    # ------------------------------------------------------------------
    # 便捷方法（等价于 generate + 固定 method）
    # ------------------------------------------------------------------

    def normal(
        self,
        shape: Tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """正态分布。"""
        return self.generate(shape, method="normal", dtype=dtype, mean=mean, std=std)

    def uniform(
        self,
        shape: Tuple[int, ...],
        low: float = -1.0,
        high: float = 1.0,
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """均匀分布。"""
        return self.generate(shape, method="uniform", dtype=dtype, low=low, high=high)

    def zeros(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """全零。"""
        return self.generate(shape, method="zeros", dtype=dtype)

    def ones(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """全一。"""
        return self.generate(shape, method="ones", dtype=dtype)

    def xavier(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """Xavier / Glorot 初始化。"""
        return self.generate(shape, method="xavier", dtype=dtype)

    def kaiming(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """Kaiming / He 初始化。"""
        return self.generate(shape, method="kaiming", dtype=dtype)

    # ------------------------------------------------------------------
    # 内部分发
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        method: Method,
        shape: Tuple[int, ...],
        **kwargs,
    ) -> np.ndarray:
        """根据 method 分发到具体的生成逻辑，返回 float64 ndarray。"""
        if method == Method.NORMAL:
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            if mean == 0.0 and std == 1.0:
                return self._rng.standard_normal(shape)
            return self._rng.normal(mean, std, shape)

        if method == Method.UNIFORM:
            low = kwargs.get("low", -1.0)
            high = kwargs.get("high", 1.0)
            return self._rng.uniform(low, high, shape)

        if method == Method.ZEROS:
            return np.zeros(shape, dtype=np.float64)

        if method == Method.ONES:
            return np.ones(shape, dtype=np.float64)

        if method == Method.XAVIER:
            fan_in = shape[-1] if len(shape) >= 1 else 1
            fan_out = shape[0] if len(shape) >= 2 else 1
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self._rng.uniform(-limit, limit, shape)

        if method == Method.KAIMING:
            fan_in = shape[-1] if len(shape) >= 1 else 1
            std = np.sqrt(2.0 / fan_in)
            return self._rng.normal(0, std, shape)

        raise ValueError(f"未实现的方法: {method}")  # pragma: no cover
