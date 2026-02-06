"""公共随机数生成器

将 datagen / ops.datagen / frontend.datagen 中重复的随机数生成逻辑
提取为一个公共工具类。

功能:
    1. 基础随机生成: generate(shape, method, dtype) → ndarray
    2. 策略解析:     generate_from_strategy(strategy, context) → (ndarray, shape)
    3. 量化模拟:     generate(..., qtype="bfp8") → 经过 quantize→dequantize 的 fp32 数据

用法:
    from aidevtools.core.random import RandomGenerator

    rng = RandomGenerator(seed=42)

    # 基本生成
    x = rng.generate((2, 3), method="normal")
    w = rng.generate((64, 128), method="xavier")
    b = rng.generate((64,), method="uniform", low=-0.1, high=0.1)

    # 量化模拟 (quantize→dequantize, 带精度损失)
    x_bfp8 = rng.generate((2, 3), method="normal", qtype="bfp8")

    # 策略字符串解析 (供 datagen 模块使用)
    data, shape = rng.generate_from_strategy("xavier:-1,out_features", context)
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class Method(Enum):
    """随机数生成方法"""

    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"
    ONES = "ones"
    XAVIER = "xavier"
    KAIMING = "kaiming"
    QA_UNIFORM = "qa_uniform"      # 量化感知均匀分布
    QA_NORMAL = "qa_normal"        # 量化感知正态分布(截断)


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


def parse_shape(spec: str, context: Dict[str, Any]) -> Tuple[int, ...]:
    """解析 shape 规格字符串。

    支持:
        "-1"              → input_shape[-1]
        "-2,-1"           → (input_shape[-2], input_shape[-1])
        "out_features"    → context["out_features"]
        "-1,out_features" → (input_shape[-1], out_features)

    Args:
        spec: shape 规格字符串
        context: 上下文字典，需含 "input_shape" 键

    Returns:
        解析后的 shape 元组
    """
    input_shape = context.get("input_shape", (1,))
    parts = [p.strip() for p in spec.split(",")]
    result = []

    for part in parts:
        if not part:
            continue

        if part.lstrip("-").isdigit():
            idx = int(part)
            if abs(idx) <= len(input_shape):
                result.append(input_shape[idx])
            else:
                result.append(1)
        elif part in context:
            val = context[part]
            if isinstance(val, (int, np.integer)):
                result.append(int(val))
            elif isinstance(val, tuple):
                result.extend(val)
            else:
                result.append(int(val))
        else:
            raise ValueError(f"无法解析 shape 规格: {part}")

    return tuple(result)


class RandomGenerator:
    """公共随机数生成器

    集中管理随机数生成、策略解析、量化模拟三个职责。
    各 datagen 模块只需持有一个 RandomGenerator 实例即可。

    量化感知随机数 (QA-aware) 作为全局配置:
        rng = RandomGenerator(seed=42)
        rng.set_qa_config(enabled=True, center=1.0, amplitude=0.5)
        # 之后所有 generate 调用自动应用 QA 约束
        x = rng.generate((100, 200), method="normal")  # 自动受控范围

    Args:
        seed: 随机种子，None 表示不固定种子。
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # 全局量化感知配置
        self._qa_enabled: bool = False
        self._qa_center: float = 1.0
        self._qa_amplitude: float = 0.5

    def set_qa_config(
        self,
        enabled: bool = False,
        center: float = 1.0,
        amplitude: float = 0.5,
    ):
        """设置全局量化感知随机数配置。

        这是一个全局开关，启用后所有 generate / generate_from_strategy
        调用自动将数据范围限制在受控区间内。

        Args:
            enabled: 是否启用量化感知模式
            center: 中心值 (默认 1.0)
            amplitude: 波动幅度 (默认 0.5)
        """
        self._qa_enabled = enabled
        self._qa_center = center
        self._qa_amplitude = amplitude

    @property
    def qa_enabled(self) -> bool:
        """量化感知模式是否启用"""
        return self._qa_enabled

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
        qtype: Optional[str] = None,
        qa_aware: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """生成随机数组。

        Args:
            shape: 输出形状。
            method: 生成方法 — "normal" | "uniform" | "zeros" | "ones"
                    | "xavier" | "kaiming" | "qa_uniform" | "qa_normal"。
            dtype: 输出 numpy dtype，支持字符串别名如 "fp32"、"float16"。
            qtype: 可选量化类型 (如 "bfp8", "bfp16", "gfloat8")。
                   指定时，生成的 fp32 数据会经过 quantize→dequantize
                   模拟量化精度损失，适用于模糊比对场景。
                   None / "fp32" / "float32" 表示不做量化。
            qa_aware: 是否启用量化感知模式。None 表示使用全局配置 (set_qa_config)。
            **kwargs: 方法相关参数：
                normal  — mean (float, 默认 0), std (float, 默认 1)
                uniform — low (float, 默认 -1), high (float, 默认 1)
                xavier  — (自动由 shape 推导 fan_in / fan_out)
                kaiming — (自动由 shape 推导 fan_in)
                qa_uniform/qa_normal —
                    center (float, 默认 1.0), amplitude (float, 默认 0.5),
                    signed (bool, 默认 True)

        Returns:
            np.ndarray，形状为 shape，dtype 为指定类型。
        """
        m = _resolve_method(method)
        out_dtype = _resolve_dtype(dtype)

        # 确定是否启用 QA: 参数覆盖 > 全局配置
        use_qa = qa_aware if qa_aware is not None else self._qa_enabled

        # 量化感知模式: 替换生成方法为受控范围
        if use_qa and m not in (Method.QA_UNIFORM, Method.QA_NORMAL,
                                 Method.ZEROS, Method.ONES):
            center = kwargs.pop("center", kwargs.pop("qa_center", self._qa_center))
            amplitude = kwargs.pop("amplitude", kwargs.pop("qa_amplitude", self._qa_amplitude))
            signed = kwargs.pop("signed", True)
            data = self._dispatch(
                Method.QA_UNIFORM, shape,
                center=center, amplitude=amplitude, signed=signed
            ).astype(out_dtype)
        else:
            data = self._dispatch(m, shape, **kwargs).astype(out_dtype)

        # 量化模拟
        if qtype and qtype not in ("fp32", "float32"):
            data = self._simulate_quantize(data, qtype)

        return data

    # ------------------------------------------------------------------
    # 策略解析 (供 datagen 模块使用)
    # ------------------------------------------------------------------

    def generate_from_strategy(
        self,
        strategy: str,
        context: Dict[str, Any],
        qtype: Optional[str] = None,
        qa_aware: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """根据 auto_gen 策略字符串生成数据。

        合并了三个 datagen 模块中重复的 _gen_from_strategy / _generate_param
        逻辑。strategy 格式为 @register_op 的 auto_gen 配置值。

        Args:
            strategy: 策略字符串，如 "input", "xavier", "kaiming:-1,out_features",
                      "normal:0,0.01,-1", "uniform:-0.1,0.1,-1", "same:input" 等
            context: 上下文字典，需含 "input_shape"，可含 "out_features" 等
            qtype: 可选量化类型，指定时对生成数据做 quantize→dequantize

        Returns:
            (data: np.ndarray, shape: tuple)
        """
        input_shape = context.get("input_shape", (1,))

        # qa_aware: None → 使用全局配置，由 generate() 内部处理

        # --- 简写策略 ---
        if strategy == "input" or strategy == "random":
            shape = input_shape
            data = self.generate(shape, method="normal", qa_aware=qa_aware)

        elif strategy == "xavier":
            out_features = context.get("out_features", input_shape[-1])
            in_features = input_shape[-1]
            shape = (out_features, in_features)
            data = self.generate(shape, method="xavier", qa_aware=qa_aware)

        elif strategy == "kaiming":
            out_features = context.get("out_features", input_shape[-1])
            in_features = input_shape[-1]
            shape = (out_features, in_features)
            data = self.generate(shape, method="kaiming", qa_aware=qa_aware)

        elif strategy == "uniform":
            shape = (context.get("out_features", input_shape[-1]),)
            data = self.generate(shape, method="uniform", qa_aware=qa_aware,
                                 low=-0.1, high=0.1)

        # --- 带参数的策略 ---
        elif strategy.startswith("zeros:"):
            shape = parse_shape(strategy[6:], context)
            data = self.zeros(shape)

        elif strategy.startswith("ones:"):
            shape = parse_shape(strategy[5:], context)
            data = self.ones(shape)

        elif strategy.startswith("xavier:"):
            shape = parse_shape(strategy[7:], context)
            data = self.generate(shape, method="xavier", qa_aware=qa_aware)

        elif strategy.startswith("kaiming:"):
            shape = parse_shape(strategy[8:], context)
            data = self.generate(shape, method="kaiming", qa_aware=qa_aware)

        elif strategy.startswith("same:"):
            ref_param = strategy[5:]
            ref_shape = context.get(f"{ref_param}_shape")
            if ref_shape is None:
                ref_shape = input_shape
            shape = ref_shape
            data = self.generate(shape, method="normal", qa_aware=qa_aware)

        elif strategy.startswith("normal:"):
            parts = strategy[7:].split(",")
            if len(parts) >= 2:
                mean = float(parts[0])
                std = float(parts[1])
                shape_spec = ",".join(parts[2:]) if len(parts) > 2 else "-1"
            else:
                mean, std = 0.0, 1.0
                shape_spec = parts[0]
            shape = parse_shape(shape_spec, context)
            data = self.generate(shape, method="normal", qa_aware=qa_aware,
                                 mean=mean, std=std)

        elif strategy.startswith("uniform:"):
            parts = strategy[8:].split(",")
            if len(parts) >= 3:
                low = float(parts[0])
                high = float(parts[1])
                shape_spec = ",".join(parts[2:])
            elif len(parts) == 2 and parts[0].lstrip("-").replace(".", "").isdigit():
                low = float(parts[0])
                high = float(parts[1])
                shape_spec = "-1"
            else:
                low, high = -0.1, 0.1
                shape_spec = ",".join(parts)
            shape = parse_shape(shape_spec, context)
            data = self.generate(shape, method="uniform", qa_aware=qa_aware,
                                 low=low, high=high)

        else:
            raise ValueError(f"未知生成策略: {strategy}")

        # 量化模拟
        if qtype and qtype not in ("fp32", "float32"):
            data = self._simulate_quantize(data, qtype)

        return data, shape

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

    def qa_uniform(
        self,
        shape: Tuple[int, ...],
        center: float = 1.0,
        amplitude: float = 0.5,
        signed: bool = True,
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """量化感知均匀分布。

        生成数据绝对值在 [center - amplitude, center + amplitude] 范围内，
        可选随机正负号。确保输出动态范围受控。

        原理: 对于 matmul(A, B)，若 A, B 元素绝对值在 [lo, hi] 区间，
        则输出每个元素绝对值在 [K*lo^2, K*hi^2] 区间 (K 为内积维度)，
        max/min 比值 ≤ (hi/lo)^2，可控。

        Args:
            shape: 输出形状
            center: 中心值 (默认 1.0)
            amplitude: 波动幅度 (默认 0.5), 值域 [center-amplitude, center+amplitude]
            signed: 是否随机添加正负号 (默认 True)
            dtype: 输出 dtype
        """
        return self.generate(shape, method="qa_uniform", dtype=dtype,
                             center=center, amplitude=amplitude, signed=signed)

    def qa_normal(
        self,
        shape: Tuple[int, ...],
        center: float = 1.0,
        amplitude: float = 0.5,
        signed: bool = True,
        dtype: Union[str, np.dtype, type, None] = np.float32,
    ) -> np.ndarray:
        """量化感知截断正态分布。

        在 center 附近按正态分布采样，截断到 [center - amplitude, center + amplitude]，
        可选随机正负号。

        Args:
            shape: 输出形状
            center: 中心值 (默认 1.0)
            amplitude: 波动幅度 (默认 0.5)
            signed: 是否随机添加正负号 (默认 True)
            dtype: 输出 dtype
        """
        return self.generate(shape, method="qa_normal", dtype=dtype,
                             center=center, amplitude=amplitude, signed=signed)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        method: Method,
        shape: Tuple[int, ...],
        **kwargs,
    ) -> np.ndarray:
        """根据 method 分发到具体的生成逻辑。"""
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

        if method == Method.QA_UNIFORM:
            return self._qa_uniform(shape, **kwargs)

        if method == Method.QA_NORMAL:
            return self._qa_normal(shape, **kwargs)

        raise ValueError(f"未实现的方法: {method}")  # pragma: no cover

    def _qa_uniform(
        self,
        shape: Tuple[int, ...],
        center: float = 1.0,
        amplitude: float = 0.5,
        signed: bool = True,
        **_kwargs,
    ) -> np.ndarray:
        """量化感知均匀分布实现。

        生成绝对值在 [center - amplitude, center + amplitude] 范围的数据，
        可选随机符号。这确保:
        - 所有值的绝对值具有有限的动态范围
        - max_abs / min_abs ≤ (center + amplitude) / (center - amplitude)
        - 适用于需要 quantize→dequantize 后输出可控的场景

        业界参考:
        - Quantization-Aware Training (QAT) 中的 fake quantization
        - HAWQ (Hessian AWare Quantization) 的分层量化策略
        - 均匀量化的最优区间选择 (min-max / percentile)
        """
        lo = max(center - amplitude, 1e-7)  # 确保正值
        hi = center + amplitude
        base = self._rng.uniform(lo, hi, shape)
        if signed:
            signs = self._rng.choice(np.array([-1.0, 1.0]), shape)
            base = base * signs
        return base.astype(np.float64)

    def _qa_normal(
        self,
        shape: Tuple[int, ...],
        center: float = 1.0,
        amplitude: float = 0.5,
        signed: bool = True,
        **_kwargs,
    ) -> np.ndarray:
        """量化感知截断正态分布实现。

        在 center 附近使用正态分布采样 (std=amplitude/3 → 99.7% 在 3σ 内)，
        截断到 [center - amplitude, center + amplitude]。
        """
        lo = max(center - amplitude, 1e-7)
        hi = center + amplitude
        std = amplitude / 3.0  # 3σ 覆盖
        base = self._rng.normal(center, std, shape)
        base = np.clip(base, lo, hi)
        if signed:
            signs = self._rng.choice(np.array([-1.0, 1.0]), shape)
            base = base * signs
        return base.astype(np.float64)

    @staticmethod
    def _simulate_quantize(data: np.ndarray, qtype: str) -> np.ndarray:
        """量化→反量化，模拟精度损失。延迟导入避免循环依赖。"""
        from aidevtools.formats.quantize import simulate_quantize
        return simulate_quantize(data.astype(np.float32), qtype)
