"""
数据生成器

提供输入和权重数据的生成功能。
"""

from typing import Optional, Tuple, Union

import numpy as np

from .types import DistType, DType, Tensor, TensorMeta


class DataGenerator:
    """
    数据生成器

    使用示例:
        gen = DataGenerator(seed=42)
        x = gen.gen_input(shape=(2, 64), dtype="bfp16", dist="normal")
        w = gen.gen_weight(shape=(64, 64), dtype="bfp16", init="xavier")
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 随机种子 (可选)
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._counter = 0

    def reset(self, seed: Optional[int] = None):
        """重置生成器"""
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._counter = 0

    def gen_input(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, DType] = DType.FP32,
        dist: Union[str, DistType] = DistType.NORMAL,
        name: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        """
        生成输入数据

        Args:
            shape: 数据形状
            dtype: 数据类型
            dist: 数据分布
            name: 名称
            **kwargs: 额外参数
                - mean: 正态分布均值 (默认 0)
                - std: 正态分布标准差 (默认 1)
                - low: 均匀分布下界 (默认 -1)
                - high: 均匀分布上界 (默认 1)

        Returns:
            Tensor
        """
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)
        if isinstance(dist, str):
            dist = DistType(dist.lower())

        # 生成数据
        data = self._gen_data(shape, dist, **kwargs)

        # 名称
        if name is None:
            name = f"input_{self._counter}"
            self._counter += 1

        return Tensor(
            data=data,
            meta=TensorMeta(shape=shape, dtype=dtype, name=name),
        )

    def gen_weight(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, DType] = DType.FP32,
        init: Union[str, DistType] = DistType.XAVIER,
        name: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        """
        生成权重数据

        Args:
            shape: 数据形状
            dtype: 数据类型
            init: 初始化方法
            name: 名称
            **kwargs: 额外参数

        Returns:
            Tensor
        """
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)
        if isinstance(init, str):
            dist_values = [d.value for d in DistType]
            init = DistType(init.lower()) if init.lower() in dist_values else DistType.XAVIER

        # 生成数据
        data = self._gen_data(shape, init, **kwargs)

        # 名称
        if name is None:
            name = f"weight_{self._counter}"
            self._counter += 1

        return Tensor(
            data=data,
            meta=TensorMeta(shape=shape, dtype=dtype, name=name),
        )

    def gen_zeros(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, DType] = DType.FP32,
        name: Optional[str] = None,
    ) -> Tensor:
        """生成全零数据"""
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)

        data = np.zeros(shape, dtype=np.float32)

        if name is None:
            name = f"zeros_{self._counter}"
            self._counter += 1

        return Tensor(
            data=data,
            meta=TensorMeta(shape=shape, dtype=dtype, name=name),
        )

    def gen_ones(
        self,
        shape: Tuple[int, ...],
        dtype: Union[str, DType] = DType.FP32,
        name: Optional[str] = None,
    ) -> Tensor:
        """生成全一数据"""
        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)

        data = np.ones(shape, dtype=np.float32)

        if name is None:
            name = f"ones_{self._counter}"
            self._counter += 1

        return Tensor(
            data=data,
            meta=TensorMeta(shape=shape, dtype=dtype, name=name),
        )

    def _gen_data(
        self, shape: Tuple[int, ...], dist: DistType, **kwargs
    ) -> np.ndarray:
        """根据分布生成数据"""
        if dist == DistType.NORMAL:
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            data = self._rng.normal(mean, std, shape).astype(np.float32)

        elif dist == DistType.UNIFORM:
            low = kwargs.get("low", -1.0)
            high = kwargs.get("high", 1.0)
            data = self._rng.uniform(low, high, shape).astype(np.float32)

        elif dist == DistType.ZEROS:
            data = np.zeros(shape, dtype=np.float32)

        elif dist == DistType.ONES:
            data = np.ones(shape, dtype=np.float32)

        elif dist == DistType.XAVIER:
            # Xavier/Glorot 初始化
            fan_in = shape[-1] if len(shape) >= 1 else 1
            fan_out = shape[0] if len(shape) >= 2 else 1
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            data = self._rng.uniform(-limit, limit, shape).astype(np.float32)

        elif dist == DistType.KAIMING:
            # Kaiming/He 初始化
            fan_in = shape[-1] if len(shape) >= 1 else 1
            std = np.sqrt(2.0 / fan_in)
            data = self._rng.normal(0, std, shape).astype(np.float32)

        else:
            raise ValueError(f"Unknown distribution: {dist}")

        return data
