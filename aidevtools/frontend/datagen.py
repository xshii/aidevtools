"""
数据生成器

提供输入和权重数据的生成功能。

支持两种使用方式:
1. 手动指定: gen.gen_input(shape, dtype)
2. 自动生成: gen.gen_op("linear", input_shape, out_features=3072)
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from aidevtools.core.random import RandomGenerator
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
        self._rand = RandomGenerator(seed)
        self._counter = 0

    def reset(self, seed: Optional[int] = None):
        """重置生成器"""
        if seed is not None:
            self.seed = seed
        self._rand.reset(self.seed)
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

    def gen_op(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        dtype: Union[str, DType] = DType.FP32,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        根据 @register_op 的 auto_gen 配置自动生成算子所需的所有数据

        Args:
            op_name: 算子名称 (必须已通过 @register_op 注册)
            input_shape: 主输入 shape
            dtype: 数据类型
            **kwargs: 额外参数 (如 out_features, num_heads 等)

        Returns:
            Dict[param_name, Tensor]

        Example:
            gen = DataGenerator(seed=42)

            # 自动生成 linear 所需的 input, weight, bias
            data = gen.gen_op("linear", input_shape=(512, 768), out_features=3072)
            # data = {
            #     "input": Tensor(...),
            #     "weight": Tensor(...),
            #     "bias": Tensor(...),
            # }

            # 自动生成 layernorm 所需的 x, gamma, beta
            data = gen.gen_op("layernorm", input_shape=(512, 768))
        """
        from aidevtools.ops.registry import get_op_meta

        if isinstance(dtype, str):
            dtype = DType.from_str(dtype)

        meta = get_op_meta(op_name)
        if meta is None:
            raise ValueError(f"算子 '{op_name}' 未注册，请使用 @register_op 注册")

        result = {}
        context = {"input_shape": input_shape, **kwargs}

        for param, strategy in meta.auto_gen.items():
            is_weight = param in meta.weight_params

            # 解析策略，生成数据
            data, shape = self._gen_from_strategy(strategy, context, is_weight)

            # 创建 Tensor
            tensor = Tensor(
                data=data,
                meta=TensorMeta(shape=shape, dtype=dtype, name=f"{op_name}.{param}"),
            )
            result[param] = tensor

            # 更新 context
            context[f"{param}_shape"] = shape
            context[param] = data

        return result

    def gen_op_with_golden(
        self,
        op_name: str,
        input_shape: Tuple[int, ...],
        dtype: Union[str, DType] = DType.FP32,
        **kwargs,
    ) -> Tuple[Dict[str, Tensor], np.ndarray]:
        """
        生成算子数据并计算 golden 输出

        Args:
            op_name: 算子名称
            input_shape: 主输入 shape
            dtype: 数据类型
            **kwargs: 额外参数

        Returns:
            (data_dict, golden_output)

        Example:
            gen = DataGenerator(seed=42)
            data, golden = gen.gen_op_with_golden(
                "linear",
                input_shape=(512, 768),
                out_features=3072,
            )
            print(f"golden: {golden.shape}")  # (512, 3072)
        """
        from aidevtools.ops.registry import get_op_instance, get_op_meta

        # 生成数据
        data = self.gen_op(op_name, input_shape, dtype, **kwargs)

        # 获取算子实例
        op = get_op_instance(op_name)
        if op is None:
            raise ValueError(f"算子 '{op_name}' 无法实例化")

        meta = get_op_meta(op_name)

        # 构建参数
        args = []
        kwargs_call = {}
        for inp in meta.inputs:
            if inp in data:
                args.append(data[inp].data)
        for opt in meta.optional:
            if opt in data:
                kwargs_call[opt] = data[opt].data

        # 计算 golden
        golden = op.cpu_golden(*args, **kwargs_call)

        return data, golden

    def _gen_from_strategy(
        self,
        strategy: str,
        context: Dict[str, Any],
        is_weight: bool,
    ) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """根据 auto_gen 策略生成数据（委托给 RandomGenerator）"""
        return self._rand.generate_from_strategy(strategy, context)

    def _gen_data(
        self, shape: Tuple[int, ...], dist: Union[str, DistType], **kwargs
    ) -> np.ndarray:
        """根据分布生成数据（委托给 RandomGenerator）"""
        method = dist.value if isinstance(dist, DistType) else dist
        return self._rand.generate(shape, method=method, **kwargs)
