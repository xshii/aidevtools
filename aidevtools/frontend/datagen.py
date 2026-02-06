"""
数据生成器

提供输入和权重数据的生成功能。

支持两种使用方式:
1. 手动指定: gen.gen_input(shape, dtype)
2. 自动生成: gen.gen_op("linear", input_shape, out_features=3072)
"""

from typing import Any, Dict, Optional, Tuple, Union

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
        """根据 auto_gen 策略生成数据"""
        input_shape = context.get("input_shape", (1,))

        # 简写策略
        if strategy == "input":
            shape = input_shape
            data = self._rng.standard_normal(shape).astype(np.float32)

        elif strategy == "random":
            shape = input_shape
            data = self._rng.standard_normal(shape).astype(np.float32)

        elif strategy == "xavier":
            out_features = context.get("out_features", input_shape[-1])
            in_features = input_shape[-1]
            shape = (out_features, in_features)
            fan_in, fan_out = in_features, out_features
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            data = self._rng.uniform(-limit, limit, shape).astype(np.float32)

        elif strategy == "kaiming":
            out_features = context.get("out_features", input_shape[-1])
            in_features = input_shape[-1]
            shape = (out_features, in_features)
            std = np.sqrt(2.0 / in_features)
            data = self._rng.normal(0, std, shape).astype(np.float32)

        elif strategy == "uniform":
            shape = (context.get("out_features", input_shape[-1]),)
            data = self._rng.uniform(-0.1, 0.1, shape).astype(np.float32)

        # 带参数的策略
        elif strategy.startswith("zeros:"):
            shape = self._parse_shape(strategy[6:], context)
            data = np.zeros(shape, dtype=np.float32)

        elif strategy.startswith("ones:"):
            shape = self._parse_shape(strategy[5:], context)
            data = np.ones(shape, dtype=np.float32)

        elif strategy.startswith("xavier:"):
            shape = self._parse_shape(strategy[7:], context)
            fan_in = shape[-1] if len(shape) >= 1 else 1
            fan_out = shape[0] if len(shape) >= 2 else 1
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            data = self._rng.uniform(-limit, limit, shape).astype(np.float32)

        elif strategy.startswith("kaiming:"):
            shape = self._parse_shape(strategy[8:], context)
            fan_in = shape[-1] if len(shape) >= 1 else 1
            std = np.sqrt(2.0 / fan_in)
            data = self._rng.normal(0, std, shape).astype(np.float32)

        elif strategy.startswith("uniform:"):
            parts = strategy[8:].split(",")
            if len(parts) >= 3:
                low, high = float(parts[0]), float(parts[1])
                shape_spec = ",".join(parts[2:])
            else:
                low, high = -0.1, 0.1
                shape_spec = ",".join(parts)
            shape = self._parse_shape(shape_spec, context)
            data = self._rng.uniform(low, high, shape).astype(np.float32)

        elif strategy.startswith("same:"):
            ref_param = strategy[5:]
            ref_shape = context.get(f"{ref_param}_shape")
            if ref_shape is None:
                raise ValueError(f"same:{ref_param} 引用的参数未生成")
            shape = ref_shape
            data = self._rng.standard_normal(shape).astype(np.float32)

        elif strategy.startswith("normal:"):
            parts = strategy[7:].split(",")
            if len(parts) >= 2:
                mean, std = float(parts[0]), float(parts[1])
                shape_spec = ",".join(parts[2:]) if len(parts) > 2 else "-1"
            else:
                mean, std = 0.0, 1.0
                shape_spec = parts[0]
            shape = self._parse_shape(shape_spec, context)
            data = self._rng.normal(mean, std, shape).astype(np.float32)

        else:
            raise ValueError(f"未知生成策略: {strategy}")

        return data, shape

    def _parse_shape(
        self, spec: str, context: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """解析 shape 规格"""
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
