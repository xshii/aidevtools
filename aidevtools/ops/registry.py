"""统一算子注册表

单一定义源：算子只需在一处定义，自动注册到所有需要的地方。

用法:
    from aidevtools.ops.registry import register_op, Op

    @register_op(
        inputs=["x", "weight"],
        optional=["bias"],
        description="线性变换 y = x @ weight + bias",
        auto_gen={
            "x": "input",           # 主输入，可以是 shape 或 array
            "weight": "xavier:-1,out_features",  # Xavier 初始化，shape=[x.shape[-1], out_features]
            "bias": "zeros:out_features",        # 零初始化
        },
    )
    class Linear(Op):
        name = "linear"

        def golden_python(self, x, weight, bias=None):
            ...

        def reference(self, x, weight, bias=None):
            ...

auto_gen 策略说明:
    - "input": 主输入，可以是 tuple(shape) 或 ndarray
    - "random": 随机初始化
    - "ones:dim": 全1数组，dim 指定从输入推断 shape 的维度 (-1 表示最后一维)
    - "zeros:dim": 全0数组
    - "xavier:in_dim,out_dim": Xavier 初始化，需要指定输入输出维度
    - "same:param": 与另一个参数相同 shape
"""
from typing import Any, Callable, Dict, List, Optional, Type

from aidevtools.core.log import logger
from aidevtools.ops._op_registry import OpMeta, _op_registry


def register_op(
    inputs: List[str] = None,
    optional: List[str] = None,
    description: str = "",
    has_cpp_golden: bool = False,
    auto_gen: Dict[str, str] = None,
    # Profile 配置 (用于 Paper Analysis)
    compute_unit: str = "vector",
    memory_pattern: str = "sequential",
    flops_fn: Callable = None,
    weight_params: List[str] = None,
):
    """
    算子注册装饰器

    Args:
        inputs: 必需输入参数名列表
        optional: 可选输入参数名列表
        description: 算子描述
        has_cpp_golden: 是否有 C++ golden 实现
        auto_gen: auto API 参数生成配置
        compute_unit: 计算单元 ("cube" | "vector")，用于 Paper Analysis
        memory_pattern: 访存模式 ("sequential" | "strided" | "random")
        flops_fn: FLOPs 计算函数 flops_fn(shapes_dict) -> int
        weight_params: 权重参数名列表 (用于区分 input_bytes 和 weight_bytes)

    Returns:
        类装饰器

    Example:
        @register_op(
            inputs=["x", "gamma", "beta"],
            optional=["eps"],
            description="Layer Normalization",
            auto_gen={
                "x": "input",
                "gamma": "ones:-1",
                "beta": "zeros:-1",
            },
            compute_unit="vector",
            flops_fn=lambda s: 8 * s["x_size"],  # ~8 ops/element
            weight_params=["gamma", "beta"],
        )
        class LayerNorm(Op):
            name = "layernorm"
            ...
    """
    def decorator(cls: Type) -> Type:
        # 获取算子名
        if not hasattr(cls, 'name') or cls.name is None:
            raise ValueError(f"算子类 {cls.__name__} 必须定义 name 属性")

        op_name = cls.name

        # 默认 auto_gen: 第一个输入为 input，其他为 random
        default_auto_gen = {}
        input_list = inputs or ["x"]
        for i, inp in enumerate(input_list):
            if i == 0:
                default_auto_gen[inp] = "input"
            else:
                default_auto_gen[inp] = "random"

        # 如果没有提供 flops_fn，检查类是否有自定义的 compute_flops 方法
        actual_flops_fn = flops_fn
        if actual_flops_fn is None:
            # 检查是否是自定义实现（不是基类的默认实现）
            from aidevtools.ops.base import is_compute_flops_overridden
            if is_compute_flops_overridden(cls):
                actual_flops_fn = cls.compute_flops

        # 创建元信息
        meta = OpMeta(
            name=op_name,
            inputs=input_list,
            optional=optional or [],
            description=description or f"{op_name} 算子",
            has_cpp_golden=has_cpp_golden,
            auto_gen=auto_gen or default_auto_gen,
            op_class=cls,
            # Profile 配置
            compute_unit=compute_unit,
            memory_pattern=memory_pattern,
            flops_fn=actual_flops_fn,
            weight_params=weight_params or [],
        )

        # 注册到全局表
        _op_registry[op_name] = meta
        logger.debug(f"注册算子: {op_name}")

        return cls

    return decorator


# ============================================================
# 注册表查询 API
# ============================================================

def get_op_meta(name: str) -> Optional[OpMeta]:
    """获取算子元信息"""
    return _op_registry.get(name)


def get_op_info(name: str) -> Dict[str, Any]:
    """
    获取算子信息 (兼容旧 API)

    Returns:
        {"inputs": [...], "optional": [...], "description": "..."}
    """
    meta = _op_registry.get(name)
    if meta is None:
        # 返回默认值 (兼容未注册的算子)
        return {
            "inputs": ["x"],
            "optional": [],
            "description": f"自定义算子: {name}",
        }
    return {
        "inputs": meta.inputs,
        "optional": meta.optional,
        "description": meta.description,
    }


def list_ops() -> List[str]:
    """列出所有注册的算子"""
    return list(_op_registry.keys())


def validate_op(name: str) -> bool:
    """检查算子是否有效"""
    return name in _op_registry


def get_op_instance(name: str) -> Optional[Any]:
    """
    获取算子实例

    如果实例不存在，会自动创建
    """
    meta = _op_registry.get(name)
    if meta is None:
        return None

    if meta.op_instance is None and meta.op_class is not None:
        meta.op_instance = meta.op_class()

    return meta.op_instance


def get_all_ops() -> Dict[str, OpMeta]:
    """获取所有注册的算子"""
    return _op_registry.copy()


def get_cpp_golden_ops() -> List[str]:
    """
    获取所有标记为有 C++ golden 实现的算子

    Returns:
        算子名列表

    用法:
        from aidevtools.ops.registry import get_cpp_golden_ops

        # 获取应该有 C++ golden 的算子
        cpp_ops = get_cpp_golden_ops()
        # ['matmul', 'softmax', 'layernorm', 'transpose']
    """
    return [name for name, meta in _op_registry.items() if meta.has_cpp_golden]


def check_cpp_golden_registered() -> Dict[str, bool]:
    """
    检查 C++ golden 注册状态

    Returns:
        {算子名: 是否已实现 cpu_golden 方法}

    用法:
        from aidevtools.ops.registry import check_cpp_golden_registered

        status = check_cpp_golden_registered()
        # {'matmul': True, 'softmax': True, 'layernorm': True, ...}
    """
    from aidevtools.ops.base import is_cpu_golden_overridden

    result = {}
    for name in get_cpp_golden_ops():
        meta = _op_registry.get(name)
        if meta and meta.op_class:
            # 检查类是否实现了 cpu_golden 方法
            result[name] = is_cpu_golden_overridden(meta.op_class)
        else:
            result[name] = False
    return result
