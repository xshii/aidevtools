"""统一算子注册表

单一定义源：算子只需在一处定义，自动注册到所有需要的地方。

用法:
    from aidevtools.ops.registry import register_op, Op

    @register_op(
        inputs=["x", "weight"],
        optional=["bias"],
        description="线性变换 y = x @ weight + bias",
    )
    class Linear(Op):
        name = "linear"

        def golden_python(self, x, weight, bias=None):
            ...

        def reference(self, x, weight, bias=None):
            ...
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Type, Callable
import numpy as np

from aidevtools.core.log import logger


# ============================================================
# 算子元信息
# ============================================================

@dataclass
class OpMeta:
    """算子元信息"""
    name: str
    inputs: List[str] = field(default_factory=lambda: ["x"])
    optional: List[str] = field(default_factory=list)
    description: str = ""
    has_cpp_golden: bool = False
    # 算子类引用 (运行时填充)
    op_class: Optional[Type] = None
    # 算子实例 (运行时填充)
    op_instance: Optional[Any] = None


# ============================================================
# 全局注册表
# ============================================================

_op_registry: Dict[str, OpMeta] = {}


def register_op(
    inputs: List[str] = None,
    optional: List[str] = None,
    description: str = "",
    has_cpp_golden: bool = False,
):
    """
    算子注册装饰器

    Args:
        inputs: 必需输入参数名列表
        optional: 可选输入参数名列表
        description: 算子描述
        has_cpp_golden: 是否有 C++ golden 实现

    Returns:
        类装饰器

    Example:
        @register_op(
            inputs=["x", "weight"],
            optional=["bias"],
            description="线性变换",
        )
        class Linear(Op):
            name = "linear"
            ...
    """
    def decorator(cls: Type) -> Type:
        # 获取算子名
        if not hasattr(cls, 'name') or cls.name is None:
            raise ValueError(f"算子类 {cls.__name__} 必须定义 name 属性")

        op_name = cls.name

        # 创建元信息
        meta = OpMeta(
            name=op_name,
            inputs=inputs or ["x"],
            optional=optional or [],
            description=description or f"{op_name} 算子",
            has_cpp_golden=has_cpp_golden,
            op_class=cls,
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
    from aidevtools.ops.base import Op

    result = {}
    for name in get_cpp_golden_ops():
        meta = _op_registry.get(name)
        if meta and meta.op_class:
            # 检查类是否实现了 cpu_golden 方法
            has_method = (
                hasattr(meta.op_class, 'cpu_golden') and
                meta.op_class.cpu_golden is not Op.cpu_golden
            )
            result[name] = has_method
        else:
            result[name] = False
    return result
