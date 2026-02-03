"""算子注册表 (内部模块)

将注册表数据结构独立出来避免循环导入：
- registry.py 定义 OpMeta 和 register_op 装饰器，使用这里的注册表
- base.py 通过这里获取 op 元信息，避免直接导入 registry.py
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type


@dataclass
class OpMeta:
    """算子元信息"""

    name: str
    inputs: List[str] = field(default_factory=lambda: ["x"])
    optional: List[str] = field(default_factory=list)
    description: str = ""
    has_cpp_golden: bool = False
    # auto API 参数生成配置
    auto_gen: Dict[str, str] = field(default_factory=dict)
    # 算子类引用 (运行时填充)
    op_class: Optional[Type] = None
    # 算子实例 (运行时填充)
    op_instance: Optional[Any] = None

    # ============================================================
    # Profile 相关配置 (用于 Paper Analysis)
    # ============================================================
    compute_unit: str = "vector"  # "cube" | "vector"
    memory_pattern: str = "sequential"  # "sequential" | "strided" | "random"
    # FLOPs 计算函数: flops_fn(shapes) -> int
    # shapes 是从输入推断的形状字典
    flops_fn: Optional[Callable] = None
    # 权重参数名 (用于区分 input_bytes 和 weight_bytes)
    weight_params: List[str] = field(default_factory=list)


# 全局注册表
_op_registry: Dict[str, OpMeta] = {}


def register_op_meta(name: str, meta: OpMeta) -> None:
    """注册算子元信息"""
    _op_registry[name] = meta


def get_op_meta(name: str) -> Optional[OpMeta]:
    """获取算子元信息"""
    return _op_registry.get(name)


def get_all_op_names() -> List[str]:
    """获取所有已注册算子名"""
    return list(_op_registry.keys())


def get_op_registry() -> Dict[str, OpMeta]:
    """获取完整注册表 (只读)"""
    return _op_registry.copy()
