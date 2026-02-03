"""算子注册表

从统一注册表 (ops.registry) 获取算子信息。
为了向后兼容，保留原有 API。

注意：算子定义现在集中在 ops/_functional.py，使用 @register_op 装饰器。
"""
from typing import Any, Dict

# 导入 _functional 模块以触发算子注册
from aidevtools.ops import _functional  # noqa: F401

# 从统一注册表导入
from aidevtools.ops.registry import (
    get_op_info,
    list_ops,
    validate_op,
)

# 额外的算子定义（不在 _functional.py 中实现的）
# 这些算子仅用于 xlsx 模板，可能没有对应的 Python 实现
_extra_ops: Dict[str, Dict[str, Any]] = {
    "conv2d": {
        "inputs": ["x", "weight"],
        "optional": ["bias", "stride", "padding"],
        "description": "2D 卷积",
    },
    "reshape": {
        "inputs": ["x"],
        "optional": ["shape"],
        "description": "形状变换",
    },
    "concat": {
        "inputs": ["tensors"],
        "optional": ["axis"],
        "description": "张量拼接",
    },
    "split": {
        "inputs": ["x"],
        "optional": ["num_splits", "axis"],
        "description": "张量分割",
    },
    "pooling": {
        "inputs": ["x"],
        "optional": ["kernel_size", "stride", "mode"],
        "description": "池化 (max/avg)",
    },
}


def get_default_ops() -> Dict[str, Dict[str, Any]]:
    """
    获取默认算子注册表

    Returns:
        合并后的算子注册表 (统一注册表 + 额外算子)
    """
    # 从统一注册表获取所有算子
    result = {}
    for name in list_ops():
        result[name] = get_op_info(name)

    # 添加额外的算子
    for name, info in _extra_ops.items():
        if name not in result:
            result[name] = info

    return result


# 重新导出 API (向后兼容)
__all__ = [
    "get_op_info",
    "list_ops",
    "validate_op",
    "get_default_ops",
]
