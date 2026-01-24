"""通用工具函数

提供 shape、dtype 等常用解析函数，消除代码重复。
"""
from typing import Optional, Tuple

import numpy as np


def parse_shape(shape_str: str) -> Optional[Tuple[int, ...]]:
    """
    解析 shape 字符串为 tuple

    Args:
        shape_str: 形如 "1,64,32,32" 的字符串

    Returns:
        shape tuple，如 (1, 64, 32, 32)；空字符串返回 None

    Examples:
        >>> parse_shape("1,64,32,32")
        (1, 64, 32, 32)
        >>> parse_shape("1, 64, 32")  # 带空格
        (1, 64, 32)
        >>> parse_shape("")
        None
    """
    if not shape_str or not shape_str.strip():
        return None
    return tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())


def parse_dtype(dtype_str: str) -> np.dtype:
    """
    解析 dtype 字符串为 numpy dtype

    Args:
        dtype_str: dtype 名称，如 "float32", "float16", "int32"

    Returns:
        numpy dtype 对象

    Raises:
        AttributeError: 如果 dtype 名称无效

    Examples:
        >>> parse_dtype("float32")
        dtype('float32')
        >>> parse_dtype("float16")
        dtype('float16')
    """
    return getattr(np, dtype_str)


def parse_list(list_str: str, separator: str = ",") -> list:
    """
    解析逗号分隔的列表字符串

    Args:
        list_str: 形如 "a,b,c" 的字符串
        separator: 分隔符，默认逗号

    Returns:
        列表，空字符串返回空列表

    Examples:
        >>> parse_list("a, b, c")
        ['a', 'b', 'c']
        >>> parse_list("")
        []
    """
    if not list_str or not list_str.strip():
        return []
    return [x.strip() for x in list_str.split(separator) if x.strip()]


def format_shape(shape: Tuple[int, ...]) -> str:
    """
    将 shape tuple 格式化为字符串

    Args:
        shape: shape tuple

    Returns:
        格式化的字符串

    Examples:
        >>> format_shape((1, 64, 32, 32))
        '1,64,32,32'
    """
    return ",".join(str(x) for x in shape)


def safe_getattr(module, name: str, default=None):
    """
    安全获取属性，失败返回默认值

    Args:
        module: 模块或对象
        name: 属性名
        default: 默认值

    Returns:
        属性值或默认值
    """
    try:
        return getattr(module, name)
    except AttributeError:
        return default
