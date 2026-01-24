"""Trace 装饰器"""
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.base import save as save_data

_records: List[Dict[str, Any]] = []
_counter: Dict[str, int] = {}

def trace(fn=None, *, name: str = None, save_input: bool = True):  # pylint: disable=unused-argument
    """
    插桩装饰器，记录函数输入输出

    用法:
        @trace
        def conv2d(x, weight):
            ...

        @trace(name="my_conv")
        def conv2d(x, weight):
            ...
    """
    def decorator(func):
        op_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 计数
            idx = _counter.get(op_name, 0)
            _counter[op_name] = idx + 1
            full_name = f"{op_name}_{idx}"

            # 执行
            logger.debug(f"trace: {full_name} 开始")
            output = func(*args, **kwargs)
            logger.debug(f"trace: {full_name} 完成")

            # 记录
            record = {
                "name": full_name,
                "op": op_name,
                "input": args[0] if args else None,
                "weight": args[1] if len(args) > 1 else None,
                "output": output,
            }
            _records.append(record)
            return output
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator

def _is_array_like(obj):
    """检查是否为数组类型"""
    return isinstance(obj, np.ndarray) or (hasattr(obj, '__array__') and not isinstance(obj, dict))


def dump(output_dir: str = "./workspace", fmt: str = "raw"):
    """导出所有记录"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for r in _records:
        name = r["name"]
        # 保存输出 (golden)
        if r["output"] is not None and _is_array_like(r["output"]):
            save_data(str(path / f"{name}_golden.bin"), np.asarray(r["output"]), fmt=fmt)
        # 保存输入
        if r["input"] is not None and _is_array_like(r["input"]):
            save_data(str(path / f"{name}_input.bin"), np.asarray(r["input"]), fmt=fmt)
        # 保存权重
        if r["weight"] is not None and _is_array_like(r["weight"]):
            save_data(str(path / f"{name}_weight.bin"), np.asarray(r["weight"]), fmt=fmt)
        logger.info(f"dump: {name}")


def clear():
    """清空记录"""
    _records.clear()
    _counter.clear()
