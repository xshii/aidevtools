"""Trace 装饰器"""
import csv
from pathlib import Path
from functools import wraps
from typing import List, Dict, Any
import numpy as np

from aidevtools.core.log import logger
from aidevtools.formats.base import save as save_data

_records: List[Dict[str, Any]] = []
_counter: Dict[str, int] = {}

def trace(fn=None, *, name: str = None, save_input: bool = True):
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

def dump(output_dir: str = "./workspace", format: str = "raw"):
    """导出所有记录"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for r in _records:
        name = r["name"]
        # 保存输出 (golden)
        if r["output"] is not None:
            save_data(str(path / f"{name}_golden.bin"), np.asarray(r["output"]), format=format)
        # 保存输入
        if r["input"] is not None:
            save_data(str(path / f"{name}_input.bin"), np.asarray(r["input"]), format=format)
        # 保存权重
        if r["weight"] is not None:
            save_data(str(path / f"{name}_weight.bin"), np.asarray(r["weight"]), format=format)
        logger.info(f"dump: {name}")

def gen_csv(output_dir: str = "./workspace", model_name: str = "model") -> str:
    """生成 compare.csv"""
    path = Path(output_dir)

    # 文件名去重
    csv_path = path / f"{model_name}_compare.csv"
    i = 1
    while csv_path.exists():
        csv_path = path / f"{model_name}_compare_{i:03d}.csv"
        i += 1

    # 写 CSV
    fields = ["op_name", "mode", "input_bin", "weight_bin", "golden_bin",
              "result_bin", "skip", "status", "max_abs", "qsnr", "detail_link", "note"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in _records:
            name = r["name"]
            row = {
                "op_name": name,
                "mode": "single",
                "input_bin": str(path / f"{name}_input.bin") if r["input"] is not None else "",
                "weight_bin": str(path / f"{name}_weight.bin") if r["weight"] is not None else "",
                "golden_bin": str(path / f"{name}_golden.bin"),
                "result_bin": str(path / f"{name}_sim_out.bin"),  # 假定命名
                "skip": "false",
                "status": "",
                "max_abs": "",
                "qsnr": "",
                "detail_link": "",
                "note": "",
            }
            writer.writerow(row)

    logger.info(f"生成 compare 表: {csv_path}")
    return str(csv_path)

def clear():
    """清空记录"""
    _records.clear()
    _counter.clear()
