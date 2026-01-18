"""算子基础框架

设计说明：
- 调用算子时，同时执行 golden 和 reference 两种实现
- golden: 用户注册的精确实现（如 C++ binding）
- reference: 内置的 numpy 参考实现
- 两者的输出分别保存，供后续比对
"""
import numpy as np
from typing import Callable, Dict, Any, List
from pathlib import Path

from aidevtools.core.log import logger

# Golden 实现注册表
_golden_registry: Dict[str, Callable] = {}

# 记录列表
_records: List[Dict[str, Any]] = []
_counter: Dict[str, int] = {}


def register_golden(name: str):
    """
    注册 Golden 实现（用户的精确实现，如 C++ binding）

    示例:
        from my_cpp_lib import cpp_linear

        @register_golden("linear")
        def golden_linear(x, weight, bias=None):
            return cpp_linear(x, weight, bias)
    """
    def decorator(func: Callable):
        _golden_registry[name] = func
        logger.info(f"注册 Golden 实现: {name}")
        return func
    return decorator


def has_golden(name: str) -> bool:
    """检查是否有 Golden 实现"""
    return name in _golden_registry


def get_records() -> List[Dict[str, Any]]:
    """获取所有记录"""
    return _records


def clear():
    """清空记录"""
    _records.clear()
    _counter.clear()


class Op:
    """
    算子基类

    调用时同时执行:
    1. reference 实现 (numpy) -> 保存为 golden（参考标准）
    2. golden 实现 (用户注册) -> 保存为 result（待验证结果）

    如果未注册 golden，则只执行 reference，result 留空待后续填充。
    """
    name: str = None  # 算子名，子类必须定义

    def __init__(self):
        if self.name is None:
            raise ValueError("算子必须定义 name 属性")

    def reference(self, *args, **kwargs) -> np.ndarray:
        """
        参考实现（numpy），子类必须实现

        这是标准答案，用于比对
        """
        raise NotImplementedError(f"{self.name} 未实现 reference 方法")

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        调用算子

        同时执行 reference 和 golden（如已注册），记录两者输出
        返回 reference 的结果（作为流程中的数据流）
        """
        # 计数
        idx = _counter.get(self.name, 0)
        _counter[self.name] = idx + 1
        full_name = f"{self.name}_{idx}"

        # 执行 reference（numpy 参考实现）
        ref_output = self.reference(*args, **kwargs)

        # 执行 golden（如已注册）
        golden_output = None
        if has_golden(self.name):
            try:
                golden_impl = _golden_registry[self.name]
                golden_output = golden_impl(*args, **kwargs)
                logger.debug(f"{full_name}: golden 执行完成")
            except Exception as e:
                logger.warn(f"{full_name}: golden 执行失败 - {e}")

        # 记录
        record = {
            "name": full_name,
            "op": self.name,
            "input": args[0] if args else None,
            "weight": args[1] if len(args) > 1 else kwargs.get("weight"),
            "golden": ref_output,      # reference 输出作为 golden（标准答案）
            "result": golden_output,   # 用户 golden 实现的输出作为 result（待验证）
        }
        _records.append(record)

        # 返回 reference 结果，保持数据流
        return ref_output

    def __repr__(self):
        has_g = "✓" if has_golden(self.name) else "✗"
        return f"<Op {self.name} golden={has_g}>"


def _is_array_like(obj):
    """检查是否为数组类型"""
    return isinstance(obj, np.ndarray) or (hasattr(obj, '__array__') and not isinstance(obj, (dict, list)))


def dump(output_dir: str = "./workspace", format: str = "raw"):
    """
    导出所有记录的数据

    Args:
        output_dir: 输出目录
        format: 数据格式 ("raw", "npy", "npz")
    """
    from aidevtools.formats.base import save as save_data

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for r in _records:
        name = r["name"]
        # 保存 golden（reference 输出，标准答案）
        if r["golden"] is not None and _is_array_like(r["golden"]):
            save_data(str(path / f"{name}_golden.bin"), np.asarray(r["golden"]), format=format)
        # 保存 result（用户 golden 实现的输出）
        if r["result"] is not None and _is_array_like(r["result"]):
            save_data(str(path / f"{name}_result.bin"), np.asarray(r["result"]), format=format)
        # 保存输入
        if r["input"] is not None and _is_array_like(r["input"]):
            save_data(str(path / f"{name}_input.bin"), np.asarray(r["input"]), format=format)
        # 保存权重
        if r["weight"] is not None and _is_array_like(r["weight"]):
            save_data(str(path / f"{name}_weight.bin"), np.asarray(r["weight"]), format=format)
        logger.info(f"dump: {name}")


def gen_csv(output_dir: str = "./workspace", model_name: str = "model") -> str:
    """
    生成 compare.csv

    Args:
        output_dir: 输出目录
        model_name: 模型名称

    Returns:
        CSV 文件路径
    """
    import csv

    path = Path(output_dir)

    # 文件名去重
    csv_path = path / f"{model_name}_compare.csv"
    i = 1
    while csv_path.exists():
        csv_path = path / f"{model_name}_compare_{i:03d}.csv"
        i += 1

    # 写 CSV
    fields = ["op_name", "mode", "input_bin", "weight_bin", "golden_bin",
              "result_bin", "dtype", "shape", "qtype", "skip", "note"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in _records:
            name = r["name"]
            golden = np.asarray(r["golden"]) if r["golden"] is not None else None
            has_result = r["result"] is not None

            row = {
                "op_name": name,
                "mode": "single",
                "input_bin": str(path / f"{name}_input.bin") if r["input"] is not None else "",
                "weight_bin": str(path / f"{name}_weight.bin") if r["weight"] is not None else "",
                "golden_bin": str(path / f"{name}_golden.bin"),
                "result_bin": str(path / f"{name}_result.bin") if has_result else "",
                "dtype": str(golden.dtype) if golden is not None else "",
                "shape": ",".join(str(d) for d in golden.shape) if golden is not None else "",
                "qtype": "",  # 空为精确比对
                "skip": "false",
                "note": "" if has_result else "result 待填充",
            }
            writer.writerow(row)

    logger.info(f"生成 compare 表: {csv_path}")
    return str(csv_path)
