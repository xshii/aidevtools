"""算子基础框架

设计说明：
- 每个算子包含3种计算形式：
  1. cpu_golden: C++ Golden 实现（子类方法，has_cpp_golden=True 的算子）
  2. golden_python: Python Golden 实现（子类实现）
  3. reference: 高精度参考实现（numpy fp32/fp64，用于 fuzzy 比对）

- 调用算子时，根据全局配置 golden_mode 选择执行哪种 golden：
  - golden_mode="cpp": 使用 cpu_golden 方法（如果存在）
  - golden_mode="python": 使用 golden_python

- reference 始终执行，用于 fuzzy 比对
"""
import numpy as np
from typing import Callable, Dict, Any, List, Optional
from pathlib import Path

from aidevtools.core.log import logger
from aidevtools.core.config import get_config, set_config

# Golden 实现注册表 (C++ bindings)
_golden_cpp_registry: Dict[str, Callable] = {}

# 记录列表
_records: List[Dict[str, Any]] = []
_counter: Dict[str, int] = {}


def set_golden_mode(mode: str) -> None:
    """
    设置 Golden 模式

    Args:
        mode: "cpp" | "python" | "none"
            - "cpp": 使用注册的 C++ golden 实现
            - "python": 使用内置的 Python golden 实现
            - "none": 不计算 golden（golden 在外部计算）

    注意: 也可使用 set_config(golden_mode=...) 统一设置
    """
    if mode not in ("cpp", "python", "none"):
        raise ValueError(f"golden_mode 必须是 'cpp', 'python' 或 'none'，而不是 '{mode}'")
    if mode == "none":
        set_config(golden_mode="python", compute_golden=False)
    else:
        set_config(golden_mode=mode, compute_golden=True)
    logger.info(f"设置 golden_mode = {mode}")


def set_compute_golden(enabled: bool) -> None:
    """
    设置是否执行 golden 计算

    Args:
        enabled: True=执行本地 golden 计算, False=跳过（golden 在外部计算）

    注意: 也可使用 set_config(compute_golden=...) 统一设置
    """
    set_config(compute_golden=enabled)
    logger.info(f"设置 compute_golden = {enabled}")


def get_compute_golden() -> bool:
    """获取是否执行 golden 计算"""
    return get_config().compute_golden


def get_golden_mode() -> str:
    """获取当前 Golden 模式"""
    return get_config().golden_mode


def register_golden_cpp(name: str) -> Callable[[Callable], Callable]:
    """
    注册 C++ Golden 实现

    示例:
        from my_cpp_lib import cpp_linear

        @register_golden_cpp("linear")
        def golden_linear(x, weight, bias=None):
            return cpp_linear(x, weight, bias)
    """
    def decorator(func: Callable) -> Callable:
        _golden_cpp_registry[name] = func
        logger.info(f"注册 C++ Golden 实现: {name}")
        return func
    return decorator


def has_golden_cpp(name: str) -> bool:
    """检查是否有 C++ Golden 实现"""
    return name in _golden_cpp_registry


def get_records() -> List[Dict[str, Any]]:
    """获取所有记录"""
    return _records


def clear() -> None:
    """清空记录"""
    _records.clear()
    _counter.clear()


class Op:
    """
    算子基类

    子类必须实现：
    - name: 算子名称
    - golden_python(): Python Golden 实现
    - reference(): 高精度参考实现（用于 fuzzy 比对）

    可选：
    - 通过 @register_golden_cpp 注册 C++ Golden 实现

    调用时：
    1. 根据 golden_mode 执行 golden_cpp 或 golden_python -> 保存为 golden
    2. 执行 reference -> 保存为 reference（用于 fuzzy 比对）
    3. 返回 golden 的结果（作为数据流）
    """
    name: str = None  # 算子名，子类必须定义

    def __init__(self):
        if self.name is None:
            raise ValueError("算子必须定义 name 属性")

    def golden_python(self, *args, **kwargs) -> np.ndarray:
        """
        Python Golden 实现，子类必须实现

        这是 Python 版本的精确实现
        """
        raise NotImplementedError(f"{self.name} 未实现 golden_python 方法")

    def reference(self, *args, **kwargs) -> np.ndarray:
        """
        高精度参考实现（numpy fp32/fp64）

        用于 fuzzy 比对，子类必须实现
        """
        raise NotImplementedError(f"{self.name} 未实现 reference 方法")

    def cpu_golden(self, *args, **kwargs) -> np.ndarray:
        """
        C++ Golden 实现，有 has_cpp_golden=True 的子类可实现
        """
        raise NotImplementedError(f"{self.name} 未实现 cpu_golden 方法")

    def _get_golden(self, *args, **kwargs) -> np.ndarray:
        """
        获取 Golden 输出（根据配置选择 cpp 或 python）
        """
        mode = get_golden_mode()

        if mode == "cpp":
            # 优先使用类方法 cpu_golden
            if hasattr(self.__class__, 'cpu_golden') and self.__class__.cpu_golden is not Op.cpu_golden:
                return self.cpu_golden(*args, **kwargs)
            # 兼容旧的注册方式
            if has_golden_cpp(self.name):
                return _golden_cpp_registry[self.name](*args, **kwargs)
            raise RuntimeError(
                f"算子 '{self.name}' 没有 C++ Golden 实现，"
                f"请在算子类中实现 cpu_golden 方法，"
                f"或设置 set_golden_mode('python') 使用 Python 实现"
            )
        else:
            return self.golden_python(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        调用算子

        执行流程：
        1. 如果 compute_golden=True，执行 golden (cpp 或 python) -> 保存为 golden
        2. 执行 reference -> 保存为 reference（用于 fuzzy 比对）
        3. 返回 golden 或 reference 的结果
        """
        # 计数
        idx = _counter.get(self.name, 0)
        _counter[self.name] = idx + 1
        full_name = f"{self.name}_{idx}"

        # 执行 golden（如果启用）
        golden_output = None
        if get_compute_golden():
            golden_output = self._get_golden(*args, **kwargs)
            logger.debug(f"{full_name}: golden ({get_golden_mode()}) 执行完成")

        # 执行 reference（用于 fuzzy 比对）
        ref_output = self.reference(*args, **kwargs)
        logger.debug(f"{full_name}: reference 执行完成")

        # 记录
        record = {
            "name": full_name,
            "op": self.name,
            "input": args[0] if args else None,
            "weight": args[1] if len(args) > 1 else kwargs.get("weight"),
            "golden": golden_output,      # golden 输出（cpp 或 python），可能为 None
            "reference": ref_output,      # 高精度参考（用于 fuzzy 比对）
        }
        _records.append(record)

        # 返回结果：优先 golden，否则 reference
        return golden_output if golden_output is not None else ref_output

    def __repr__(self):
        has_cpp = "✓" if has_golden_cpp(self.name) else "✗"
        return f"<Op {self.name} cpp={has_cpp}>"


def _is_array_like(obj: Any) -> bool:
    """检查是否为数组类型"""
    return isinstance(obj, np.ndarray) or (hasattr(obj, '__array__') and not isinstance(obj, (dict, list)))


def dump(output_dir: str = "./workspace", format: str = "raw") -> None:
    """
    导出所有记录的数据

    Args:
        output_dir: 输出目录
        format: 数据格式 ("raw", "npy", "npz")

    导出文件：
        - {name}_golden.bin: Golden 输出
        - {name}_reference.bin: 高精度参考（用于 fuzzy 比对）
        - {name}_input.bin: 输入数据
        - {name}_weight.bin: 权重数据
    """
    from aidevtools.formats.base import save as save_data

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for r in _records:
        name = r["name"]
        # 保存 golden
        if r["golden"] is not None and _is_array_like(r["golden"]):
            save_data(str(path / f"{name}_golden.bin"), np.asarray(r["golden"]), format=format)
        # 保存 reference
        if r["reference"] is not None and _is_array_like(r["reference"]):
            save_data(str(path / f"{name}_reference.bin"), np.asarray(r["reference"]), format=format)
        # 保存输入
        if r["input"] is not None and _is_array_like(r["input"]):
            save_data(str(path / f"{name}_input.bin"), np.asarray(r["input"]), format=format)
        # 保存权重
        if r["weight"] is not None and _is_array_like(r["weight"]):
            save_data(str(path / f"{name}_weight.bin"), np.asarray(r["weight"]), format=format)
        logger.info(f"dump: {name}")
