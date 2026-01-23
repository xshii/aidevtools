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
from functools import wraps
from typing import Callable, Dict, Any, List, Optional
from pathlib import Path

from aidevtools.core.log import logger
from aidevtools.core.config import get_config, set_config


def fp32_reference(func: Callable) -> Callable:
    """
    装饰器：确保 ndarray 输入转为 fp32，输出也为 fp32

    用于简化 reference() 方法的实现。

    Example:
        @fp32_reference
        def reference(self, x, y):
            return x * y  # 自动 fp32 计算，返回 fp32
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 转换位置参数 (跳过 self)
        new_args = []
        for i, arg in enumerate(args):
            if i > 0 and isinstance(arg, np.ndarray):
                new_args.append(arg.astype(np.float32))
            else:
                new_args.append(arg)

        # 转换关键字参数
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                new_kwargs[k] = v.astype(np.float32)
            else:
                new_kwargs[k] = v

        # 执行并转换结果
        result = func(*new_args, **new_kwargs)
        if isinstance(result, np.ndarray):
            return result.astype(np.float32)
        return result
    return wrapper


# 保留别名以兼容
fp64_reference = fp32_reference

# Golden 实现注册表 (C++ bindings)
_golden_cpp_registry: Dict[str, Callable] = {}

# 记录列表
_records: List[Dict[str, Any]] = []
_counter: Dict[str, int] = {}

# Profile 列表 (用于 Paper Analysis)
_profiles: List[Any] = []  # List[OpProfile]
_profile_enabled: bool = True  # 是否自动生成 profile
_profile_only: bool = False  # profile-only 模式：跳过 golden/reference，只生成 profile


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
    """清空记录和 profiles"""
    _records.clear()
    _counter.clear()
    _profiles.clear()


def set_profile_enabled(enabled: bool) -> None:
    """设置是否自动生成 profile (用于 Paper Analysis)"""
    global _profile_enabled
    _profile_enabled = enabled


def get_profile_enabled() -> bool:
    """获取是否自动生成 profile"""
    return _profile_enabled


def set_profile_only(enabled: bool) -> None:
    """
    设置 profile-only 模式

    在 profile-only 模式下，调用算子只生成 profile，不执行 golden/reference 计算。
    适用于 Paper Analysis 场景，只需要收集算子信息而不需要实际计算结果。

    Args:
        enabled: True=启用 profile-only 模式, False=正常执行模式

    Example:
        from aidevtools import ops
        from aidevtools.ops.nn import linear, relu

        # 启用 profile-only 模式
        ops.set_profile_only(True)
        ops.clear()

        # 定义模型（不执行实际计算）
        x = np.zeros((4, 512, 768), dtype=np.float16)
        w = np.zeros((768, 768), dtype=np.float16)
        linear(x, w)
        relu(x)

        # 获取 profiles 用于分析
        profiles = ops.get_profiles()
    """
    global _profile_only
    _profile_only = enabled
    if enabled:
        # profile-only 模式自动启用 profile 生成
        set_profile_enabled(True)
    logger.info(f"设置 profile_only = {enabled}")


def get_profile_only() -> bool:
    """获取是否处于 profile-only 模式"""
    return _profile_only


class profile_only:
    """
    profile-only 模式的上下文管理器

    在此上下文中，调用算子只生成 profile，不执行 golden/reference 计算。
    适用于 Paper Analysis 场景。

    Example:
        from aidevtools import ops
        from aidevtools.ops.nn import linear, relu
        from aidevtools.analysis import PaperAnalyzer

        with ops.profile_only():
            x = np.zeros((4, 512, 768), dtype=np.float16)
            w = np.zeros((768, 768), dtype=np.float16)
            linear(x, w)
            relu(x)
            profiles = ops.get_profiles()

        # 分析
        analyzer = PaperAnalyzer(chip="npu_910")
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()
    """

    def __init__(self, auto_clear: bool = True):
        """
        Args:
            auto_clear: 是否在进入上下文时自动清空 profiles (默认 True)
        """
        self._auto_clear = auto_clear
        self._previous_state = False

    def __enter__(self):
        self._previous_state = get_profile_only()
        if self._auto_clear:
            clear()
        set_profile_only(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_profile_only(self._previous_state)
        return False


def get_profiles() -> List[Any]:
    """
    获取收集的 OpProfile 列表 (用于 Paper Analysis)

    Returns:
        List[OpProfile]

    Example:
        from aidevtools import ops
        from aidevtools.analysis import PaperAnalyzer

        ops.clear()
        ops.linear(x, w)
        ops.relu(y)
        ops.softmax(z)

        # 获取 profiles 用于分析
        profiles = ops.get_profiles()
        analyzer = PaperAnalyzer(chip="npu_910")
        analyzer.add_profiles(profiles)
        result = analyzer.analyze()
    """
    return _profiles.copy()


def _create_profile(op_name: str, full_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
    """
    根据算子元信息自动创建 OpProfile

    Args:
        op_name: 算子名称 (如 "linear")
        full_name: 完整名称 (如 "linear_0")
        args: 调用参数
        kwargs: 调用关键字参数

    Returns:
        OpProfile 或 None
    """
    from aidevtools.ops.registry import get_op_meta

    meta = get_op_meta(op_name)
    if meta is None:
        return None

    # 懒加载 OpProfile (避免循环导入)
    try:
        from aidevtools.analysis.profile import OpProfile, dtype_bytes
    except ImportError:
        return None

    # 收集输入形状和字节数
    input_bytes = 0
    weight_bytes = 0
    output_bytes = 0
    shapes = {}
    dtype = "fp16"

    # 解析参数
    param_names = meta.inputs + meta.optional
    param_values = {}

    for i, arg in enumerate(args):
        if i < len(param_names):
            param_values[param_names[i]] = arg

    param_values.update(kwargs)

    # 计算字节数
    for name, value in param_values.items():
        if not _is_array_like(value):
            continue

        arr = np.asarray(value)
        nbytes = arr.nbytes
        shapes[f"{name}_shape"] = arr.shape
        shapes[f"{name}_size"] = arr.size

        # 推断 dtype
        if arr.dtype == np.float16:
            dtype = "fp16"
        elif arr.dtype == np.float32:
            dtype = "fp32"

        # 区分 input/weight
        if name in meta.weight_params:
            weight_bytes += nbytes
        else:
            input_bytes += nbytes

    # 计算输出字节数 (从第一个输入推断)
    first_input = args[0] if args else None
    if first_input is not None and _is_array_like(first_input):
        output_bytes = np.asarray(first_input).nbytes

    # 计算 FLOPs
    flops = 0
    if meta.flops_fn is not None:
        try:
            flops = meta.flops_fn(shapes)
        except Exception:
            flops = 0

    # 创建 profile
    profile = OpProfile(
        name=full_name,
        op_type=op_name,
        shapes=shapes,
        dtype=dtype,
        flops=int(flops),
        compute_unit=meta.compute_unit,
        input_bytes=int(input_bytes),
        weight_bytes=int(weight_bytes),
        output_bytes=int(output_bytes),
        memory_pattern=meta.memory_pattern,
    )

    return profile


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
        - profile-only 模式: 只生成 profile，不执行计算
        - 正常模式:
          1. 如果 compute_golden=True，执行 golden (cpp 或 python) -> 保存为 golden
          2. 执行 reference -> 保存为 reference（用于 fuzzy 比对）
          3. 返回 golden 或 reference 的结果
        """
        # 计数
        idx = _counter.get(self.name, 0)
        _counter[self.name] = idx + 1
        full_name = f"{self.name}_{idx}"

        # profile-only 模式：只生成 profile，跳过计算
        if _profile_only:
            if _profile_enabled:
                profile = _create_profile(self.name, full_name, args, kwargs)
                if profile is not None:
                    _profiles.append(profile)
                    logger.debug(f"{full_name}: profile 生成完成 (profile-only)")
            # 返回第一个输入（保持数据流）
            return args[0] if args else None

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

        # 自动生成 profile (用于 Paper Analysis)
        if _profile_enabled:
            profile = _create_profile(self.name, full_name, args, kwargs)
            if profile is not None:
                _profiles.append(profile)
                logger.debug(f"{full_name}: profile 生成完成")

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
