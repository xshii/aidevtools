"""算子基础框架

设计说明：
- 每个算子实现 cpu_golden 或 gpu_golden 方法（C++ 实现）
- 每个算子实现 torch_reference 方法（使用 torch 计算 reference）
- torch 对外部不可见，用户只用自定义 API

工作流：
1. 用户调用 F.matmul, F.softmax 等自定义 API
2. cpu_golden/gpu_golden 计算 golden（C++ 实现）
3. torch_reference 计算 reference（torch fp32，作为 ground truth）
4. 比对 golden 和 reference

比对模式：
- SINGLE_OP: 每个算子独立比对（默认）
- FULL_GRAPH: 只比对最终输出
- MIXED: 自动生成单算子 + 双算子组合测试
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from aidevtools.core.config import get_config, set_config
from aidevtools.core.log import logger


# ============================================================
# 比对模式
# ============================================================


class CompareMode(Enum):
    """比对模式

    - SINGLE_OP: 每个算子独立比对（默认，当前行为）
    - FULL_GRAPH: 只比对最终输出
    - MIXED: 自动生成单算子 + 双算子组合测试
    """
    SINGLE_OP = "single_op"
    FULL_GRAPH = "full_graph"
    MIXED = "mixed"


@dataclass
class OpNode:
    """计算图中的算子节点"""
    name: str           # 算子全名（如 "matmul_0"）
    op_type: str        # 算子类型（如 "matmul"）
    inputs: List[str] = field(default_factory=list)   # 输入节点名（source_op）
    input_data: Dict[str, Any] = field(default_factory=dict)  # 输入数据快照
    output_data: Optional[np.ndarray] = None  # 输出数据


# 全局状态
_compare_mode: CompareMode = CompareMode.SINGLE_OP
_graph: Dict[str, OpNode] = {}  # 计算图：op_name -> OpNode
_compare_points: set = set()    # 标记需要比对的算子


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
    """清空记录、profiles 和计算图，重置比对模式"""
    global _compare_mode  # pylint: disable=global-statement
    _records.clear()
    _counter.clear()
    _profiles.clear()
    _graph.clear()
    _compare_points.clear()
    _compare_mode = CompareMode.SINGLE_OP  # 重置为默认模式


def set_profile_enabled(enabled: bool) -> None:
    """设置是否自动生成 profile (用于 Paper Analysis)"""
    global _profile_enabled  # pylint: disable=global-statement
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
        from aidevtools import ops, F

        # 启用 profile-only 模式
        ops.set_profile_only(True)
        ops.clear()

        # 定义模型（不执行实际计算）
        x = np.zeros((4, 512, 768), dtype=np.float16)
        w = np.zeros((768, 768), dtype=np.float16)
        F.linear(x, w)
        F.relu(x)

        # 获取 profiles 用于分析
        profiles = ops.get_profiles()
    """
    global _profile_only  # pylint: disable=global-statement
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
        from aidevtools import ops, F
        from aidevtools.analysis import PaperAnalyzer

        with ops.profile_only():
            x = np.zeros((4, 512, 768), dtype=np.float16)
            w = np.zeros((768, 768), dtype=np.float16)
            F.linear(x, w)
            F.relu(x)
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


# ============================================================
# 比对模式 API
# ============================================================


def set_compare_mode(mode: CompareMode) -> None:
    """设置比对模式

    Args:
        mode: CompareMode 枚举值
            - SINGLE_OP: 每个算子独立比对（默认）
            - FULL_GRAPH: 只比对最终输出
            - MIXED: 自动生成单算子 + 双算子组合测试

    Example:
        >>> from aidevtools.ops.base import CompareMode
        >>> import aidevtools.ops as ops
        >>>
        >>> ops.set_compare_mode(CompareMode.MIXED)
        >>> ops.clear()
        >>>
        >>> y = F.matmul(x, w)
        >>> y = F.gelu(y)
        >>>
        >>> test_cases = ops.generate_test_cases()
    """
    global _compare_mode  # pylint: disable=global-statement
    _compare_mode = mode
    logger.info(f"设置 compare_mode = {mode.value}")


def get_compare_mode() -> CompareMode:
    """获取当前比对模式"""
    return _compare_mode


def mark_compare_point(op_name: str) -> None:
    """标记需要比对的算子

    在 FULL_GRAPH 模式下，可以用此函数标记中间需要比对的算子。

    Args:
        op_name: 算子名（如 "matmul_0"）
    """
    _compare_points.add(op_name)


def get_graph() -> Dict[str, OpNode]:
    """获取计算图

    Returns:
        算子名到 OpNode 的映射
    """
    return _graph.copy()


def get_graph_ops() -> List[str]:
    """获取计算图中所有算子名（按执行顺序）

    Returns:
        算子名列表
    """
    return list(_graph.keys())


def _should_compare(op_name: str) -> bool:
    """判断当前算子是否需要比对

    Args:
        op_name: 算子全名（如 "matmul_0"）

    Returns:
        是否需要比对
    """
    if _compare_mode == CompareMode.SINGLE_OP:
        # 单算子模式：每个算子都比对
        return True
    elif _compare_mode == CompareMode.FULL_GRAPH:
        # 完整图模式：只比对标记点（如果有），否则不比对
        return op_name in _compare_points
    else:
        # 混合模式：记录但不比对，由 generate_test_cases 生成测试
        return False


def _extract_source_ops(args: tuple, kwargs: dict) -> List[str]:
    """从参数中提取来源算子

    Args:
        args: 位置参数
        kwargs: 关键字参数

    Returns:
        来源算子名列表
    """
    from aidevtools.ops.traced_tensor import TracedTensor

    source_ops = []
    for arg in args:
        if isinstance(arg, TracedTensor) and arg.source_op is not None:
            source_ops.append(arg.source_op)
    for v in kwargs.values():
        if isinstance(v, TracedTensor) and v.source_op is not None:
            source_ops.append(v.source_op)
    return source_ops


def _record_graph_node(
    full_name: str,
    op_name: str,
    args: tuple,
    kwargs: dict,
    output_data: np.ndarray,
) -> None:
    """记录计算图节点

    Args:
        full_name: 算子全名（如 "matmul_0"）
        op_name: 算子类型（如 "matmul"）
        args: 位置参数
        kwargs: 关键字参数
        output_data: 输出数据
    """
    from aidevtools.ops.traced_tensor import TracedTensor

    # 提取来源算子
    source_ops = _extract_source_ops(args, kwargs)

    # 收集输入数据快照（用于后续重放）
    input_data = {}
    for i, arg in enumerate(args):
        if isinstance(arg, TracedTensor):
            input_data[f"arg_{i}"] = arg.data.copy()
        elif isinstance(arg, np.ndarray):
            input_data[f"arg_{i}"] = arg.copy()
    for k, v in kwargs.items():
        if isinstance(v, TracedTensor):
            input_data[k] = v.data.copy()
        elif isinstance(v, np.ndarray):
            input_data[k] = v.copy()

    # 创建节点
    node = OpNode(
        name=full_name,
        op_type=op_name,
        inputs=source_ops,
        input_data=input_data,
        output_data=output_data.copy() if output_data is not None else None,
    )
    _graph[full_name] = node


def compare_final() -> Optional[Dict[str, Any]]:
    """比对最终输出（FULL_GRAPH 模式）

    返回最后一个算子的比对结果。

    Returns:
        比对结果字典，包含 golden、reference、diff 等
    """
    if not _records:
        return None
    return _records[-1]


def _parse_param_values(args: tuple, kwargs: dict, param_names: list) -> dict:
    """解析参数为字典"""
    param_values = {}
    for i, arg in enumerate(args):
        if i < len(param_names):
            param_values[param_names[i]] = arg
    param_values.update(kwargs)
    return param_values


def _collect_array_info(param_values: dict, weight_params: set) -> tuple:
    """收集数组信息：shapes, dtype, input_bytes, weight_bytes"""
    shapes, dtype = {}, "fp16"
    input_bytes, weight_bytes = 0, 0

    for name, value in param_values.items():
        if not _is_array_like(value):
            continue

        arr = np.asarray(value)
        shapes[f"{name}_shape"] = arr.shape
        shapes[f"{name}_size"] = arr.size

        if arr.dtype == np.float16:
            dtype = "fp16"
        elif arr.dtype == np.float32:
            dtype = "fp32"

        if name in weight_params:
            weight_bytes += arr.nbytes
        else:
            input_bytes += arr.nbytes

    return shapes, dtype, input_bytes, weight_bytes


def _create_profile(op_name: str, full_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
    """根据算子元信息自动创建 OpProfile"""
    from aidevtools.ops._op_registry import get_op_meta

    meta = get_op_meta(op_name)
    if meta is None:
        return None

    try:
        from aidevtools.analysis.profile import OpProfile
    except ImportError:
        return None

    param_values = _parse_param_values(args, kwargs, meta.inputs + meta.optional)
    shapes, dtype, input_bytes, weight_bytes = _collect_array_info(
        param_values, set(meta.weight_params)
    )

    # 输出字节数
    first_input = args[0] if args else None
    output_bytes = (
        np.asarray(first_input).nbytes
        if first_input is not None and _is_array_like(first_input)
        else 0
    )

    # FLOPs
    flops = 0
    if meta.flops_fn is not None:
        try:
            flops = meta.flops_fn(shapes)
        except Exception:
            pass

    return OpProfile(
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


class Op:
    """
    算子基类

    子类必须实现：
    - name: 算子名称
    - cpu_golden() 或 gpu_golden(): Golden 实现

    可选：
    - compute_flops(): FLOPs 计算

    工作流：
    1. 使用 torch fp16 生成随机数据
    2. 使用 golden builder 量化数据
    3. 使用 cpu_golden/gpu_golden 计算 golden
    4. 使用 torch fp16 计算 reference (ground truth)
    5. 比对 golden 和 reference
    """

    name: str = None  # 算子名，子类必须定义

    def __init__(self):
        if self.name is None:
            raise ValueError("算子必须定义 name 属性")

    @staticmethod
    def compute_flops(shapes: Dict[str, Any]) -> int:  # pylint: disable=unused-argument
        """
        计算 FLOPs

        子类可重写此方法提供精确的 FLOPs 计算。

        Args:
            shapes: 形状字典，包含:
                - {param}_shape: 参数形状
                - {param}_size: 参数元素数量

        Returns:
            FLOPs 数量

        Example:
            @staticmethod
            def compute_flops(shapes):
                # MatMul: 2 * M * K * N
                a_shape = shapes.get("a_shape", (1, 1))
                b_shape = shapes.get("b_shape", (1, 1))
                M, K = a_shape[-2:]
                N = b_shape[-1]
                return 2 * M * K * N
        """
        # 默认返回 0，子类应重写
        return 0

    def cpu_golden(self, *args, **kwargs) -> np.ndarray:
        """
        C++ Golden 实现

        子类必须实现 cpu_golden 或 gpu_golden 之一
        """
        raise NotImplementedError(f"{self.name} 未实现 cpu_golden")

    def gpu_golden(self, *args, **kwargs) -> np.ndarray:
        """
        GPU Golden 实现

        子类必须实现 cpu_golden 或 gpu_golden 之一
        """
        raise NotImplementedError(f"{self.name} 未实现 gpu_golden")

    def torch_reference(self, *args, **kwargs) -> np.ndarray:
        """
        Torch Reference 实现（内部使用，外部不可见）

        使用 torch 计算高精度参考结果，作为 ground truth。
        子类应重写此方法提供 torch 实现。
        """
        raise NotImplementedError(f"{self.name} 未实现 torch_reference")

    def _get_reference(self, *args, **kwargs) -> Optional[np.ndarray]:
        """
        获取 Reference 输出（内部使用 torch）

        如果 torch_reference 未实现，返回 None
        """
        if (
            hasattr(self.__class__, "torch_reference")
            and self.__class__.torch_reference is not Op.torch_reference
        ):
            try:
                return self.torch_reference(*args, **kwargs)
            except Exception as e:
                logger.warning(f"torch_reference 计算失败: {e}")
                return None
        return None

    def _quantize_inputs(self, args: tuple, kwargs: dict) -> tuple:
        """
        对输入数据进行量化/反量化（模拟硬件精度损失）

        流程: fp16 -> 量化格式 -> fp16
        最高精度为 fp16，给 torch reference 计算
        """
        from aidevtools.ops.cpu_golden import get_cpu_golden_dtype

        dtype = get_cpu_golden_dtype()

        def quantize_array(x):
            """量化单个数组，返回 fp16"""
            if not isinstance(x, np.ndarray):
                return x
            # 先转 fp16（最高精度）
            x = np.asarray(x, dtype=np.float16)
            x_fp32 = x.astype(np.float32)  # gfloat 函数需要 fp32 输入

            if dtype == "gfp16":
                from aidevtools.formats.custom.gfloat.wrapper import (
                    fp32_to_gfloat16, gfloat16_to_fp32, is_cpp_available,
                )
                if is_cpp_available():
                    result = gfloat16_to_fp32(fp32_to_gfloat16(x_fp32))
                    return result.astype(np.float16)
            elif dtype == "gfp8":
                from aidevtools.formats.custom.gfloat.wrapper import (
                    fp32_to_gfloat8, gfloat8_to_fp32, is_cpp_available,
                )
                if is_cpp_available():
                    result = gfloat8_to_fp32(fp32_to_gfloat8(x_fp32))
                    return result.astype(np.float16)
            # gfp4 或其他格式，返回 fp16
            return x

        # 量化位置参数
        quantized_args = tuple(quantize_array(arg) for arg in args)

        # 量化关键字参数
        quantized_kwargs = {k: quantize_array(v) for k, v in kwargs.items()}

        return quantized_args, quantized_kwargs

    def _get_golden(self, *args, **kwargs) -> np.ndarray:
        """
        获取 Golden 输出

        优先级：gpu_golden > cpu_golden > 注册的 C++ golden
        如果都没有实现，抛出异常
        """
        # 优先使用 gpu_golden
        if (
            hasattr(self.__class__, "gpu_golden")
            and self.__class__.gpu_golden is not Op.gpu_golden
        ):
            return self.gpu_golden(*args, **kwargs)

        # 其次使用 cpu_golden
        if (
            hasattr(self.__class__, "cpu_golden")
            and self.__class__.cpu_golden is not Op.cpu_golden
        ):
            return self.cpu_golden(*args, **kwargs)

        # 兼容旧的注册方式
        if has_golden_cpp(self.name):
            return _golden_cpp_registry[self.name](*args, **kwargs)

        raise NotImplementedError(
            f"算子 '{self.name}' 未实现 golden，"
            f"请实现 cpu_golden 或 gpu_golden 方法"
        )

    def __call__(self, *args, **kwargs) -> Union[np.ndarray, "TracedTensor"]:  # noqa: F821
        """
        调用算子

        执行流程：
        - profile-only 模式: 只生成 profile，不执行计算
        - 正常模式: 执行 golden + reference

        返回值：
        - 如果输入包含 TracedTensor，返回 TracedTensor（带溯源信息）
        - 否则返回 np.ndarray（兼容旧行为）

        golden: cpu_golden/gpu_golden (C++ 实现)
        reference: torch_reference (torch 内部计算，外部不可见)
        """
        from aidevtools.ops.traced_tensor import TracedTensor, wrap_traced_output

        # 计数
        idx = _counter.get(self.name, 0)
        _counter[self.name] = idx + 1
        full_name = f"{self.name}_{idx}"

        # 检查输入是否包含 TracedTensor（用于决定输出类型）
        has_traced_input = any(
            isinstance(arg, TracedTensor) for arg in args
        ) or any(
            isinstance(v, TracedTensor) for v in kwargs.values()
        )

        # 获取输入的 dtype（如果有 TracedTensor）
        input_dtype = None
        for arg in args:
            if isinstance(arg, TracedTensor) and arg.dtype is not None:
                input_dtype = arg.dtype
                break
        if input_dtype is None:
            for v in kwargs.values():
                if isinstance(v, TracedTensor) and v.dtype is not None:
                    input_dtype = v.dtype
                    break

        # profile-only 模式：只生成 profile，跳过计算
        if _profile_only:
            if _profile_enabled:
                profile = _create_profile(self.name, full_name, args, kwargs)
                if profile is not None:
                    _profiles.append(profile)
                    logger.debug(f"{full_name}: profile 生成完成 (profile-only)")
            # 返回第一个输入（保持数据流）
            first_input = args[0] if args else None
            if has_traced_input and first_input is not None:
                if isinstance(first_input, TracedTensor):
                    return first_input.with_source(full_name)
                return wrap_traced_output(np.asarray(first_input), input_dtype, full_name)
            return first_input

        # 执行 golden (C++ 实现，接收原始 fp32，内部会做量化)
        golden_output = self._get_golden(*args, **kwargs)
        logger.debug(f"{full_name}: golden 执行完成")

        # 根据比对模式决定是否执行 reference 和比对
        should_compare = _should_compare(full_name)

        reference_output = None
        if should_compare:
            # 对输入数据进行量化/反量化（模拟硬件精度损失）
            quantized_args, quantized_kwargs = self._quantize_inputs(args, kwargs)

            # 执行 reference (torch 计算，使用量化/反量化后的 fp32)
            reference_output = self._get_reference(*quantized_args, **quantized_kwargs)
            if reference_output is not None:
                logger.debug(f"{full_name}: reference 执行完成")

        # 记录计算图节点（用于 MIXED 模式）
        if _compare_mode in (CompareMode.FULL_GRAPH, CompareMode.MIXED):
            _record_graph_node(full_name, self.name, args, kwargs, golden_output)

        # 记录（所有模式都记录，但 reference 可能为 None）
        record = {
            "name": full_name,
            "op": self.name,
            "input": args[0] if args else None,  # 原始输入
            "weight": args[1] if len(args) > 1 else kwargs.get("weight"),
            "golden": golden_output,
            "reference": reference_output,
        }
        _records.append(record)

        # 自动生成 profile (用于 Paper Analysis)
        if _profile_enabled:
            profile = _create_profile(self.name, full_name, args, kwargs)
            if profile is not None:
                _profiles.append(profile)
                logger.debug(f"{full_name}: profile 生成完成")

        # 返回 TracedTensor（如果输入包含 TracedTensor）或 np.ndarray
        if has_traced_input:
            return wrap_traced_output(golden_output, input_dtype, full_name)
        return golden_output

    def __repr__(self):
        has_cpp = "✓" if has_golden_cpp(self.name) else "✗"
        return f"<Op {self.name} cpp={has_cpp}>"


# ============================================================
# 辅助函数 (供 registry.py 使用，避免循环导入)
# ============================================================


def is_compute_flops_overridden(cls: type) -> bool:
    """检查类是否重写了 compute_flops 方法"""
    return hasattr(cls, "compute_flops") and cls.compute_flops is not Op.compute_flops


def is_cpu_golden_overridden(cls: type) -> bool:
    """检查类是否重写了 cpu_golden 方法"""
    return hasattr(cls, "cpu_golden") and cls.cpu_golden is not Op.cpu_golden


def _is_array_like(obj: Any) -> bool:
    """检查是否为数组类型"""
    return isinstance(obj, np.ndarray) or (
        hasattr(obj, "__array__") and not isinstance(obj, (dict, list))
    )


def dump(output_dir: str = "./workspace", fmt: str = "raw") -> None:
    """
    导出所有记录的数据

    Args:
        output_dir: 输出目录
        fmt: 数据格式 ("raw", "npy", "npz")

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
        if r.get("golden") is not None and _is_array_like(r["golden"]):
            save_data(str(path / f"{name}_golden.bin"), np.asarray(r["golden"]), fmt=fmt)
        # 保存 reference
        if r.get("reference") is not None and _is_array_like(r["reference"]):
            save_data(str(path / f"{name}_reference.bin"), np.asarray(r["reference"]), fmt=fmt)
        # 保存输入
        if r.get("input") is not None and _is_array_like(r["input"]):
            save_data(str(path / f"{name}_input.bin"), np.asarray(r["input"]), fmt=fmt)
        # 保存权重
        if r.get("weight") is not None and _is_array_like(r["weight"]):
            save_data(str(path / f"{name}_weight.bin"), np.asarray(r["weight"]), fmt=fmt)
        logger.info(f"dump: {name}")
