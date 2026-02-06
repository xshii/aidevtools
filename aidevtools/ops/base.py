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

架构 (v2):
- 使用 ExecutionContext 管理状态，支持并发执行
- 所有状态操作通过 get_context() 获取上下文
"""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from aidevtools.core.config import get_config, set_config
from aidevtools.core.log import logger

# 从 context 模块导入
from aidevtools.ops.context import (
    CompareMode,
    OpNode,
    ExecutionContext,
    get_context,
    execution_context,
    profile_only_context as profile_only,
)


def fp32_reference(func: Callable) -> Callable:
    """装饰器：确保 ndarray 输入转为 fp32，输出也为 fp32"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for i, arg in enumerate(args):
            if i > 0 and isinstance(arg, np.ndarray):
                new_args.append(arg.astype(np.float32))
            else:
                new_args.append(arg)

        new_kwargs = {
            k: v.astype(np.float32) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }

        result = func(*new_args, **new_kwargs)
        if isinstance(result, np.ndarray):
            return result.astype(np.float32)
        return result

    return wrapper


# ============================================================
# Golden 配置 (使用 GlobalConfig)
# ============================================================


def set_golden_mode(mode: str) -> None:
    """设置 Golden 模式: "cpp" | "python" | "none" """
    if mode not in ("cpp", "python", "none"):
        raise ValueError(f"golden_mode 必须是 'cpp', 'python' 或 'none'")
    if mode == "none":
        set_config(golden_mode="python", compute_golden=False)
    else:
        set_config(golden_mode=mode, compute_golden=True)


def get_golden_mode() -> str:
    return get_config().golden_mode


def set_compute_golden(enabled: bool) -> None:
    set_config(compute_golden=enabled)


def get_compute_golden() -> bool:
    return get_config().compute_golden


# ============================================================
# Context 代理函数
# ============================================================


def clear() -> None:
    """清空当前上下文"""
    get_context().clear()


def get_records() -> List[Dict[str, Any]]:
    return get_context().get_records()


def get_profiles() -> List[Any]:
    return get_context().get_profiles()


def set_profile_enabled(enabled: bool) -> None:
    get_context().set_profile_enabled(enabled)


def get_profile_enabled() -> bool:
    return get_context().get_profile_enabled()


def set_profile_only(enabled: bool) -> None:
    get_context().set_profile_only(enabled)


def get_profile_only() -> bool:
    return get_context().get_profile_only()


def set_compare_mode(mode: CompareMode) -> None:
    get_context().set_compare_mode(mode)


def get_compare_mode() -> CompareMode:
    return get_context().get_compare_mode()


def mark_compare_point(op_name: str) -> None:
    get_context().mark_compare_point(op_name)


def get_graph() -> Dict[str, OpNode]:
    return get_context().get_graph()


def get_graph_ops() -> List[str]:
    return get_context().get_graph_ops()


def compare_final() -> Optional[Dict[str, Any]]:
    return get_context().compare_final()


# ============================================================
# Golden C++ 注册 (全局共享)
# ============================================================

_golden_cpp_registry: Dict[str, Callable] = {}


def register_golden_cpp(name: str) -> Callable[[Callable], Callable]:
    """注册 C++ Golden 实现"""
    def decorator(func: Callable) -> Callable:
        _golden_cpp_registry[name] = func
        return func
    return decorator


def has_golden_cpp(name: str) -> bool:
    return name in _golden_cpp_registry


# ============================================================
# 辅助函数
# ============================================================


def _is_array_like(obj: Any) -> bool:
    return isinstance(obj, np.ndarray) or (
        hasattr(obj, "__array__") and not isinstance(obj, (dict, list))
    )


def _extract_source_ops(args: tuple, kwargs: dict) -> List[str]:
    """从参数中提取来源算子"""
    from aidevtools.ops.traced_tensor import TracedTensor

    source_ops = []
    for arg in args:
        if isinstance(arg, TracedTensor) and arg.source_op is not None:
            source_ops.append(arg.source_op)
    for v in kwargs.values():
        if isinstance(v, TracedTensor) and v.source_op is not None:
            source_ops.append(v.source_op)
    return source_ops


def _collect_input_data(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """收集输入数据快照"""
    from aidevtools.ops.traced_tensor import TracedTensor

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
    return input_data


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

    # 解析参数
    param_names = meta.inputs + meta.optional
    param_values = {}
    for i, arg in enumerate(args):
        if i < len(param_names):
            param_values[param_names[i]] = arg
    param_values.update(kwargs)

    # 收集 shapes
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
        if name in meta.weight_params:
            weight_bytes += arr.nbytes
        else:
            input_bytes += arr.nbytes

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


# ============================================================
# Op 基类
# ============================================================


class Op:
    """算子基类"""

    name: str = None

    def __init__(self):
        if self.name is None:
            raise ValueError("算子必须定义 name 属性")

    @staticmethod
    def compute_flops(shapes: Dict[str, Any]) -> int:
        return 0

    def cpu_golden(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"{self.name} 未实现 cpu_golden")

    def gpu_golden(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"{self.name} 未实现 gpu_golden")

    def torch_reference(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError(f"{self.name} 未实现 torch_reference")

    def _get_reference(self, *args, **kwargs) -> Optional[np.ndarray]:
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
        """对输入数据进行量化/反量化"""
        from aidevtools.ops.cpu_golden import get_cpu_golden_dtype

        dtype = get_cpu_golden_dtype()

        def quantize_array(x):
            if not isinstance(x, np.ndarray):
                return x
            x = np.asarray(x, dtype=np.float16)
            x_fp32 = x.astype(np.float32)

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
            return x

        quantized_args = tuple(quantize_array(arg) for arg in args)
        quantized_kwargs = {k: quantize_array(v) for k, v in kwargs.items()}
        return quantized_args, quantized_kwargs

    def _get_golden(self, *args, **kwargs) -> np.ndarray:
        """获取 Golden 输出"""
        if (
            hasattr(self.__class__, "gpu_golden")
            and self.__class__.gpu_golden is not Op.gpu_golden
        ):
            return self.gpu_golden(*args, **kwargs)

        if (
            hasattr(self.__class__, "cpu_golden")
            and self.__class__.cpu_golden is not Op.cpu_golden
        ):
            return self.cpu_golden(*args, **kwargs)

        if has_golden_cpp(self.name):
            return _golden_cpp_registry[self.name](*args, **kwargs)

        raise NotImplementedError(f"算子 '{self.name}' 未实现 golden")

    def __call__(self, *args, **kwargs) -> Union[np.ndarray, "TracedTensor"]:  # noqa: F821
        """调用算子"""
        from aidevtools.ops.traced_tensor import TracedTensor, wrap_traced_output

        ctx = get_context()

        # 计数
        idx = ctx.next_counter(self.name)
        full_name = f"{self.name}_{idx}"

        # 检查 TracedTensor
        has_traced_input = any(
            isinstance(arg, TracedTensor) for arg in args
        ) or any(
            isinstance(v, TracedTensor) for v in kwargs.values()
        )

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

        # profile-only 模式
        if ctx.get_profile_only():
            if ctx.get_profile_enabled():
                profile = _create_profile(self.name, full_name, args, kwargs)
                if profile is not None:
                    ctx.add_profile(profile)
            first_input = args[0] if args else None
            if has_traced_input and first_input is not None:
                if isinstance(first_input, TracedTensor):
                    return first_input.with_source(full_name)
                return wrap_traced_output(np.asarray(first_input), input_dtype, full_name)
            return first_input

        # 执行 golden
        golden_output = self._get_golden(*args, **kwargs)

        # 比对
        reference_output = None
        if ctx.should_compare(full_name):
            quantized_args, quantized_kwargs = self._quantize_inputs(args, kwargs)
            reference_output = self._get_reference(*quantized_args, **quantized_kwargs)

        # 记录计算图
        compare_mode = ctx.get_compare_mode()
        if compare_mode in (CompareMode.FULL_GRAPH, CompareMode.MIXED):
            source_ops = _extract_source_ops(args, kwargs)
            input_data = _collect_input_data(args, kwargs)
            ctx.record_graph_node(full_name, self.name, source_ops, input_data, golden_output)

        # 记录
        record = {
            "name": full_name,
            "op": self.name,
            "input": args[0] if args else None,
            "weight": args[1] if len(args) > 1 else kwargs.get("weight"),
            "golden": golden_output,
            "reference": reference_output,
        }
        ctx.add_record(record)

        # 生成 profile
        if ctx.get_profile_enabled():
            profile = _create_profile(self.name, full_name, args, kwargs)
            if profile is not None:
                ctx.add_profile(profile)

        if has_traced_input:
            return wrap_traced_output(golden_output, input_dtype, full_name)
        return golden_output

    def __repr__(self):
        has_cpp = "✓" if has_golden_cpp(self.name) else "✗"
        return f"<Op {self.name} cpp={has_cpp}>"


# ============================================================
# 辅助函数 (供 registry.py 使用)
# ============================================================


def is_compute_flops_overridden(cls: type) -> bool:
    return hasattr(cls, "compute_flops") and cls.compute_flops is not Op.compute_flops


def is_cpu_golden_overridden(cls: type) -> bool:
    return hasattr(cls, "cpu_golden") and cls.cpu_golden is not Op.cpu_golden


def dump(output_dir: str = "./workspace", fmt: str = "raw") -> None:
    """导出所有记录的数据"""
    from aidevtools.formats.base import save as save_data

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    for r in get_records():
        name = r["name"]
        if r.get("golden") is not None and _is_array_like(r["golden"]):
            save_data(str(path / f"{name}_golden.bin"), np.asarray(r["golden"]), fmt=fmt)
        if r.get("reference") is not None and _is_array_like(r["reference"]):
            save_data(str(path / f"{name}_reference.bin"), np.asarray(r["reference"]), fmt=fmt)
        if r.get("input") is not None and _is_array_like(r["input"]):
            save_data(str(path / f"{name}_input.bin"), np.asarray(r["input"]), fmt=fmt)
        if r.get("weight") is not None and _is_array_like(r["weight"]):
            save_data(str(path / f"{name}_weight.bin"), np.asarray(r["weight"]), fmt=fmt)
        logger.info(f"dump: {name}")
