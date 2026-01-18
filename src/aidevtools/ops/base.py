"""算子基础框架"""
import numpy as np
from typing import Callable, Dict, Any
from functools import wraps

from aidevtools.trace import trace

# 全局模式: "golden" | "reference"
_mode = "reference"

# Golden 实现注册表
_golden_registry: Dict[str, Callable] = {}


def set_mode(mode: str):
    """
    设置算子模式

    Args:
        mode: "golden" - 精确比对模式（调用用户注册的 Golden 实现）
              "reference" - 模糊比对模式（调用内置 numpy/torch 实现）
    """
    global _mode
    if mode not in ("golden", "reference"):
        raise ValueError(f"无效模式: {mode}, 可选: golden, reference")
    _mode = mode


def get_mode() -> str:
    """获取当前模式"""
    return _mode


def register_golden(name: str):
    """
    注册 Golden 实现

    示例:
        from my_cpp_lib import cpp_linear

        @register_golden("linear")
        def golden_linear(x, weight, bias=None):
            return cpp_linear(x, weight, bias)
    """
    def decorator(func: Callable):
        _golden_registry[name] = func
        return func
    return decorator


def get_golden(name: str) -> Callable:
    """获取 Golden 实现"""
    if name not in _golden_registry:
        raise ValueError(f"未注册 Golden 实现: {name}, 请使用 @register_golden('{name}') 注册")
    return _golden_registry[name]


def has_golden(name: str) -> bool:
    """检查是否有 Golden 实现"""
    return name in _golden_registry


class Op:
    """
    算子基类

    使用方法:
        class Linear(Op):
            name = "linear"

            def reference(self, x, weight, bias=None):
                '''numpy 参考实现'''
                y = np.matmul(x, weight)
                if bias is not None:
                    y = y + bias
                return y

        # 使用
        linear = Linear()
        y = linear(x, weight, bias)  # 根据 mode 自动选择实现
    """
    name: str = None  # 算子名，子类必须定义

    def __init__(self, enable_trace: bool = True):
        """
        Args:
            enable_trace: 是否启用 trace 记录
        """
        if self.name is None:
            raise ValueError("算子必须定义 name 属性")
        self.enable_trace = enable_trace
        self._call_count = 0

    def reference(self, *args, **kwargs) -> np.ndarray:
        """
        参考实现（numpy/torch），子类必须实现

        模糊比对模式下调用此方法
        """
        raise NotImplementedError(f"{self.name} 未实现 reference 方法")

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """调用算子"""
        # 选择实现
        if _mode == "golden":
            if not has_golden(self.name):
                raise ValueError(
                    f"Golden 模式下未找到 {self.name} 的 Golden 实现，"
                    f"请使用 @register_golden('{self.name}') 注册"
                )
            impl = get_golden(self.name)
        else:
            impl = self.reference

        # 执行
        if self.enable_trace:
            # 包装 trace
            trace_name = f"{self.name}_{self._call_count}"
            self._call_count += 1

            @trace(name=self.name)
            def traced_call(*a, **kw):
                return impl(*a, **kw)

            return traced_call(*args, **kwargs)
        else:
            return impl(*args, **kwargs)

    def __repr__(self):
        return f"<Op {self.name} mode={_mode}>"
