"""执行上下文管理

提供线程安全的执行上下文，替代全局状态，支持并发执行。

用法:
    from aidevtools.ops.context import ExecutionContext, get_context

    # 方式 1: 使用上下文管理器（推荐）
    with ExecutionContext() as ctx:
        ctx.clear()
        # 执行操作...
        profiles = ctx.get_profiles()

    # 方式 2: 使用默认上下文（兼容旧代码）
    ctx = get_context()
    ctx.clear()
    # 执行操作...

    # 方式 3: 独立上下文（并发执行）
    ctx1 = ExecutionContext()
    ctx2 = ExecutionContext()
    # ctx1 和 ctx2 状态完全隔离
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class CompareMode(Enum):
    """比对模式

    - SINGLE_OP: 每个算子独立比对（默认）
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


class ExecutionContext:
    """执行上下文

    管理算子执行的所有状态，支持并发执行。
    每个 ExecutionContext 实例有独立的状态空间。
    """

    def __init__(self):
        # 比对模式
        self._compare_mode: CompareMode = CompareMode.SINGLE_OP
        # 计算图
        self._graph: Dict[str, OpNode] = {}
        # 标记需要比对的算子
        self._compare_points: set = set()
        # 记录列表
        self._records: List[Dict[str, Any]] = []
        # 算子计数器
        self._counter: Dict[str, int] = {}
        # Profile 列表
        self._profiles: List[Any] = []
        self._profile_enabled: bool = True
        self._profile_only: bool = False
        # Golden C++ 注册表（共享）
        self._golden_cpp_registry: Dict[str, Callable] = {}
        # 锁（线程安全）
        self._lock = threading.RLock()

    def __enter__(self) -> "ExecutionContext":
        """上下文管理器入口"""
        _push_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """上下文管理器出口"""
        _pop_context()
        return False

    # ============================================================
    # 比对模式
    # ============================================================

    def set_compare_mode(self, mode: CompareMode) -> None:
        """设置比对模式"""
        with self._lock:
            self._compare_mode = mode

    def get_compare_mode(self) -> CompareMode:
        """获取比对模式"""
        return self._compare_mode

    def mark_compare_point(self, op_name: str) -> None:
        """标记需要比对的算子"""
        with self._lock:
            self._compare_points.add(op_name)

    def should_compare(self, op_name: str) -> bool:
        """判断当前算子是否需要比对"""
        if self._compare_mode == CompareMode.SINGLE_OP:
            return True
        elif self._compare_mode == CompareMode.FULL_GRAPH:
            return op_name in self._compare_points
        else:
            return False

    # ============================================================
    # 计算图
    # ============================================================

    def get_graph(self) -> Dict[str, OpNode]:
        """获取计算图"""
        with self._lock:
            return self._graph.copy()

    def get_graph_ops(self) -> List[str]:
        """获取计算图中所有算子名（按执行顺序）"""
        with self._lock:
            return list(self._graph.keys())

    def record_graph_node(
        self,
        full_name: str,
        op_type: str,
        inputs: List[str],
        input_data: Dict[str, Any],
        output_data: Optional[np.ndarray],
    ) -> None:
        """记录计算图节点"""
        node = OpNode(
            name=full_name,
            op_type=op_type,
            inputs=inputs,
            input_data=input_data,
            output_data=output_data.copy() if output_data is not None else None,
        )
        with self._lock:
            self._graph[full_name] = node

    # ============================================================
    # 记录
    # ============================================================

    def get_records(self) -> List[Dict[str, Any]]:
        """获取所有记录"""
        with self._lock:
            return self._records.copy()

    def add_record(self, record: Dict[str, Any]) -> None:
        """添加记录"""
        with self._lock:
            self._records.append(record)

    def compare_final(self) -> Optional[Dict[str, Any]]:
        """比对最终输出"""
        with self._lock:
            if not self._records:
                return None
            return self._records[-1]

    # ============================================================
    # 计数器
    # ============================================================

    def next_counter(self, op_name: str) -> int:
        """获取下一个算子编号"""
        with self._lock:
            idx = self._counter.get(op_name, 0)
            self._counter[op_name] = idx + 1
            return idx

    def get_counter(self, op_name: str) -> int:
        """获取当前算子计数"""
        return self._counter.get(op_name, 0)

    # ============================================================
    # Profile
    # ============================================================

    def set_profile_enabled(self, enabled: bool) -> None:
        """设置是否自动生成 profile"""
        self._profile_enabled = enabled

    def get_profile_enabled(self) -> bool:
        """获取是否自动生成 profile"""
        return self._profile_enabled

    def set_profile_only(self, enabled: bool) -> None:
        """设置 profile-only 模式"""
        self._profile_only = enabled
        if enabled:
            self._profile_enabled = True

    def get_profile_only(self) -> bool:
        """获取是否处于 profile-only 模式"""
        return self._profile_only

    def get_profiles(self) -> List[Any]:
        """获取收集的 profiles"""
        with self._lock:
            return self._profiles.copy()

    def add_profile(self, profile: Any) -> None:
        """添加 profile"""
        with self._lock:
            self._profiles.append(profile)

    # ============================================================
    # Golden 注册
    # ============================================================

    def register_golden_cpp(self, name: str, func: Callable) -> None:
        """注册 C++ Golden 实现"""
        with self._lock:
            self._golden_cpp_registry[name] = func

    def has_golden_cpp(self, name: str) -> bool:
        """检查是否有 C++ Golden 实现"""
        return name in self._golden_cpp_registry

    def get_golden_cpp(self, name: str) -> Optional[Callable]:
        """获取 C++ Golden 实现"""
        return self._golden_cpp_registry.get(name)

    # ============================================================
    # 清理
    # ============================================================

    def clear(self) -> None:
        """清空所有状态"""
        with self._lock:
            self._records.clear()
            self._counter.clear()
            self._profiles.clear()
            self._graph.clear()
            self._compare_points.clear()
            self._compare_mode = CompareMode.SINGLE_OP

    def reset(self) -> None:
        """重置（保留 golden 注册）"""
        self.clear()


# ============================================================
# 全局上下文管理（兼容旧代码）
# ============================================================

# 默认上下文
_default_context: Optional[ExecutionContext] = None
_context_lock = threading.Lock()

# 上下文栈（用于嵌套 with 语句）
_context_stack: threading.local = threading.local()


def _get_stack() -> List[ExecutionContext]:
    """获取当前线程的上下文栈"""
    if not hasattr(_context_stack, "stack"):
        _context_stack.stack = []
    return _context_stack.stack


def _push_context(ctx: ExecutionContext) -> None:
    """压入上下文"""
    _get_stack().append(ctx)


def _pop_context() -> Optional[ExecutionContext]:
    """弹出上下文"""
    stack = _get_stack()
    if stack:
        return stack.pop()
    return None


def get_context() -> ExecutionContext:
    """获取当前执行上下文

    优先级：
    1. 当前线程的上下文栈顶
    2. 默认全局上下文（懒初始化）
    """
    # 检查栈顶
    stack = _get_stack()
    if stack:
        return stack[-1]

    # 使用默认上下文
    global _default_context
    if _default_context is None:
        with _context_lock:
            if _default_context is None:
                _default_context = ExecutionContext()
    return _default_context


def set_default_context(ctx: ExecutionContext) -> None:
    """设置默认上下文"""
    global _default_context
    with _context_lock:
        _default_context = ctx


@contextmanager
def execution_context(ctx: Optional[ExecutionContext] = None):
    """执行上下文管理器

    Args:
        ctx: 上下文实例，None 表示创建新上下文

    Example:
        with execution_context() as ctx:
            ctx.clear()
            # 执行操作...
    """
    if ctx is None:
        ctx = ExecutionContext()

    _push_context(ctx)
    try:
        yield ctx
    finally:
        _pop_context()


# ============================================================
# profile_only 上下文管理器
# ============================================================


class profile_only_context:
    """profile-only 模式的上下文管理器"""

    def __init__(self, auto_clear: bool = True, ctx: Optional[ExecutionContext] = None):
        self._auto_clear = auto_clear
        self._ctx = ctx
        self._previous_state = False

    def __enter__(self):
        self._ctx = self._ctx or get_context()
        self._previous_state = self._ctx.get_profile_only()
        if self._auto_clear:
            self._ctx.clear()
        self._ctx.set_profile_only(True)
        return self._ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ctx.set_profile_only(self._previous_state)
        return False
