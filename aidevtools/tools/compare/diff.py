"""比对核心逻辑

统一使用 aidevtools.compare 核心模块进行计算。
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from aidevtools.compare.metrics import (
    calc_all_metrics as _calc_all_metrics,
    calc_qsnr,
    calc_cosine,
)
from aidevtools.compare.strategy import ExactStrategy


# === 结果类型 ===


@dataclass
class DiffResult:
    """比对结果"""

    passed: bool
    max_abs: float
    mean_abs: float
    max_rel: float
    qsnr: float
    cosine: float
    total_elements: int
    exceed_count: int  # 超阈值元素数


@dataclass
class ExactResult:
    """精确比对结果"""

    passed: bool
    mismatch_count: int
    first_diff_offset: int  # -1 表示无差异
    max_abs: float


@dataclass
class IsCloseResult:
    """
    IsClose 比对结果

    类似 numpy.isclose: |a - b| <= atol + rtol * |b|
    """

    passed: bool
    total_elements: int
    exceed_count: int
    exceed_ratio: float
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    mean_rel_error: float
    atol: float
    rtol: float
    max_exceed_ratio: float


# === 比对函数 ===


def compare_exact(
    golden: np.ndarray, result: np.ndarray, max_abs: float = 0.0, max_count: int = 0
) -> ExactResult:
    """精确比对"""
    core_result = ExactStrategy.compare(golden, result, max_abs, max_count)
    return ExactResult(
        passed=core_result.passed,
        mismatch_count=core_result.mismatch_count,
        first_diff_offset=core_result.first_diff_offset,
        max_abs=core_result.max_abs,
    )


def compare_full(
    golden: np.ndarray, result: np.ndarray, atol: float = 1e-5, rtol: float = 1e-5
) -> DiffResult:
    """完整对比"""
    m = _calc_all_metrics(golden, result, atol=atol, rtol=rtol)
    passed = m.exceed_count == 0

    return DiffResult(
        passed=passed,
        max_abs=m.max_abs,
        mean_abs=m.mean_abs,
        max_rel=m.max_rel,
        qsnr=m.qsnr,
        cosine=m.cosine,
        total_elements=m.total_elements,
        exceed_count=m.exceed_count,
    )


def compare_isclose(
    golden: np.ndarray,
    result: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    max_exceed_ratio: float = 0.0,
) -> IsCloseResult:
    """
    IsClose 比对 - 逐元素误差检查

    判断条件: |result - golden| <= atol + rtol * |golden|
    通过条件: exceed_ratio <= max_exceed_ratio
    """
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    if len(g) != len(r):
        raise ValueError(f"Shape mismatch: golden={golden.shape}, result={result.shape}")

    m = _calc_all_metrics(golden, result, atol=atol, rtol=rtol)

    abs_err = np.abs(g - r)
    g_abs = np.abs(g)
    rel_err = np.zeros_like(abs_err)
    nonzero_mask = g_abs > 1e-12
    np.divide(abs_err, g_abs, out=rel_err, where=nonzero_mask)
    mean_rel_error = float(rel_err.mean()) if len(rel_err) > 0 else 0.0

    exceed_ratio = m.exceed_count / m.total_elements if m.total_elements > 0 else 0.0
    passed = exceed_ratio <= max_exceed_ratio

    return IsCloseResult(
        passed=passed,
        total_elements=m.total_elements,
        exceed_count=m.exceed_count,
        exceed_ratio=exceed_ratio,
        max_abs_error=m.max_abs,
        max_rel_error=m.max_rel,
        mean_abs_error=m.mean_abs,
        mean_rel_error=mean_rel_error,
        atol=atol,
        rtol=rtol,
        max_exceed_ratio=max_exceed_ratio,
    )


def print_isclose_result(result: IsCloseResult, name: str = ""):
    """打印 IsClose 比对结果"""
    status = "PASS" if result.passed else "FAIL"
    name_str = f"[{name}] " if name else ""

    print(f"\n{name_str}IsClose 比对结果: {status}")
    print("-" * 50)
    print("  参数:")
    print(f"    atol (绝对门限):     {result.atol:.2e}")
    print(f"    rtol (相对门限):     {result.rtol:.2e}")
    print(f"    max_exceed_ratio:    {result.max_exceed_ratio:.2%}")
    print("  统计:")
    print(f"    总元素数:            {result.total_elements:,}")
    print(f"    超限元素数:          {result.exceed_count:,}")
    print(f"    超限比例:            {result.exceed_ratio:.4%}")
    print("  误差:")
    print(f"    最大绝对误差:        {result.max_abs_error:.6e}")
    print(f"    平均绝对误差:        {result.mean_abs_error:.6e}")
    print(f"    最大相对误差:        {result.max_rel_error:.6e}")
    print(f"    平均相对误差:        {result.mean_rel_error:.6e}")
    print("-" * 50)
