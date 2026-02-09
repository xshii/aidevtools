"""比对核心逻辑

统一使用 aidevtools.compare 核心模块进行计算，消除重复逻辑。
本模块保留原有类型和 API 以维持向后兼容。
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# === 核心计算委托给 compare 模块 ===
from aidevtools.compare.metrics import (
    calc_all_metrics as _calc_all_metrics,
    calc_qsnr,
    calc_cosine,
)
from aidevtools.compare.strategy import ExactStrategy


# === 本模块专有类型 (保持向后兼容) ===


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

    passed: bool  # 是否通过精度要求
    total_elements: int  # 总元素数
    exceed_count: int  # 超过门限的元素数
    exceed_ratio: float  # 超限比例 (exceed_count / total_elements)
    max_abs_error: float  # 最大绝对误差
    max_rel_error: float  # 最大相对误差
    mean_abs_error: float  # 平均绝对误差
    mean_rel_error: float  # 平均相对误差
    # 参数
    atol: float  # 绝对误差门限
    rtol: float  # 相对误差门限
    max_exceed_ratio: float  # 允许的最大超限比例


@dataclass
class CompareThresholds:
    """比对阈值配置"""

    # 精确比对
    exact_max_abs: float = 0.0
    exact_max_count: int = 0
    # 模糊比对
    fuzzy_atol: float = 1e-5
    fuzzy_rtol: float = 1e-3
    fuzzy_min_qsnr: float = 30.0
    fuzzy_min_cosine: float = 0.999


@dataclass
class FullCompareResult:
    """
    完整比对结果 (3 列)

    - exact: 精确比对
    - fuzzy_pure: 模糊比对 (全程 fp32)
    - fuzzy_qnt: 模糊比对 (带量化)
    """

    op_name: str
    op_id: int

    # 三列结果
    exact: ExactResult
    fuzzy_pure: DiffResult
    fuzzy_qnt: DiffResult

    # 汇总状态
    @property
    def status(self) -> str:
        """
        状态判定:
        - PERFECT: exact 通过
        - PASS: exact 不过，但 fuzzy_qnt 通过
        - QUANT_ISSUE: fuzzy_pure 通过，fuzzy_qnt 不过 (量化问题)
        - FAIL: 都不过
        """
        if self.exact.passed:
            return "PERFECT"
        if self.fuzzy_qnt.passed:
            return "PASS"
        if self.fuzzy_pure.passed and not self.fuzzy_qnt.passed:
            return "QUANT_ISSUE"
        return "FAIL"


# === 核心比对函数 (委托给 compare 模块) ===


def compare_exact(
    golden: np.ndarray, result: np.ndarray, max_abs: float = 0.0, max_count: int = 0
) -> ExactResult:
    """
    精确比对 - 委托给 compare.exact 模块

    Args:
        golden: golden 数据
        result: 待比对数据
        max_abs: 允许的最大绝对误差 (0=bit级精确)
        max_count: 允许超阈值的元素个数

    Returns:
        ExactResult
    """
    core_result = ExactStrategy.compare(golden, result, max_abs, max_count)
    return ExactResult(
        passed=core_result.passed,
        mismatch_count=core_result.mismatch_count,
        first_diff_offset=core_result.first_diff_offset,
        max_abs=core_result.max_abs,
    )


def compare_block(
    golden: np.ndarray, result: np.ndarray, block_size: int = 256, threshold: float = 1e-5
) -> List[Dict]:
    """
    分块对比 (256 byte 粒度)
    返回每个 block 的对比结果
    """
    g_flat = golden.flatten().view(np.uint8)
    r_flat = result.flatten().view(np.uint8)

    blocks = []
    for i in range(0, len(g_flat), block_size):
        g_block = g_flat[i : i + block_size].view(golden.dtype)
        r_block = r_flat[i : i + block_size].view(result.dtype)

        if len(g_block) == 0:
            continue

        # 使用核心模块的单次遍历计算
        m = _calc_all_metrics(g_block, r_block)

        blocks.append(
            {
                "offset": i,
                "size": len(g_block) * g_block.itemsize,
                "max_abs": m.max_abs,
                "qsnr": m.qsnr,
                "passed": m.max_abs < threshold,
            }
        )

    return blocks


def compare_full(
    golden: np.ndarray, result: np.ndarray, atol: float = 1e-5, rtol: float = 1e-5
) -> DiffResult:
    """完整对比 - 使用单次遍历优化"""
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

    # 使用核心模块做主要计算
    m = _calc_all_metrics(golden, result, atol=atol, rtol=rtol)

    # mean_rel 需要额外计算 (AllMetrics 只有 max_rel)
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


def _apply_fuzzy_thresholds(diff: DiffResult, thresholds: CompareThresholds) -> DiffResult:
    """应用模糊比对阈值，返回新的 DiffResult"""
    passed = (
        diff.passed
        and diff.qsnr >= thresholds.fuzzy_min_qsnr
        and diff.cosine >= thresholds.fuzzy_min_cosine
    )
    return DiffResult(
        passed=passed,
        max_abs=diff.max_abs,
        mean_abs=diff.mean_abs,
        max_rel=diff.max_rel,
        qsnr=diff.qsnr,
        cosine=diff.cosine,
        total_elements=diff.total_elements,
        exceed_count=diff.exceed_count,
    )


def compare_3col(
    op_name: str,
    op_id: int,
    result: np.ndarray,
    golden_pure: np.ndarray,
    golden_qnt: np.ndarray,
    thresholds: CompareThresholds = None,
) -> FullCompareResult:
    """
    三列比对

    Args:
        op_name: 算子名称
        op_id: 算子 ID
        result: DUT 输出
        golden_pure: 纯 fp32 golden
        golden_qnt: 量化感知 golden
        thresholds: 比对阈值配置

    Returns:
        FullCompareResult
    """
    if thresholds is None:
        thresholds = CompareThresholds()

    # 1. 精确比对 (result vs golden_pure)
    exact = compare_exact(golden_pure, result, thresholds.exact_max_abs, thresholds.exact_max_count)

    # 2. 模糊比对 - 纯 fp32 (result vs golden_pure)
    fuzzy_pure = compare_full(golden_pure, result, thresholds.fuzzy_atol, thresholds.fuzzy_rtol)
    fuzzy_pure = _apply_fuzzy_thresholds(fuzzy_pure, thresholds)

    # 3. 模糊比对 - 量化感知 (result vs golden_qnt)
    fuzzy_qnt = compare_full(golden_qnt, result, thresholds.fuzzy_atol, thresholds.fuzzy_rtol)
    fuzzy_qnt = _apply_fuzzy_thresholds(fuzzy_qnt, thresholds)

    return FullCompareResult(
        op_name=op_name,
        op_id=op_id,
        exact=exact,
        fuzzy_pure=fuzzy_pure,
        fuzzy_qnt=fuzzy_qnt,
    )


def _format_result_row(r: FullCompareResult) -> str:
    """格式化单行结果"""
    marks = (
        "✓" if r.exact.passed else "✗",
        "✓" if r.fuzzy_pure.passed else "✗",
        "✓" if r.fuzzy_qnt.passed else "✗",
    )
    max_abs = f"{r.fuzzy_qnt.max_abs:.2e}"
    qsnr = f"{r.fuzzy_qnt.qsnr:.1f}" if r.fuzzy_qnt.qsnr != float("inf") else "inf"
    cosine = f"{r.fuzzy_qnt.cosine:.6f}"
    name = f"{r.op_name}_{r.op_id}"
    return (
        f"{name:<15} {marks[0]:^7} {marks[1]:^7} {marks[2]:^7} "
        f"{max_abs:>10} {qsnr:>8} {cosine:>8} {r.status:^12}"
    )


def print_compare_table(results: List[FullCompareResult]):
    """打印比对结果表格"""
    print()
    print("=" * 100)
    header = (
        f"{'op_name':<15} {'exact':^7} {'f_pure':^7} {'f_qnt':^7} "
        f"{'max_abs':>10} {'qsnr':>8} {'cosine':>8} {'status':^12}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        print(_format_result_row(r))

    print("=" * 100)

    # 汇总统计
    status_counts = {"PERFECT": 0, "PASS": 0, "QUANT_ISSUE": 0, "FAIL": 0}
    for r in results:
        if r.status in status_counts:
            status_counts[r.status] += 1

    summary = (
        f"\nSummary: {status_counts['PERFECT']} PERFECT, {status_counts['PASS']} PASS, "
        f"{status_counts['QUANT_ISSUE']} QUANT_ISSUE, {status_counts['FAIL']} FAIL "
        f"(total: {len(results)})"
    )
    print(summary)
    print()
