"""
精确比对

支持 bit 级精确比对和允许小误差的精确比对。
"""

import numpy as np

from .types import ExactResult, _PreparedPair


def _compare_exact_prepared(
    p: _PreparedPair,
    golden_orig: np.ndarray,
    result_orig: np.ndarray,
    max_abs: float = 0.0,
    max_count: int = 0,
) -> ExactResult:
    """使用预处理数据的精确比对 — 复用 p.abs_err 避免重复计算

    当 max_abs > 0 时，直接使用 p.abs_err 而不再重新计算 |g - r|。
    当 max_abs == 0 时，仍需原始数组做字节级比对。
    """
    max_abs_actual = float(p.abs_err.max()) if p.total > 0 else 0.0

    if max_abs == 0:
        # bit 级精确比对 (需要 contiguous 数组)
        g_cont = np.ascontiguousarray(golden_orig)
        r_cont = np.ascontiguousarray(result_orig)
        mismatch_mask = g_cont.view(np.uint8) != r_cont.view(np.uint8)
        mismatch_count = int(np.sum(mismatch_mask))
        first_diff = int(np.argmax(mismatch_mask)) if mismatch_count > 0 else -1
    else:
        # 允许一定误差 — 复用 p.abs_err
        exceed_mask = p.abs_err > max_abs
        mismatch_count = int(np.sum(exceed_mask))
        first_diff = int(np.argmax(exceed_mask)) if mismatch_count > 0 else -1

    return ExactResult(
        passed=mismatch_count <= max_count,
        mismatch_count=mismatch_count,
        first_diff_offset=first_diff,
        max_abs=max_abs_actual,
        total_elements=p.total,
    )


def compare_exact(
    golden: np.ndarray,
    result: np.ndarray,
    max_abs: float = 0.0,
    max_count: int = 0,
) -> ExactResult:
    """
    精确比对

    Args:
        golden: golden 数据
        result: 待比对数据
        max_abs: 允许的最大绝对误差 (0=bit级精确)
        max_count: 允许超阈值的元素个数

    Returns:
        ExactResult
    """
    p = _PreparedPair.from_arrays(golden, result)
    return _compare_exact_prepared(p, golden, result, max_abs, max_count)


def compare_bit(golden: bytes, result: bytes) -> bool:
    """
    bit 级对比，完全一致

    Args:
        golden: golden 字节数据
        result: 待比对字节数据

    Returns:
        是否完全一致
    """
    return golden == result
