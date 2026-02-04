"""
精确比对

支持 bit 级精确比对和允许小误差的精确比对。
"""

import numpy as np

from .types import ExactResult


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
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    total = len(g)

    abs_err = np.abs(g - r)
    max_abs_actual = float(abs_err.max()) if len(abs_err) > 0 else 0.0

    if max_abs == 0:
        # bit 级精确比对 (需要 contiguous 数组)
        g_cont = np.ascontiguousarray(golden)
        r_cont = np.ascontiguousarray(result)
        mismatch_mask = g_cont.view(np.uint8) != r_cont.view(np.uint8)
        mismatch_count = int(np.sum(mismatch_mask))
        first_diff = int(np.argmax(mismatch_mask)) if mismatch_count > 0 else -1
    else:
        # 允许一定误差
        exceed_mask = abs_err > max_abs
        mismatch_count = int(np.sum(exceed_mask))
        first_diff = int(np.argmax(exceed_mask)) if mismatch_count > 0 else -1

    passed = mismatch_count <= max_count

    return ExactResult(
        passed=passed,
        mismatch_count=mismatch_count,
        first_diff_offset=first_diff,
        max_abs=max_abs_actual,
        total_elements=total,
    )


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
