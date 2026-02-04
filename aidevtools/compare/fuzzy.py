"""
模糊比对

基于 QSNR、余弦相似度等指标的模糊比对。
"""

import numpy as np

from .metrics import calc_cosine, calc_qsnr, calc_exceed_count
from .types import FuzzyResult, CompareConfig


def compare_fuzzy(
    golden: np.ndarray,
    result: np.ndarray,
    config: CompareConfig = None,
) -> FuzzyResult:
    """
    模糊比对

    判定条件:
    1. 超阈值元素比例 <= max_exceed_ratio
    2. QSNR >= min_qsnr
    3. Cosine >= min_cosine

    Args:
        golden: 参考数据
        result: 待比对数据
        config: 比对配置

    Returns:
        FuzzyResult
    """
    if config is None:
        config = CompareConfig()

    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    total = len(g)

    # 计算误差统计
    abs_err = np.abs(g - r)
    max_abs = float(abs_err.max()) if total > 0 else 0.0
    mean_abs = float(abs_err.mean()) if total > 0 else 0.0

    # 相对误差 (避免除零警告)
    g_abs = np.abs(g)
    rel_err = np.zeros_like(abs_err)
    nonzero_mask = g_abs > 1e-12
    np.divide(abs_err, g_abs, out=rel_err, where=nonzero_mask)
    max_rel = float(rel_err.max()) if total > 0 else 0.0

    # QSNR 和余弦相似度
    qsnr = calc_qsnr(golden, result)
    cosine = calc_cosine(golden, result)

    # 超阈值统计
    exceed_count = calc_exceed_count(
        golden, result, config.fuzzy_atol, config.fuzzy_rtol
    )
    exceed_ratio = exceed_count / total if total > 0 else 0.0

    # 判定是否通过
    passed = (
        exceed_ratio <= config.fuzzy_max_exceed_ratio
        and qsnr >= config.fuzzy_min_qsnr
        and cosine >= config.fuzzy_min_cosine
    )

    return FuzzyResult(
        passed=passed,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        qsnr=qsnr,
        cosine=cosine,
        total_elements=total,
        exceed_count=exceed_count,
    )


def compare_isclose(
    golden: np.ndarray,
    result: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    max_exceed_ratio: float = 0.0,
) -> FuzzyResult:
    """
    IsClose 比对 - 类似 numpy.isclose

    判断条件: |result - golden| <= atol + rtol * |golden|
    通过条件: exceed_ratio <= max_exceed_ratio

    Args:
        golden: 参考数据 (golden)
        result: 待比对数据 (DUT 输出)
        atol: 绝对误差门限
        rtol: 相对误差门限
        max_exceed_ratio: 允许的最大超限比例

    Returns:
        FuzzyResult
    """
    config = CompareConfig(
        fuzzy_atol=atol,
        fuzzy_rtol=rtol,
        fuzzy_max_exceed_ratio=max_exceed_ratio,
        fuzzy_min_qsnr=0.0,  # 不检查 QSNR
        fuzzy_min_cosine=0.0,  # 不检查余弦
    )
    return compare_fuzzy(golden, result, config)
