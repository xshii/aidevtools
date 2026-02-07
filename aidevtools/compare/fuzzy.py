"""
模糊比对

基于 QSNR、余弦相似度等指标的模糊比对。
使用单次遍历合并计算 + early exit 优化。
"""

from .metrics import calc_all_metrics_early_exit
from .types import FuzzyResult, CompareConfig


def compare_fuzzy(
    golden,
    result,
    config=None,
):
    """
    模糊比对 (单次遍历 + early exit 优化版)

    判定条件:
    1. 超阈值元素比例 <= max_exceed_ratio
    2. QSNR >= min_qsnr
    3. Cosine >= min_cosine

    优化:
    - 单次 flatten + float64 转换 (原先 4 次)
    - cosine 不达标 → early exit
    - exceed 不达标 → early exit
    - 大数组 2-3x 提速, 失败 case 额外 2-3x
    """
    if config is None:
        config = CompareConfig()

    m = calc_all_metrics_early_exit(
        golden, result,
        atol=config.fuzzy_atol,
        rtol=config.fuzzy_rtol,
        min_qsnr=config.fuzzy_min_qsnr,
        min_cosine=config.fuzzy_min_cosine,
        max_exceed_ratio=config.fuzzy_max_exceed_ratio,
    )

    exceed_ratio = m.exceed_count / m.total_elements if m.total_elements > 0 else 0.0

    passed = (
        exceed_ratio <= config.fuzzy_max_exceed_ratio
        and m.qsnr >= config.fuzzy_min_qsnr
        and m.cosine >= config.fuzzy_min_cosine
    )

    return FuzzyResult(
        passed=passed,
        max_abs=m.max_abs,
        mean_abs=m.mean_abs,
        max_rel=m.max_rel,
        qsnr=m.qsnr,
        cosine=m.cosine,
        total_elements=m.total_elements,
        exceed_count=m.exceed_count,
    )


def compare_isclose(golden, result, atol=1e-5, rtol=1e-3, max_exceed_ratio=0.0):
    """
    IsClose 比对 - 类似 numpy.isclose

    判断条件: |result - golden| <= atol + rtol * |golden|
    通过条件: exceed_ratio <= max_exceed_ratio
    """
    config = CompareConfig(
        fuzzy_atol=atol,
        fuzzy_rtol=rtol,
        fuzzy_max_exceed_ratio=max_exceed_ratio,
        fuzzy_min_qsnr=0.0,
        fuzzy_min_cosine=0.0,
    )
    return compare_fuzzy(golden, result, config)
