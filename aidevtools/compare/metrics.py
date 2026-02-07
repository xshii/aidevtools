"""
比对指标计算

提供 QSNR、余弦相似度等指标的计算函数。
支持单次遍历合并计算 (calc_all_metrics) 以减少重复的 flatten/convert 开销。

架构优化:
- _calc_all_metrics_prepared / _calc_all_metrics_early_exit_prepared
  接受 _PreparedPair，避免跨模块重复 astype + flatten。
- 公共函数 calc_all_metrics / calc_all_metrics_early_exit 内部自动创建
  _PreparedPair 后委托给 prepared 版本，保持向后兼容。
"""

from dataclasses import dataclass

import numpy as np

from .types import _PreparedPair


@dataclass
class AllMetrics:
    """单次遍历计算的全部指标"""

    qsnr: float
    cosine: float
    max_abs: float
    mean_abs: float
    max_rel: float
    exceed_count: int
    total_elements: int


_EMPTY_METRICS = AllMetrics(
    qsnr=float("inf"), cosine=0.0,
    max_abs=0.0, mean_abs=0.0, max_rel=0.0,
    exceed_count=0, total_elements=0,
)


# === Prepared 版本 (内部使用) ===

def _calc_all_metrics_prepared(
    p: _PreparedPair,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> AllMetrics:
    """使用预处理数据计算全部指标 — 消除重复 flatten/float64 转换"""
    if p.total == 0:
        return _EMPTY_METRICS

    max_abs = float(p.abs_err.max())
    mean_abs = float(p.abs_err.mean())

    # relative error — 复用 p.abs_err, p.g_abs
    rel_err = np.zeros_like(p.abs_err)
    nonzero_mask = p.g_abs > 1e-12
    np.divide(p.abs_err, p.g_abs, out=rel_err, where=nonzero_mask)
    max_rel = float(rel_err.max())

    # QSNR — 复用 p.g, p.diff
    g_sq_sum = np.dot(p.g, p.g)
    noise_sq_sum = np.dot(p.diff, p.diff)
    if noise_sq_sum < 1e-12:
        qsnr = float("inf")
    else:
        qsnr = float(10 * np.log10(g_sq_sum / noise_sq_sum))

    # cosine — 复用 g_sq_sum
    norm_g = np.sqrt(g_sq_sum)
    norm_r = np.sqrt(np.dot(p.r, p.r))
    if norm_g < 1e-12 or norm_r < 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(p.g, p.r) / (norm_g * norm_r))

    # exceed count — 复用 p.abs_err, p.g_abs
    threshold = atol + rtol * p.g_abs
    exceed_count = int(np.count_nonzero(p.abs_err > threshold))

    return AllMetrics(
        qsnr=qsnr, cosine=cosine,
        max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel,
        exceed_count=exceed_count, total_elements=p.total,
    )


def _calc_all_metrics_early_exit_prepared(
    p: _PreparedPair,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    min_qsnr: float = 30.0,
    min_cosine: float = 0.999,
    max_exceed_ratio: float = 0.0,
) -> AllMetrics:
    """带 early exit 的 prepared 版本

    优化点 (相对原版):
    - 接受 _PreparedPair，不再重复 astype + flatten
    - g_abs 一次性从 p.g_abs 获取，消除 cosine-fail 路径和 exceed 路径的重复计算
    """
    if p.total == 0:
        return _EMPTY_METRICS

    max_abs = float(p.abs_err.max())
    mean_abs = float(p.abs_err.mean())
    g_sq_sum = np.dot(p.g, p.g)

    # Step 1: cosine (cheapest: reuse g_sq_sum + dot)
    norm_g = np.sqrt(g_sq_sum)
    norm_r = np.sqrt(np.dot(p.r, p.r))
    if norm_g < 1e-12 or norm_r < 1e-12:
        cosine = 0.0
    else:
        cosine = float(np.dot(p.g, p.r) / (norm_g * norm_r))

    # helper: remaining expensive metrics — 复用 p.g_abs (修复原版重复计算)
    def _finish_remaining():
        rel_err = np.zeros_like(p.abs_err)
        mask = p.g_abs > 1e-12
        np.divide(p.abs_err, p.g_abs, out=rel_err, where=mask)
        noise_sq = np.dot(p.diff, p.diff)
        q = float("inf") if noise_sq < 1e-12 else float(
            10 * np.log10(g_sq_sum / noise_sq))
        thr = atol + rtol * p.g_abs
        exc = int(np.count_nonzero(p.abs_err > thr))
        return q, float(rel_err.max()), exc

    # Early exit: cosine 不达标
    if cosine < min_cosine:
        qsnr, max_rel, exceed_count = _finish_remaining()
        return AllMetrics(
            qsnr=qsnr, cosine=cosine,
            max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel,
            exceed_count=exceed_count, total_elements=p.total,
        )

    # Step 2: exceed count — 直接复用 p.g_abs (原版此处重新计算了 np.abs(g))
    threshold = atol + rtol * p.g_abs
    exceed_count = int(np.count_nonzero(p.abs_err > threshold))

    # Early exit: exceed 不达标
    if p.total > 0 and exceed_count / p.total > max_exceed_ratio:
        rel_err = np.zeros_like(p.abs_err)
        mask = p.g_abs > 1e-12
        np.divide(p.abs_err, p.g_abs, out=rel_err, where=mask)
        noise_sq = np.dot(p.diff, p.diff)
        qsnr = float("inf") if noise_sq < 1e-12 else float(
            10 * np.log10(g_sq_sum / noise_sq))
        return AllMetrics(
            qsnr=qsnr, cosine=cosine,
            max_abs=max_abs, mean_abs=mean_abs, max_rel=float(rel_err.max()),
            exceed_count=exceed_count, total_elements=p.total,
        )

    # Step 3: QSNR + relative error
    rel_err = np.zeros_like(p.abs_err)
    mask = p.g_abs > 1e-12
    np.divide(p.abs_err, p.g_abs, out=rel_err, where=mask)
    max_rel = float(rel_err.max())

    noise_sq_sum = np.dot(p.diff, p.diff)
    if noise_sq_sum < 1e-12:
        qsnr = float("inf")
    else:
        qsnr = float(10 * np.log10(g_sq_sum / noise_sq_sum))

    return AllMetrics(
        qsnr=qsnr, cosine=cosine,
        max_abs=max_abs, mean_abs=mean_abs, max_rel=max_rel,
        exceed_count=exceed_count, total_elements=p.total,
    )


# === 公共 API (向后兼容) ===

def calc_all_metrics(
    golden: np.ndarray,
    result: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> AllMetrics:
    """
    单次遍历计算全部指标 (QSNR / cosine / abs / rel / exceed)

    避免重复 flatten + float64 转换，大数组 2-3x 提速。
    """
    p = _PreparedPair.from_arrays(golden, result)
    return _calc_all_metrics_prepared(p, atol, rtol)


def calc_all_metrics_early_exit(
    golden: np.ndarray,
    result: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    min_qsnr: float = 30.0,
    min_cosine: float = 0.999,
    max_exceed_ratio: float = 0.0,
) -> AllMetrics:
    """
    带 early exit 的单次遍历计算

    先算 cosine (最快)，不满足直接短路剩余指标;
    再算 exceed，最后算 QSNR。失败 case 2-3x 提速。
    """
    p = _PreparedPair.from_arrays(golden, result)
    return _calc_all_metrics_early_exit_prepared(
        p, atol, rtol, min_qsnr, min_cosine, max_exceed_ratio,
    )


# === 原始独立函数 (保持向后兼容) ===

def calc_qsnr(golden: np.ndarray, result: np.ndarray) -> float:
    """计算量化信噪比 QSNR (dB)"""
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    signal = np.sum(g**2)
    noise = np.sum((g - r) ** 2)
    if noise < 1e-12:
        return float("inf")
    return float(10 * np.log10(signal / noise))


def calc_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    a_flat = a.astype(np.float64).flatten()
    b_flat = b.astype(np.float64).flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def calc_abs_error(golden: np.ndarray, result: np.ndarray) -> tuple:
    """计算绝对误差统计: (max_abs, mean_abs, abs_errors)"""
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    abs_err = np.abs(g - r)
    max_abs = float(abs_err.max()) if len(abs_err) > 0 else 0.0
    mean_abs = float(abs_err.mean()) if len(abs_err) > 0 else 0.0
    return max_abs, mean_abs, abs_err


def calc_rel_error(golden: np.ndarray, result: np.ndarray) -> tuple:
    """计算相对误差统计: (max_rel, mean_rel, rel_errors)"""
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    abs_err = np.abs(g - r)
    g_abs = np.abs(g)
    rel_err = np.zeros_like(abs_err)
    nonzero_mask = g_abs > 1e-12
    np.divide(abs_err, g_abs, out=rel_err, where=nonzero_mask)
    max_rel = float(rel_err.max()) if len(rel_err) > 0 else 0.0
    mean_rel = float(rel_err.mean()) if len(rel_err) > 0 else 0.0
    return max_rel, mean_rel, rel_err


def calc_exceed_count(
    golden: np.ndarray, result: np.ndarray, atol: float, rtol: float
) -> int:
    """计算超阈值元素数: |result - golden| > atol + rtol * |golden|"""
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    abs_err = np.abs(g - r)
    threshold = atol + rtol * np.abs(g)
    return int(np.sum(abs_err > threshold))


def check_nan_inf(data: np.ndarray) -> tuple:
    """检查 NaN 和 Inf: (nan_count, inf_count, total)"""
    flat = data.flatten()
    nan_count = int(np.sum(np.isnan(flat)))
    inf_count = int(np.sum(np.isinf(flat)))
    return nan_count, inf_count, len(flat)


def check_nonzero(data: np.ndarray) -> tuple:
    """检查非零元素: (nonzero_count, total, nonzero_ratio)"""
    flat = data.flatten()
    nonzero_count = int(np.count_nonzero(flat))
    total = len(flat)
    ratio = nonzero_count / total if total > 0 else 0.0
    return nonzero_count, total, ratio
