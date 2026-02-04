"""
比对指标计算

提供 QSNR、余弦相似度等指标的计算函数。
"""

import numpy as np


def calc_qsnr(golden: np.ndarray, result: np.ndarray) -> float:
    """
    计算量化信噪比 QSNR (dB)

    QSNR = 10 * log10(signal_power / noise_power)

    Args:
        golden: 参考数据
        result: 待比对数据

    Returns:
        QSNR 值 (dB), 无噪声时返回 inf
    """
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    signal = np.sum(g**2)
    noise = np.sum((g - r) ** 2)

    if noise < 1e-12:
        return float("inf")
    return float(10 * np.log10(signal / noise))


def calc_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算余弦相似度

    cosine = (a · b) / (||a|| * ||b||)

    Args:
        a: 向量 a
        b: 向量 b

    Returns:
        余弦相似度 [-1, 1], 零向量返回 0.0
    """
    a_flat = a.astype(np.float64).flatten()
    b_flat = b.astype(np.float64).flatten()

    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def calc_abs_error(golden: np.ndarray, result: np.ndarray) -> tuple:
    """
    计算绝对误差统计

    Args:
        golden: 参考数据
        result: 待比对数据

    Returns:
        (max_abs, mean_abs, abs_errors)
    """
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    abs_err = np.abs(g - r)
    max_abs = float(abs_err.max()) if len(abs_err) > 0 else 0.0
    mean_abs = float(abs_err.mean()) if len(abs_err) > 0 else 0.0

    return max_abs, mean_abs, abs_err


def calc_rel_error(golden: np.ndarray, result: np.ndarray) -> tuple:
    """
    计算相对误差统计

    Args:
        golden: 参考数据
        result: 待比对数据

    Returns:
        (max_rel, mean_rel, rel_errors)
    """
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    abs_err = np.abs(g - r)
    # 避免除零: 使用 np.divide 的 where 参数，仅在非零处计算
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
    """
    计算超阈值元素数

    判定条件: |result - golden| > atol + rtol * |golden|

    Args:
        golden: 参考数据
        result: 待比对数据
        atol: 绝对容差
        rtol: 相对容差

    Returns:
        超阈值元素数
    """
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    abs_err = np.abs(g - r)
    threshold = atol + rtol * np.abs(g)

    return int(np.sum(abs_err > threshold))


def check_nan_inf(data: np.ndarray) -> tuple:
    """
    检查 NaN 和 Inf

    Args:
        data: 输入数据

    Returns:
        (nan_count, inf_count, total)
    """
    flat = data.flatten()
    nan_count = int(np.sum(np.isnan(flat)))
    inf_count = int(np.sum(np.isinf(flat)))
    return nan_count, inf_count, len(flat)


def check_nonzero(data: np.ndarray) -> tuple:
    """
    检查非零元素

    Args:
        data: 输入数据

    Returns:
        (nonzero_count, total, nonzero_ratio)
    """
    flat = data.flatten()
    nonzero_count = int(np.count_nonzero(flat))
    total = len(flat)
    ratio = nonzero_count / total if total > 0 else 0.0
    return nonzero_count, total, ratio
