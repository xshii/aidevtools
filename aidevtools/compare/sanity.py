"""
Golden 自检

验证 Golden 数据的有效性:
1. 数值非全零/全一
2. 无 NaN/Inf
3. 数值范围合理
4. Golden_qnt vs Golden_pure 的 QSNR >= 阈值
"""

import numpy as np

from .metrics import calc_qsnr, check_nan_inf, check_nonzero
from .types import SanityResult, CompareConfig


def check_golden_sanity(
    golden_pure: np.ndarray,
    golden_qnt: np.ndarray = None,
    config: CompareConfig = None,
) -> SanityResult:
    """
    Golden 自检

    检查项:
    1. non_zero: 数据非全零
    2. no_nan_inf: 无 NaN/Inf
    3. range_valid: 数值范围合理 (可选)
    4. qsnr_valid: golden_qnt vs golden_pure QSNR >= 阈值

    Args:
        golden_pure: 纯 fp32 Golden
        golden_qnt: 量化感知 Golden (可选)
        config: 比对配置

    Returns:
        SanityResult
    """
    if config is None:
        config = CompareConfig()

    result = SanityResult(valid=True, checks={}, messages=[])

    # 1. 检查非零
    _, _, nonzero_ratio = check_nonzero(golden_pure)
    result.non_zero = nonzero_ratio >= config.sanity_min_nonzero_ratio
    result.checks["non_zero"] = result.non_zero

    if not result.non_zero:
        result.messages.append(
            f"Golden 非零比例过低: {nonzero_ratio:.2%} < {config.sanity_min_nonzero_ratio:.2%}"
        )
        result.valid = False

    # 2. 检查 NaN/Inf
    nan_count, inf_count, total = check_nan_inf(golden_pure)
    nan_ratio = nan_count / total if total > 0 else 0.0
    inf_ratio = inf_count / total if total > 0 else 0.0

    result.no_nan_inf = (
        nan_ratio <= config.sanity_max_nan_ratio
        and inf_ratio <= config.sanity_max_inf_ratio
    )
    result.checks["no_nan_inf"] = result.no_nan_inf

    if not result.no_nan_inf:
        if nan_count > 0:
            result.messages.append(f"Golden 包含 NaN: {nan_count}/{total}")
        if inf_count > 0:
            result.messages.append(f"Golden 包含 Inf: {inf_count}/{total}")
        result.valid = False

    # 3. 检查数值范围 (简单检查: 不能全为同一值)
    unique_count = len(np.unique(golden_pure.flatten()[:1000]))  # 采样检查
    result.range_valid = unique_count > 1
    result.checks["range_valid"] = result.range_valid

    if not result.range_valid:
        result.messages.append("Golden 数值单一 (可能为常数)")
        result.valid = False

    # 4. 检查量化 QSNR (golden_qnt vs golden_pure)
    if golden_qnt is not None:
        qsnr = calc_qsnr(golden_pure, golden_qnt)
        result.qsnr_valid = qsnr >= config.sanity_min_qsnr or qsnr == float("inf")
        result.checks["qsnr_valid"] = result.qsnr_valid
        result.checks["qsnr_value"] = qsnr

        if not result.qsnr_valid:
            result.messages.append(
                f"Golden 量化 QSNR 过低: {qsnr:.2f} dB < {config.sanity_min_qsnr:.2f} dB"
            )
            result.valid = False
    else:
        result.qsnr_valid = True
        result.checks["qsnr_valid"] = True

    return result


def check_data_sanity(data: np.ndarray, name: str = "data") -> SanityResult:
    """
    通用数据自检

    Args:
        data: 输入数据
        name: 数据名称

    Returns:
        SanityResult
    """
    result = SanityResult(valid=True, checks={}, messages=[])

    # 检查非零
    _, _, nonzero_ratio = check_nonzero(data)
    result.non_zero = nonzero_ratio >= 0.01
    result.checks["non_zero"] = result.non_zero

    if not result.non_zero:
        result.messages.append(f"{name} 非零比例过低: {nonzero_ratio:.2%}")
        result.valid = False

    # 检查 NaN/Inf
    nan_count, inf_count, total = check_nan_inf(data)
    result.no_nan_inf = nan_count == 0 and inf_count == 0
    result.checks["no_nan_inf"] = result.no_nan_inf

    if not result.no_nan_inf:
        if nan_count > 0:
            result.messages.append(f"{name} 包含 NaN: {nan_count}/{total}")
        if inf_count > 0:
            result.messages.append(f"{name} 包含 Inf: {inf_count}/{total}")
        result.valid = False

    return result
