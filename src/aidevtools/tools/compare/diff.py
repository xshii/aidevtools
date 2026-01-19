"""比对核心逻辑"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from aidevtools.core.log import logger


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
        elif self.fuzzy_qnt.passed:
            return "PASS"
        elif self.fuzzy_pure.passed and not self.fuzzy_qnt.passed:
            return "QUANT_ISSUE"
        else:
            return "FAIL"


def compare_exact(golden: np.ndarray, result: np.ndarray,
                  max_abs: float = 0.0, max_count: int = 0) -> ExactResult:
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

    abs_err = np.abs(g - r)
    max_abs_actual = float(abs_err.max()) if len(abs_err) > 0 else 0.0

    if max_abs == 0:
        # bit 级精确比对
        mismatch_mask = (golden.view(np.uint8) != result.view(np.uint8))
        mismatch_count = int(np.sum(mismatch_mask))
        first_diff = np.argmax(mismatch_mask) if mismatch_count > 0 else -1
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
    )

def compare_bit(golden: bytes, result: bytes) -> bool:
    """bit 级对比，完全一致"""
    return golden == result

def compare_block(golden: np.ndarray, result: np.ndarray,
                  block_size: int = 256, threshold: float = 1e-5) -> List[Dict]:
    """
    分块对比 (256 byte 粒度)
    返回每个 block 的对比结果
    """
    g_flat = golden.flatten().view(np.uint8)
    r_flat = result.flatten().view(np.uint8)

    blocks = []
    for i in range(0, len(g_flat), block_size):
        g_block = g_flat[i:i+block_size].view(golden.dtype)
        r_block = r_flat[i:i+block_size].view(result.dtype)

        if len(g_block) == 0:
            continue

        abs_err = np.abs(g_block.astype(np.float64) - r_block.astype(np.float64))
        max_abs = float(abs_err.max())
        qsnr = calc_qsnr(g_block, r_block)

        blocks.append({
            "offset": i,
            "size": len(g_block) * g_block.itemsize,
            "max_abs": max_abs,
            "qsnr": qsnr,
            "passed": max_abs < threshold,
        })

    return blocks

def compare_full(golden: np.ndarray, result: np.ndarray,
                 atol: float = 1e-5, rtol: float = 1e-5) -> DiffResult:
    """完整对比"""
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    abs_err = np.abs(g - r)
    rel_err = abs_err / (np.abs(g) + 1e-12)

    max_abs = float(abs_err.max())
    mean_abs = float(abs_err.mean())
    max_rel = float(rel_err.max())

    qsnr = calc_qsnr(golden, result)
    cosine = calc_cosine(g, r)

    threshold = atol + rtol * np.abs(g)
    exceed_count = int(np.sum(abs_err > threshold))
    passed = exceed_count == 0

    return DiffResult(
        passed=passed,
        max_abs=max_abs,
        mean_abs=mean_abs,
        max_rel=max_rel,
        qsnr=qsnr,
        cosine=cosine,
        total_elements=len(g),
        exceed_count=exceed_count,
    )

def calc_qsnr(golden: np.ndarray, result: np.ndarray) -> float:
    """计算 QSNR (dB)"""
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()

    signal = np.sum(g ** 2)
    noise = np.sum((g - r) ** 2)

    if noise < 1e-12:
        return float('inf')
    return float(10 * np.log10(signal / noise))

def calc_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def compare_3col(
    op_name: str,
    op_id: int,
    result: np.ndarray,
    golden_pure: np.ndarray,
    golden_qnt: np.ndarray,
    exact_max_abs: float = 0.0,
    exact_max_count: int = 0,
    fuzzy_atol: float = 1e-5,
    fuzzy_rtol: float = 1e-3,
    fuzzy_min_qsnr: float = 30.0,
    fuzzy_min_cosine: float = 0.999,
) -> FullCompareResult:
    """
    三列比对

    Args:
        op_name: 算子名称
        op_id: 算子 ID
        result: DUT 输出
        golden_pure: 纯 fp32 golden
        golden_qnt: 量化感知 golden
        exact_*: 精确比对阈值
        fuzzy_*: 模糊比对阈值

    Returns:
        FullCompareResult
    """
    # 1. 精确比对 (result vs golden_pure)
    exact = compare_exact(golden_pure, result, exact_max_abs, exact_max_count)

    # 2. 模糊比对 - 纯 fp32 (result vs golden_pure)
    fuzzy_pure = compare_full(golden_pure, result, fuzzy_atol, fuzzy_rtol)
    # 检查额外阈值
    fuzzy_pure_passed = (
        fuzzy_pure.passed and
        fuzzy_pure.qsnr >= fuzzy_min_qsnr and
        fuzzy_pure.cosine >= fuzzy_min_cosine
    )
    fuzzy_pure = DiffResult(
        passed=fuzzy_pure_passed,
        max_abs=fuzzy_pure.max_abs,
        mean_abs=fuzzy_pure.mean_abs,
        max_rel=fuzzy_pure.max_rel,
        qsnr=fuzzy_pure.qsnr,
        cosine=fuzzy_pure.cosine,
        total_elements=fuzzy_pure.total_elements,
        exceed_count=fuzzy_pure.exceed_count,
    )

    # 3. 模糊比对 - 量化感知 (result vs golden_qnt)
    fuzzy_qnt = compare_full(golden_qnt, result, fuzzy_atol, fuzzy_rtol)
    fuzzy_qnt_passed = (
        fuzzy_qnt.passed and
        fuzzy_qnt.qsnr >= fuzzy_min_qsnr and
        fuzzy_qnt.cosine >= fuzzy_min_cosine
    )
    fuzzy_qnt = DiffResult(
        passed=fuzzy_qnt_passed,
        max_abs=fuzzy_qnt.max_abs,
        mean_abs=fuzzy_qnt.mean_abs,
        max_rel=fuzzy_qnt.max_rel,
        qsnr=fuzzy_qnt.qsnr,
        cosine=fuzzy_qnt.cosine,
        total_elements=fuzzy_qnt.total_elements,
        exceed_count=fuzzy_qnt.exceed_count,
    )

    return FullCompareResult(
        op_name=op_name,
        op_id=op_id,
        exact=exact,
        fuzzy_pure=fuzzy_pure,
        fuzzy_qnt=fuzzy_qnt,
    )


def print_compare_table(results: List[FullCompareResult]):
    """打印比对结果表格"""
    # 表头
    print()
    print("=" * 100)
    print(f"{'op_name':<15} {'exact':^7} {'f_pure':^7} {'f_qnt':^7} {'max_abs':>10} {'qsnr':>8} {'cosine':>8} {'status':^12}")
    print("-" * 100)

    # 数据行
    for r in results:
        exact_mark = "✓" if r.exact.passed else "✗"
        pure_mark = "✓" if r.fuzzy_pure.passed else "✗"
        qnt_mark = "✓" if r.fuzzy_qnt.passed else "✗"

        # 使用 fuzzy_qnt 的指标
        max_abs = f"{r.fuzzy_qnt.max_abs:.2e}"
        qsnr = f"{r.fuzzy_qnt.qsnr:.1f}" if r.fuzzy_qnt.qsnr != float('inf') else "inf"
        cosine = f"{r.fuzzy_qnt.cosine:.6f}"

        name = f"{r.op_name}_{r.op_id}"
        print(f"{name:<15} {exact_mark:^7} {pure_mark:^7} {qnt_mark:^7} {max_abs:>10} {qsnr:>8} {cosine:>8} {r.status:^12}")

    print("=" * 100)

    # 汇总
    total = len(results)
    perfect = sum(1 for r in results if r.status == "PERFECT")
    passed = sum(1 for r in results if r.status == "PASS")
    quant_issue = sum(1 for r in results if r.status == "QUANT_ISSUE")
    failed = sum(1 for r in results if r.status == "FAIL")

    print(f"\nSummary: {perfect} PERFECT, {passed} PASS, {quant_issue} QUANT_ISSUE, {failed} FAIL (total: {total})")
    print()
