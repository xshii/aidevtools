"""
分块比对 + 热力图定位

对大张量进行分块比对，快速定位误差集中区域。
支持文本热力图和 per-block 指标输出。

架构优化:
- 整体只做一次 float64 转换，per-block 通过 _PreparedPair 直接引用子切片，
  避免原版 calc_all_metrics 每个 block 重复执行 astype(float64).flatten()。
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from .metrics import _calc_all_metrics_prepared
from .types import _PreparedPair


@dataclass
class BlockResult:
    """单个 block 的比对结果"""

    offset: int
    size: int
    qsnr: float
    cosine: float
    max_abs: float
    exceed_count: int
    passed: bool


def compare_blocked(
    golden: np.ndarray,
    result: np.ndarray,
    block_size: int = 1024,
    min_qsnr: float = 30.0,
    min_cosine: float = 0.999,
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> List[BlockResult]:
    """
    分块比对

    将大张量拆分为 block_size 个元素的块，分别计算指标。
    用于快速定位误差集中的区域。

    优化: 整体只做一次 astype(float64).flatten()，per-block 构造
    _PreparedPair 直接引用子切片，避免每块重复转换。

    Args:
        golden: 参考数据
        result: 待比对数据
        block_size: 每块元素数 (默认 1024)
        min_qsnr: QSNR 阈值
        min_cosine: 余弦阈值
        atol: 绝对容差
        rtol: 相对容差

    Returns:
        每个 block 的比对结果列表
    """
    # 整体转换一次 (原版此处转换后，calc_all_metrics 每块又转换一次)
    g = golden.astype(np.float64).flatten()
    r = result.astype(np.float64).flatten()
    total = len(g)

    blocks = []
    for offset in range(0, total, block_size):
        end = min(offset + block_size, total)
        g_blk = g[offset:end]
        r_blk = r[offset:end]

        # 构造 per-block PreparedPair，直接复用切片，无额外拷贝
        diff = g_blk - r_blk
        p = _PreparedPair(
            g=g_blk, r=r_blk, diff=diff,
            abs_err=np.abs(diff), g_abs=np.abs(g_blk),
            total=end - offset,
        )
        m = _calc_all_metrics_prepared(p, atol=atol, rtol=rtol)

        passed = m.qsnr >= min_qsnr and m.cosine >= min_cosine
        blocks.append(BlockResult(
            offset=offset,
            size=end - offset,
            qsnr=m.qsnr,
            cosine=m.cosine,
            max_abs=m.max_abs,
            exceed_count=m.exceed_count,
            passed=passed,
        ))

    return blocks


def print_block_heatmap(
    blocks: List[BlockResult],
    cols: int = 40,
    show_legend: bool = True,
):
    """
    打印文本热力图

    用字符表示每个 block 的质量:
      '.' = QSNR >= 40 dB (excellent)
      'o' = QSNR >= 20 dB (good)
      'X' = QSNR >= 10 dB (marginal)
      '#' = QSNR < 10 dB (bad)

    Args:
        blocks: compare_blocked 的输出
        cols: 每行显示的 block 数
        show_legend: 是否显示图例
    """
    def _char(b):
        if b.qsnr == float("inf"):
            return "."
        if b.qsnr >= 40:
            return "."
        if b.qsnr >= 20:
            return "o"
        if b.qsnr >= 10:
            return "X"
        return "#"

    total = len(blocks)
    fail_count = sum(1 for b in blocks if not b.passed)
    worst = min(blocks, key=lambda b: b.qsnr) if blocks else None

    print(f"\n  Block Heatmap ({total} blocks, {fail_count} failed)")
    print(f"  {'='*cols}")

    for i in range(0, total, cols):
        row = blocks[i:i + cols]
        chars = "".join(_char(b) for b in row)
        start = row[0].offset
        print(f"  {start:>8} |{chars}|")

    print(f"  {'='*cols}")

    if worst:
        q_str = f"{worst.qsnr:.1f}" if worst.qsnr != float("inf") else "inf"
        print(f"  Worst block: offset={worst.offset}, QSNR={q_str} dB, "
              f"max_abs={worst.max_abs:.2e}")

    if show_legend:
        print("  Legend: . >= 40dB, o >= 20dB, X >= 10dB, # < 10dB")
    print()


def find_worst_blocks(
    blocks: List[BlockResult],
    top_n: int = 5,
) -> List[BlockResult]:
    """
    找出最差的 N 个 block

    Args:
        blocks: compare_blocked 的输出
        top_n: 返回最差的 N 个

    Returns:
        按 QSNR 从低到高排序的 block 列表
    """
    return sorted(blocks, key=lambda b: b.qsnr)[:top_n]
