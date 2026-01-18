"""比对核心逻辑"""
import numpy as np
from typing import Dict, List, Any
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
