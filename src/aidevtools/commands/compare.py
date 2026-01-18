"""比数命令"""
import numpy as np
from prettycli import command

from aidevtools.tools.compare.diff import compare_full, compare_block, calc_qsnr
from aidevtools.tools.compare.report import gen_report, gen_heatmap_svg
from aidevtools.tools.compare.runner import run_compare, archive
from aidevtools.formats.base import load
from aidevtools.core.log import logger


@command("compare", help="运行比数")
def cmd_compare(csv: str = "", op: str = "", mode: str = "", atol: str = "1e-5"):
    """
    运行比数

    Args:
        csv: compare.csv 路径
        op: 指定算子名（可选）
        mode: 指定模式 single/chain/full（可选）
        atol: 绝对误差阈值
    """
    if not csv:
        logger.error("请指定 csv 文件: compare --csv=xxx.csv")
        return 1

    run_compare(
        csv_path=csv,
        op_filter=op or None,
        mode_filter=mode or None,
        atol=float(atol),
    )
    return 0


@command("compare-archive", help="打包比数结果")
def cmd_archive(csv: str = ""):
    """打包比数结果为 zip"""
    if not csv:
        logger.error("请指定 csv 文件: compare-archive --csv=xxx.csv")
        return 1

    archive(csv)
    return 0


@command("compare-quick", help="快速比对两个文件")
def cmd_quick(golden: str = "", result: str = "", dtype: str = "float32"):
    """
    快速比对两个二进制文件

    Args:
        golden: golden 文件路径
        result: 待比对文件路径
        dtype: 数据类型
    """
    if not golden or not result:
        logger.error("请指定文件: compare-quick --golden=a.bin --result=b.bin")
        return 1

    dt = getattr(np, dtype)
    g = load(golden, format="raw", dtype=dt)
    r = load(result, format="raw", dtype=dt)

    diff = compare_full(g, r)

    status = "PASS" if diff.passed else "FAIL"
    print(f"状态: {status}")
    print(f"max_abs: {diff.max_abs:.6e}")
    print(f"qsnr: {diff.qsnr:.2f} dB")
    print(f"cosine: {diff.cosine:.6f}")

    return 0 if diff.passed else 1
