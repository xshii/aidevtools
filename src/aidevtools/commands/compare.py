"""比数命令"""
import numpy as np
from prettycli import command

from aidevtools.tools.compare.diff import compare_full
from aidevtools.tools.compare.runner import run_compare, archive
from aidevtools.trace.tracer import dump, gen_csv, clear
from aidevtools.formats.base import load
from aidevtools.core.log import logger


# === 分步流程 ===

@command("compare1-dump", help="步骤1: 导出 Golden 数据")
def cmd_1(output: str = "./workspace", format: str = "raw"):
    """
    导出 @trace 记录的 Golden 数据

    Args:
        output: 输出目录
        format: 数据格式 (raw/numpy)
    """
    dump(output, format=format)
    return 0


@command("compare2-csv", help="步骤2: 生成 compare.csv")
def cmd_2(output: str = "./workspace", model: str = "model"):
    """
    生成 compare.csv 配置表

    Args:
        output: 输出目录
        model: 模型名称
    """
    csv_path = gen_csv(output, model)
    print(f"生成: {csv_path}")
    return 0


@command("compare3-run", help="步骤3: 运行比数")
def cmd_3(csv: str = "", op: str = "", mode: str = "", atol: str = "1e-5"):
    """
    运行比数

    Args:
        csv: compare.csv 路径
        op: 指定算子名（可选）
        mode: 指定模式 single/chain/full（可选）
        atol: 绝对误差阈值
    """
    if not csv:
        logger.error("请指定 csv 文件: compare3-run --csv=xxx.csv")
        return 1

    run_compare(
        csv_path=csv,
        op_filter=op or None,
        mode_filter=mode or None,
        atol=float(atol),
    )
    return 0


@command("compare4-archive", help="步骤4: 打包比数结果")
def cmd_4(csv: str = ""):
    """打包比数结果为 zip"""
    if not csv:
        logger.error("请指定 csv 文件: compare4-archive --csv=xxx.csv")
        return 1

    archive(csv)
    return 0


# === 辅助命令 ===

@command("comparec-clear", help="清空 Golden 记录")
def cmd_c():
    """清空 @trace 记录的 Golden 数据"""
    clear()
    logger.info("Golden 记录已清空")
    return 0


@command("compareq-quick", help="快速比对两个文件（一键式）")
def cmd_q(golden: str = "", result: str = "", dtype: str = "float32"):
    """
    快速比对两个二进制文件

    Args:
        golden: golden 文件路径
        result: 待比对文件路径
        dtype: 数据类型
    """
    if not golden or not result:
        logger.error("请指定文件: compareq-quick --golden=a.bin --result=b.bin")
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
