"""比数命令"""
import numpy as np
from prettycli import command

from aidevtools.tools.compare.diff import compare_full
from aidevtools.tools.compare.runner import run_compare, archive
from aidevtools.trace.tracer import dump, gen_csv, clear
from aidevtools.formats.base import load
from aidevtools.core.log import logger


@command("compare", help="比数工具")
def cmd_compare(
    action: str = "",
    csv: str = "",
    output: str = "./workspace",
    model: str = "model",
    format: str = "raw",
    op: str = "",
    mode: str = "",
    atol: str = "1e-5",
    golden: str = "",
    result: str = "",
    dtype: str = "float32",
):
    """
    比数工具

    子命令:
        csv      生成 compare.csv 配置表
        dump     导出 Golden 数据
        run      运行比数
        archive  打包比数结果
        clear    清空 Golden 记录
        quick    快速比对两个文件

    示例:
        compare csv --output=./workspace --model=resnet
        compare dump --output=./workspace --format=raw
        compare run --csv=compare.csv
        compare archive --csv=compare.csv
        compare clear
        compare quick --golden=a.bin --result=b.bin
    """
    if not action:
        print("用法: compare <action> [options]")
        print("子命令: csv, dump, run, archive, clear, quick")
        print("输入 compare --help 查看详情")
        return 1

    if action == "csv":
        csv_path = gen_csv(output, model)
        print(f"生成: {csv_path}")
        return 0

    elif action == "dump":
        dump(output, format=format)
        return 0

    elif action == "run":
        if not csv:
            logger.error("请指定 csv 文件: compare run --csv=xxx.csv")
            return 1
        run_compare(
            csv_path=csv,
            op_filter=op or None,
            mode_filter=mode or None,
            atol=float(atol),
        )
        return 0

    elif action == "archive":
        if not csv:
            logger.error("请指定 csv 文件: compare archive --csv=xxx.csv")
            return 1
        archive(csv)
        return 0

    elif action == "clear":
        clear()
        logger.info("Golden 记录已清空")
        return 0

    elif action == "quick":
        if not golden or not result:
            logger.error("请指定文件: compare quick --golden=a.bin --result=b.bin")
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

    else:
        logger.error(f"未知子命令: {action}")
        print("可用子命令: csv, dump, run, archive, clear, quick")
        return 1
