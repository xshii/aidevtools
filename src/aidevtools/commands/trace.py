"""Trace 命令"""
from prettycli import command

from aidevtools.trace.tracer import dump, gen_csv, clear
from aidevtools.core.log import logger


@command("trace-dump", help="导出 trace 数据")
def cmd_dump(output: str = "./workspace", format: str = "raw"):
    """
    导出 trace 记录的数据

    Args:
        output: 输出目录
        format: 数据格式 (raw/numpy)
    """
    dump(output, format=format)
    return 0


@command("trace-csv", help="生成 compare.csv")
def cmd_csv(output: str = "./workspace", model: str = "model"):
    """
    生成 compare.csv 配置表

    Args:
        output: 输出目录
        model: 模型名称
    """
    csv_path = gen_csv(output, model)
    print(f"生成: {csv_path}")
    return 0


@command("trace-clear", help="清空 trace 记录")
def cmd_clear():
    """清空 trace 记录"""
    clear()
    logger.info("trace 记录已清空")
    return 0
