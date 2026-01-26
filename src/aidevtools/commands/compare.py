# pylint: disable=unused-argument
# kwargs 为 CLI 框架预留参数
"""比数命令"""

import numpy as np
from prettycli import command

from aidevtools.core.log import logger
from aidevtools.core.utils import parse_dtype, parse_list, parse_shape
from aidevtools.formats.base import load
from aidevtools.formats.quantize import list_quantize
from aidevtools.ops.base import clear as do_clear
from aidevtools.ops.base import dump as do_dump
from aidevtools.ops.base import get_records
from aidevtools.tools.compare.diff import compare_full


def _action_dump(output, **kwargs):
    """导出 Golden 数据"""
    do_dump(output, fmt=kwargs.get("format", "raw"))
    print(f"导出 Golden 数据到: {output}")
    return 0


def _action_clear(**kwargs):
    """清空 Golden 记录"""
    do_clear()
    logger.info("Golden 记录已清空")
    return 0


def _action_single(golden, result, dtype, shape, **kwargs):
    """单次比对两个文件"""
    if not golden or not result:
        logger.error(
            "请指定文件: compare single --golden=a.bin --result=b.bin --dtype=float32 --shape=1,64,32,32"
        )
        return 1
    dt = parse_dtype(dtype)
    sh = parse_shape(shape)
    g = load(golden, fmt="raw", dtype=dt, shape=sh)
    r = load(result, fmt="raw", dtype=dt, shape=sh)
    diff = compare_full(g, r)
    status = "PASS" if diff.passed else "FAIL"
    print(f"状态: {status}")
    print(f"shape: {g.shape}")
    print(f"max_abs: {diff.max_abs:.6e}")
    print(f"qsnr: {diff.qsnr:.2f} dB")
    print(f"cosine: {diff.cosine:.6f}")
    return 0 if diff.passed else 1


def _action_fuzzy(golden, result, dtype, shape, **kwargs):
    """模糊比对"""
    if not golden or not result:
        logger.error("请指定文件: compare fuzzy --golden=a.bin --result=b.bin")
        return 1
    dt = parse_dtype(dtype)
    sh = parse_shape(shape)
    g = load(golden, fmt="raw", dtype=dt, shape=sh)
    r = load(result, fmt="raw", dtype=dt, shape=sh)

    g_f64 = g.astype(np.float64).flatten()
    r_f64 = r.astype(np.float64).flatten()

    diff_val = g_f64 - r_f64
    signal_power = np.mean(g_f64**2)
    noise_power = np.mean(diff_val**2)
    qsnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

    norm_g = np.linalg.norm(g_f64)
    norm_r = np.linalg.norm(r_f64)
    cosine = np.dot(g_f64, r_f64) / (norm_g * norm_r) if norm_g > 0 and norm_r > 0 else 0.0
    max_abs = np.max(np.abs(diff_val))

    print("模式: 模糊比对 (fuzzy)")
    print(f"shape: {g.shape}")
    print(f"max_abs: {max_abs:.6e}")
    print(f"qsnr: {qsnr:.2f} dB")
    print(f"cosine: {cosine:.6f}")
    return 0


def _action_convert(golden, output, dtype, shape, target_dtype, **kwargs):
    """类型转换导出"""
    from aidevtools.formats.quantize import quantize

    if not golden:
        logger.error(
            "请指定输入文件: compare convert --golden=a.bin --output=out.bin --target_dtype=float16"
        )
        return 1
    if not target_dtype:
        logger.error("请指定目标类型: --target_dtype=float16 (可用: compare qtypes 查看)")
        return 1
    out_path = output if output != "./workspace" else golden.replace(".bin", f"_{target_dtype}.bin")

    dt = parse_dtype(dtype)
    sh = parse_shape(shape)
    data = load(golden, fmt="raw", dtype=dt, shape=sh)

    try:
        converted, meta = quantize(data, target_dtype)
    except (NotImplementedError, ValueError) as e:
        logger.error(str(e))
        return 1

    converted.tofile(out_path)
    print(f"转换: {dtype} → {target_dtype}")
    print(f"shape: {data.shape}")
    print(f"输出: {out_path}")
    if meta:
        print(f"meta: {meta}")
    return 0


def _action_qtypes(**kwargs):
    """列出支持的量化类型"""
    print("支持的量化类型:")
    for qtype in list_quantize():
        print(f"  - {qtype}")
    return 0


# Action 分发表
_ACTIONS = {
    "dump": _action_dump,
    "clear": _action_clear,
    "single": _action_single,
    "fuzzy": _action_fuzzy,
    "convert": _action_convert,
    "qtypes": _action_qtypes,
}


@command("compare", help="比数工具")
def cmd_compare(
    action: str = "",
    subaction: str = "",
    xlsx: str = "",
    output: str = "./workspace",
    model: str = "model",
    format: str = "raw",  # pylint: disable=redefined-builtin  # CLI 参数名
    golden: str = "",
    result: str = "",
    dtype: str = "float32",
    shape: str = "",
    target_dtype: str = "",
    ops: str = "",
):
    """
    比数工具

    用法:
        compare <action>        执行指定步骤

    子命令:
        dump       导出 Golden 数据
        clear      清空 Golden 记录
        single     单次比对两个文件
        fuzzy      模糊比对（跳过 bit 级比对）
        convert    类型转换导出
        qtypes     列出支持的量化类型

    xlsx 子命令:
        xlsx template   生成 xlsx 空模板
        xlsx export     从 trace 导出到 xlsx
        xlsx import     从 xlsx 生成 Python 代码
        xlsx run        从 xlsx 运行比数
        xlsx ops        列出可用算子

    参数:
        --target_dtype     转换目标类型 (float16/bfloat16/...)
        --xlsx=xxx.xlsx    指定 xlsx 文件
        --ops=linear,relu  限定算子列表（xlsx template 用）

    示例:
        compare dump --output=./workspace                       导出数据
        compare single --golden=a.bin --result=b.bin --dtype=float32 --shape=1,64,32,32
        compare fuzzy --golden=a.bin --result=b.bin --dtype=float32
        compare convert --golden=a.bin --output=a_fp16.bin --target_dtype=float16
        compare qtypes                                          列出量化类型

    xlsx 示例:
        compare xlsx template --output=config.xlsx              生成空模板
        compare xlsx template --output=config.xlsx --ops=linear,relu  限定算子
        compare xlsx export --xlsx=config.xlsx                  从 trace 导出
        compare xlsx import --xlsx=config.xlsx --output=gen.py  生成 Python
        compare xlsx run --xlsx=config.xlsx                     运行比数
    """
    # 无 action 时显示帮助
    if not action:
        print("请指定子命令，例如: compare xlsx run --xlsx=config.xlsx")
        print("查看帮助: compare --help")
        return 1

    # 使用分发表处理 action
    if action in _ACTIONS:
        return _ACTIONS[action](
            output=output,
            format=format,
            golden=golden,
            result=result,
            dtype=dtype,
            shape=shape,
            target_dtype=target_dtype,
        )
    if action == "xlsx":
        return _handle_xlsx(subaction, xlsx, output, model, fmt=format, ops_str=ops)
    logger.error(f"未知子命令: {action}")
    print("可用子命令: dump, clear, single, fuzzy, convert, qtypes, xlsx")
    return 1


# xlsx 子命令处理函数
def _xlsx_template(xlsx_path, output, model, ops_str, **kwargs):
    from aidevtools.xlsx import create_template
    from aidevtools.xlsx.op_registry import list_ops

    out_path = xlsx_path if xlsx_path else f"{output}/{model}_config.xlsx"
    ops_list = parse_list(ops_str) or None
    create_template(out_path, ops=ops_list)
    print(f"生成 xlsx 模板: {out_path}")
    print(f"限定算子: {', '.join(ops_list)}" if ops_list else f"可用算子: {', '.join(list_ops())}")
    return 0


def _xlsx_export(xlsx_path, output, model, **kwargs):
    from aidevtools.xlsx import create_template, export_xlsx

    if not xlsx_path:
        xlsx_path = f"{output}/{model}_config.xlsx"
    records = get_records()
    if not records:
        logger.warning("没有 trace 记录，请先运行算子")
        create_template(xlsx_path)
        print(f"生成空模板 (无记录): {xlsx_path}")
        return 0
    export_xlsx(xlsx_path, records)
    print(f"导出到 xlsx: {xlsx_path} ({len(records)} 条记录)")
    return 0


def _xlsx_import(xlsx_path, output, model, **kwargs):
    from aidevtools.xlsx import import_xlsx

    if not xlsx_path:
        logger.error("请指定 xlsx 文件: compare xlsx import --xlsx=config.xlsx")
        return 1
    out_py = output if output.endswith(".py") else f"{output}/generated_{model}.py"
    import_xlsx(xlsx_path, out_py)
    print(f"生成 Python 代码: {out_py}")
    return 0


def _xlsx_run(xlsx_path, output, fmt, **kwargs):
    from aidevtools.xlsx import run_xlsx

    if not xlsx_path:
        logger.error("请指定 xlsx 文件: compare xlsx run --xlsx=config.xlsx")
        return 1
    results = run_xlsx(xlsx_path, output, fmt=fmt)
    pass_count = sum(1 for r in results if r.get("status") == "PASS")
    fail_count = sum(1 for r in results if r.get("status") == "FAIL")
    skip_count = sum(1 for r in results if r.get("status") in ("SKIP", "PENDING", "ERROR"))
    print(f"比数完成: PASS={pass_count}, FAIL={fail_count}, SKIP/PENDING={skip_count}")
    print(f"结果已更新到: {xlsx_path}")
    return 0 if fail_count == 0 else 1


def _xlsx_ops(**kwargs):
    from aidevtools.xlsx.op_registry import list_ops

    print("可用算子:")
    for op in list_ops():
        print(f"  - {op}")
    return 0


# xlsx 子命令分发表
_XLSX_ACTIONS = {
    "template": _xlsx_template,
    "t": _xlsx_template,
    "export": _xlsx_export,
    "e": _xlsx_export,
    "import": _xlsx_import,
    "i": _xlsx_import,
    "run": _xlsx_run,
    "r": _xlsx_run,
    "ops": _xlsx_ops,
    "o": _xlsx_ops,
}


def _handle_xlsx(
    subaction: str, xlsx_path: str, output: str, model: str, fmt: str, ops_str: str
) -> int:
    """处理 xlsx 子命令"""
    if subaction in _XLSX_ACTIONS:
        return _XLSX_ACTIONS[subaction](
            xlsx_path=xlsx_path, output=output, model=model, fmt=fmt, ops_str=ops_str
        )
    logger.error(f"未知 xlsx 子命令: {subaction}")
    print("可用 xlsx 子命令: template(t), export(e), import(i), run(r), ops(o)")
    return 1
