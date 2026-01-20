"""比数命令"""
import numpy as np
from prettycli import command

from aidevtools.tools.compare.diff import compare_full
from aidevtools.formats.quantize import list_quantize
from aidevtools.formats.base import load
from aidevtools.core.log import logger
from aidevtools.core.utils import parse_shape, parse_dtype, parse_list
from aidevtools.ops.base import get_records, dump, clear


@command("compare", help="比数工具")
def cmd_compare(
    action: str = "",
    subaction: str = "",
    xlsx: str = "",
    output: str = "./workspace",
    model: str = "model",
    format: str = "raw",
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

    if action == "dump":
        dump(output, format=format)
        print(f"导出 Golden 数据到: {output}")
        return 0

    elif action == "clear":
        clear()
        logger.info("Golden 记录已清空")
        return 0

    elif action == "single":
        if not golden or not result:
            logger.error("请指定文件: compare single --golden=a.bin --result=b.bin --dtype=float32 --shape=1,64,32,32")
            return 1
        dt = parse_dtype(dtype)
        sh = parse_shape(shape)
        g = load(golden, format="raw", dtype=dt, shape=sh)
        r = load(result, format="raw", dtype=dt, shape=sh)
        diff = compare_full(g, r)
        status = "PASS" if diff.passed else "FAIL"
        print(f"状态: {status}")
        print(f"shape: {g.shape}")
        print(f"max_abs: {diff.max_abs:.6e}")
        print(f"qsnr: {diff.qsnr:.2f} dB")
        print(f"cosine: {diff.cosine:.6f}")
        return 0 if diff.passed else 1

    elif action == "fuzzy":
        # 模糊比对：只做 QSNR/cosine，不做 bit 级比对
        if not golden or not result:
            logger.error("请指定文件: compare fuzzy --golden=a.bin --result=b.bin")
            return 1
        dt = parse_dtype(dtype)
        sh = parse_shape(shape)
        g = load(golden, format="raw", dtype=dt, shape=sh)
        r = load(result, format="raw", dtype=dt, shape=sh)

        # 模糊比对：转为 float64 计算指标
        g_f64 = g.astype(np.float64).flatten()
        r_f64 = r.astype(np.float64).flatten()

        # QSNR
        diff_val = g_f64 - r_f64
        signal_power = np.mean(g_f64 ** 2)
        noise_power = np.mean(diff_val ** 2)
        qsnr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        # Cosine
        norm_g = np.linalg.norm(g_f64)
        norm_r = np.linalg.norm(r_f64)
        cosine = np.dot(g_f64, r_f64) / (norm_g * norm_r) if norm_g > 0 and norm_r > 0 else 0.0

        # Max abs
        max_abs = np.max(np.abs(diff_val))

        print(f"模式: 模糊比对 (fuzzy)")
        print(f"shape: {g.shape}")
        print(f"max_abs: {max_abs:.6e}")
        print(f"qsnr: {qsnr:.2f} dB")
        print(f"cosine: {cosine:.6f}")
        return 0

    elif action == "convert":
        # 类型转换导出
        from aidevtools.formats.quantize import quantize

        if not golden:
            logger.error("请指定输入文件: compare convert --golden=a.bin --output=out.bin --target_dtype=float16")
            return 1
        if not target_dtype:
            logger.error("请指定目标类型: --target_dtype=float16 (可用: compare qtypes 查看)")
            return 1
        if not output or output == "./workspace":
            output = golden.replace(".bin", f"_{target_dtype}.bin")

        dt = parse_dtype(dtype)
        sh = parse_shape(shape)
        data = load(golden, format="raw", dtype=dt, shape=sh)

        # 使用量化框架转换
        try:
            converted, meta = quantize(data, target_dtype)
        except (NotImplementedError, ValueError) as e:
            logger.error(str(e))
            return 1

        # 导出
        converted.tofile(output)
        print(f"转换: {dtype} → {target_dtype}")
        print(f"shape: {data.shape}")
        print(f"输出: {output}")
        if meta:
            print(f"meta: {meta}")
        return 0

    elif action == "qtypes":
        # 列出支持的量化类型
        print("支持的量化类型:")
        for qtype in list_quantize():
            print(f"  - {qtype}")
        return 0

    elif action == "xlsx":
        # xlsx 子命令，需要第二个参数
        return _handle_xlsx(subaction, xlsx, output, model, format, ops)

    else:
        logger.error(f"未知子命令: {action}")
        print("可用子命令: dump, clear, single, fuzzy, convert, qtypes, xlsx")
        return 1


def _handle_xlsx(subaction: str, xlsx_path: str, output: str, model: str, format: str, ops_str: str) -> int:
    """处理 xlsx 子命令"""
    from aidevtools.xlsx import create_template, export_xlsx, import_xlsx, run_xlsx
    from aidevtools.xlsx.op_registry import list_ops

    if subaction in ("template", "t"):
        # 生成空模板
        out_path = xlsx_path if xlsx_path else f"{output}/{model}_config.xlsx"
        ops_list = parse_list(ops_str) or None
        create_template(out_path, ops=ops_list)
        print(f"生成 xlsx 模板: {out_path}")
        if ops_list:
            print(f"限定算子: {', '.join(ops_list)}")
        else:
            print(f"可用算子: {', '.join(list_ops())}")
        return 0

    elif subaction in ("export", "e"):
        # 从 trace 导出
        if not xlsx_path:
            xlsx_path = f"{output}/{model}_config.xlsx"
        records = get_records()
        if not records:
            logger.warning("没有 trace 记录，请先运行算子")
            # 仍然创建空模板
            create_template(xlsx_path)
            print(f"生成空模板 (无记录): {xlsx_path}")
            return 0
        export_xlsx(xlsx_path, records)
        print(f"导出到 xlsx: {xlsx_path} ({len(records)} 条记录)")
        return 0

    elif subaction in ("import", "i"):
        # 生成 Python 代码
        if not xlsx_path:
            logger.error("请指定 xlsx 文件: compare xlsx import --xlsx=config.xlsx")
            return 1
        out_py = output if output.endswith(".py") else f"{output}/generated_{model}.py"
        code = import_xlsx(xlsx_path, out_py)
        print(f"生成 Python 代码: {out_py}")
        return 0

    elif subaction in ("run", "r"):
        # 运行比数
        if not xlsx_path:
            logger.error("请指定 xlsx 文件: compare xlsx run --xlsx=config.xlsx")
            return 1
        results = run_xlsx(xlsx_path, output, format)

        # 统计
        pass_count = sum(1 for r in results if r.get("status") == "PASS")
        fail_count = sum(1 for r in results if r.get("status") == "FAIL")
        skip_count = sum(1 for r in results if r.get("status") in ("SKIP", "PENDING", "ERROR"))

        print(f"比数完成: PASS={pass_count}, FAIL={fail_count}, SKIP/PENDING={skip_count}")
        print(f"结果已更新到: {xlsx_path}")
        return 0 if fail_count == 0 else 1

    elif subaction in ("ops", "o"):
        # 列出可用算子
        print("可用算子:")
        for op in list_ops():
            print(f"  - {op}")
        return 0

    else:
        logger.error(f"未知 xlsx 子命令: {subaction}")
        print("可用 xlsx 子命令: template(t), export(e), import(i), run(r), ops(o)")
        return 1
