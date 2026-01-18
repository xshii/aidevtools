"""比数命令"""
import numpy as np
from prettycli import command

from aidevtools.tools.compare.diff import compare_full
from aidevtools.tools.compare.runner import run_compare, archive
from aidevtools.tools.compare.quantize import list_quantize
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
    shape: str = "",
    fuzzy: str = "false",
    target_dtype: str = "",
):
    """
    比数工具

    用法:
        compare                 一键式执行完整流程 (csv→dump→run→archive)
        compare <action>        执行指定步骤

    子命令:
        1/csv      生成 compare.csv 配置表
        2/dump     导出 Golden 数据
        3/run      运行比数
        4/archive  打包比数结果
        c/clear    清空 Golden 记录
        s/single   单次比对两个文件
        f/fuzzy    模糊比对（跳过 bit 级比对）
        t/convert  类型转换导出
        q/qtypes   列出支持的量化类型

    参数:
        --fuzzy=true       启用模糊比对模式
        --target_dtype     转换目标类型 (float16/bfloat16/...)

    示例:
        compare                              一键式流程
        compare 1 --output=./workspace       生成配置表
        compare 3 --csv=compare.csv          运行比数
        compare 3 --csv=compare.csv --op=conv1   单用例比数
        compare s --golden=a.bin --result=b.bin --dtype=float32 --shape=1,64,32,32
        compare f --golden=a.bin --result=b.bin --dtype=float32   模糊比对
        compare t --golden=a.bin --output=a_fp16.bin --target_dtype=float16   类型转换
        compare q                            列出量化类型

    模糊比对流程 (代码方式):
        from aidevtools.tools.compare.fuzzy import create_fuzzy_case

        case = create_fuzzy_case("conv1")
        case.set_input("x", input_data)
        case.set_weight("w", weight_data)
        case.set_compute(lambda inputs, weights: np.matmul(inputs["x"], weights["w"]))
        case.compute_golden()
        case.export(qtype="float16")  # 导出给仿真器
    """
    # 默认一键式流程
    if not action:
        logger.info("执行一键式比数流程...")
        csv_path = gen_csv(output, model)
        print(f"[1/4] 生成: {csv_path}")
        dump(output, format=format)
        print(f"[2/4] 导出 Golden 数据完成")
        run_compare(csv_path=csv_path, op_filter=None, mode_filter=None, atol=float(atol))
        print(f"[3/4] 比数完成")
        archive(csv_path)
        print(f"[4/4] 打包完成")
        return 0

    if action in ("1", "csv", "1csv"):
        csv_path = gen_csv(output, model)
        print(f"生成: {csv_path}")
        return 0

    elif action in ("2", "dump", "2dump"):
        dump(output, format=format)
        return 0

    elif action in ("3", "run", "3run"):
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

    elif action in ("4", "archive", "4archive"):
        if not csv:
            logger.error("请指定 csv 文件: compare archive --csv=xxx.csv")
            return 1
        archive(csv)
        return 0

    elif action in ("c", "clear"):
        clear()
        logger.info("Golden 记录已清空")
        return 0

    elif action in ("s", "single"):
        if not golden or not result:
            logger.error("请指定文件: compare s --golden=a.bin --result=b.bin --dtype=float32 --shape=1,64,32,32")
            return 1
        dt = getattr(np, dtype)
        sh = tuple(int(x) for x in shape.split(",")) if shape else None
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

    elif action in ("f", "fuzzy"):
        # 模糊比对：只做 QSNR/cosine，不做 bit 级比对
        if not golden or not result:
            logger.error("请指定文件: compare f --golden=a.bin --result=b.bin")
            return 1
        dt = getattr(np, dtype)
        sh = tuple(int(x) for x in shape.split(",")) if shape else None
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

    elif action in ("t", "convert"):
        # 类型转换导出
        from aidevtools.tools.compare.quantize import quantize

        if not golden:
            logger.error("请指定输入文件: compare t --golden=a.bin --output=out.bin --target_dtype=float16")
            return 1
        if not target_dtype:
            logger.error("请指定目标类型: --target_dtype=float16 (可用: compare q 查看)")
            return 1
        if not output or output == "./workspace":
            output = golden.replace(".bin", f"_{target_dtype}.bin")

        dt = getattr(np, dtype)
        sh = tuple(int(x) for x in shape.split(",")) if shape else None
        data = load(golden, format="raw", dtype=dt, shape=sh)

        # 使用量化框架转换
        try:
            converted, meta = quantize(data, target_dtype)
        except NotImplementedError as e:
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

    elif action in ("q", "qtypes"):
        # 列出支持的量化类型
        print("支持的量化类型:")
        for qtype in list_quantize():
            print(f"  - {qtype}")
        return 0

    else:
        logger.error(f"未知子命令: {action}")
        print("可用子命令: 1/csv, 2/dump, 3/run, 4/archive, c/clear, s/single, f/fuzzy, t/convert, q/qtypes")
        return 1
