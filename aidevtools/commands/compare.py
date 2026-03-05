# pylint: disable=unused-argument
# kwargs 为 CLI 框架预留参数
"""比数命令"""

import numpy as np
from prettycli import command

from aidevtools.core.log import logger
from aidevtools.core.utils import parse_list, parse_shape
from aidevtools.formats.base import load
from aidevtools.formats.quantize import list_quantize
from aidevtools.ops.base import clear as do_clear
from aidevtools.ops.base import dump as do_dump
from aidevtools.ops.base import get_records

try:
    from aidevtools.compare.strategy import (
        BitAnalysisStrategy,
        BitLayout,
        FP32,
        FP16,
    )
    BITWISE_AVAILABLE = True
except ImportError:
    BITWISE_AVAILABLE = False


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


def _action_qtypes(**kwargs):
    """列出支持的量化类型"""
    print("支持的量化类型:")
    for qtype in list_quantize():
        print(f"  - {qtype}")
    return 0


def _action_convert(golden, output, qtype, shape, target_dtype, **kwargs):
    """类型转换导出"""
    from aidevtools.formats.quantize import quantize
    from aidevtools.formats.filename_parser import parse_filename, infer_fmt

    if not golden:
        logger.error("请指定输入文件: compare convert --golden=a.bin --target_dtype=float16")
        return 1
    if not target_dtype:
        logger.error("请指定目标类型: --target_dtype=float16 (可用: compare qtypes 查看)")
        return 1
    out_path = output if output != "./workspace" else golden.rsplit(".", 1)[0] + f"_{target_dtype}.bin"

    # 自动解读
    parsed = parse_filename(golden)
    fmt = kwargs.get("format") or (parsed["fmt"] if parsed else infer_fmt(golden))
    qt = qtype or (parsed["qtype"] if parsed else "float32")
    sh = parse_shape(shape) if shape else (parsed["shape"] if parsed else None)

    data = load(golden, fmt=fmt, qtype=qt, shape=sh)

    try:
        converted, meta = quantize(data, target_dtype)
    except (NotImplementedError, ValueError) as e:
        logger.error(str(e))
        return 1

    converted.tofile(out_path)
    print(f"转换: {qt} → {target_dtype}")
    print(f"shape: {data.shape}")
    print(f"输出: {out_path}")
    if meta:
        print(f"meta: {meta}")
    return 0


def _per_element_layout(spec):
    """从 bit_layout 字符串解析 element 0 的 BitLayout"""
    bl = spec.bit_layout
    if not bl:
        return None
    sign_bits = exp_bits = mant_bits = 0
    for token in bl.split():
        if token == "S0":
            sign_bits = 1
        elif token.startswith("E0*"):
            exp_bits = int(token.split("*")[1])
        elif token == "E0":
            exp_bits = 1
        elif token.startswith("M0*"):
            mant_bits = int(token.split("*")[1])
        elif token == "M0":
            mant_bits = 1
    if sign_bits + exp_bits + mant_bits == 0:
        return None
    return BitLayout(sign_bits, exp_bits, mant_bits, spec.name)


def _strip_shared_bytes(raw, spec, shape):
    """block format (block_size>1) 时去掉共享指数字节，只留 per-element 数据"""
    if spec.block_size <= 1:
        return raw
    num_elements = int(np.prod(shape)) if shape is not None else len(raw)
    num_blocks = int(np.ceil(num_elements / spec.block_size))
    element_bytes = np.dtype(spec.storage_dtype).itemsize
    exp_bytes_per_block = spec.bytes_per_block - spec.block_size * element_bytes
    exp_total = num_blocks * exp_bytes_per_block
    return raw[exp_total:]


def _action_diff(golden, result, format, qtype, shape, engine, **kwargs):
    """渐进式多级比数 (ProgressiveStrategy)

    架构:
        raw_g, raw_r   ──┐
             │           ├→ CompareContext (双路径数据)
             ▼ deq       │
        fp32_g, fp32_r ──┘
                          ↓
        ProgressiveStrategy 自动分级:
          L1: Exact + BitXor       → exact 过就停
          L2: Fuzzy + Sanity       → fuzzy 过就停
          L3: BitAnalysis + Blocked → 深度定位

    未指定 qtype / shape / format 时，从文件名自动解读。
    """
    if not golden or not result:
        logger.error(
            "请指定文件: compare diff --golden=softmax_bfp8_2x16x64.txt "
            "--result=softmax_bfp8_2x16x64_result.txt"
        )
        return 1

    from aidevtools.compare.engine import CompareEngine
    from aidevtools.compare.report import print_strategy_table
    from aidevtools.compare.strategy import BlockedStrategy
    from aidevtools.compare.strategy.base import CompareContext
    from aidevtools.compare.strategy.bit_analysis import BitAnalysisResult
    from aidevtools.formats._registry import get as get_fmt
    from aidevtools.formats.block_format import is_block_format, get_block_format
    from aidevtools.formats.filename_parser import parse_filename, infer_fmt

    # ── 自动解读: 从 golden 文件名提取 qtype / shape / fmt ──
    parsed = parse_filename(golden)
    fmt = format if format != "raw" else (parsed["fmt"] if parsed else infer_fmt(golden))
    qt = qtype or (parsed["qtype"] if parsed else None)
    sh = parse_shape(shape) if shape else (parsed["shape"] if parsed else None)

    # ── 加载双路径数据 ──
    # Path A: 源格式原始字节
    if is_block_format(qt):
        spec = get_block_format(qt)
        raw_dtype = spec.storage_dtype
    else:
        spec = None
        raw_dtype = np.float32
    raw_g = get_fmt(fmt).load(golden, dtype=raw_dtype)
    raw_r = get_fmt(fmt).load(result, dtype=raw_dtype)

    # Path B: 反量化为 fp32
    fp32_g = load(golden, fmt=fmt, qtype=qt, shape=sh)
    fp32_r = load(result, fmt=fmt, qtype=qt, shape=sh)

    # ── 构造 metadata: 格式感知信息 ──
    metadata = {}
    if spec is not None:
        elem_layout = _per_element_layout(spec)
        if elem_layout is not None:
            metadata["bit_layout"] = elem_layout
        metadata["source_block_size"] = spec.block_size

    # ── 构造 raw 数据: strip 共享指数字节 ──
    raw_golden_elem = _strip_shared_bytes(raw_g, spec, sh) if spec else raw_g
    raw_dut_elem = _strip_shared_bytes(raw_r, spec, sh) if spec else raw_r

    # ── 选择引擎, 构造 context, 执行 ──
    _engines = {
        "standard": CompareEngine.progressive,
        "progressive": CompareEngine.progressive,
        "quick": CompareEngine.quick_then_deep,
        "deep": CompareEngine.deep,
        "minimal": CompareEngine.minimal,
    }
    factory = _engines.get(engine or "standard", CompareEngine.progressive)
    eng = factory()
    results = eng.run(
        dut=fp32_r,
        golden=fp32_g,
        metadata={
            **metadata,
            "_raw_golden": raw_golden_elem,
            "_raw_dut": raw_dut_elem,
        },
    )

    # ── 输出报告 ──
    name = parsed["op"] if parsed else (golden.rsplit("/", 1)[-1] if "/" in golden else golden)
    print_strategy_table([results], names=[name])

    # Bit Analysis 详细输出 (key = "bit_analysis_<format>")
    bit_result = None
    for k, v in results.items():
        if k.startswith("bit_analysis") and isinstance(v, BitAnalysisResult):
            bit_result = v
            break
    if bit_result is not None:
        BitAnalysisStrategy.print_result(bit_result, name)

    # Block Heatmap 详细输出 (key 含 block_size 后缀, 如 "blocked_1024")
    blocked_result = None
    for k, v in results.items():
        if k.startswith("blocked") and isinstance(v, list):
            blocked_result = v
            break
    if blocked_result:
        BlockedStrategy.print_heatmap(blocked_result)

    # 判定退出码
    fuzzy = results.get("fuzzy_qnt") or results.get("fuzzy_pure")
    exact = results.get("exact")
    passed = fuzzy.passed if fuzzy else (exact.passed if exact else False)
    return 0 if passed else 1


# Action 分发表
_ACTIONS = {
    "dump": _action_dump,
    "clear": _action_clear,
    "convert": _action_convert,
    "qtypes": _action_qtypes,
    "diff": _action_diff,
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
    shape: str = "",
    target_dtype: str = "",
    ops: str = "",
    qtype: str = "",
    engine: str = "standard",
):
    """
    比数工具

    用法:
        compare <action>        执行指定步骤

    子命令:
        diff       多级比数 (自动解读文件名 / hex-text + dequantize + 报告)
        convert    类型转换导出
        dump       导出 Golden 数据
        clear      清空 Golden 记录
        qtypes     列出支持的量化类型

    xlsx 子命令:
        xlsx template   生成 xlsx 空模板
        xlsx export     从 trace 导出到 xlsx
        xlsx import     从 xlsx 生成 Python 代码
        xlsx run        从 xlsx 运行比数
        xlsx ops        列出可用算子

    参数:
        --qtype            DUT 量化类型 (bfp8/gfloat8/...) 不指定则从文件名解读
        --shape            输出 shape (2,16,64) 不指定则从文件名解读
        --format           文件格式 (hex_text/raw/numpy) 不指定则从扩展名解读
        --engine           比对引擎 (standard/quick/deep/minimal)
        --target_dtype     转换目标类型 (float16/bfloat16/...)
        --xlsx=xxx.xlsx    指定 xlsx 文件
        --ops=linear,relu  限定算子列表（xlsx template 用）

    文件名约定:
        {op}_{qtype}_{NxMxK}.txt              — golden
        {op}_{qtype}_{NxMxK}_result.txt        — result

    示例:
        compare diff --golden=softmax_bfp8_2x16x64.txt --result=softmax_bfp8_2x16x64_result.txt
        compare diff --golden=a.txt --result=b.txt --qtype=bfp8 --shape=2,16,64 --format=hex_text
        compare convert --golden=a.bin --target_dtype=float16
        compare qtypes
    """
    # 无 action 时显示帮助
    if not action:
        print("请指定子命令，例如: compare diff --golden=a.txt --result=b.txt")
        print("查看帮助: compare --help")
        return 1

    # 使用分发表处理 action
    if action in _ACTIONS:
        return _ACTIONS[action](
            output=output,
            format=format,
            golden=golden,
            result=result,
            shape=shape,
            target_dtype=target_dtype,
            qtype=qtype,
            engine=engine,
        )
    if action == "xlsx":
        return _handle_xlsx(subaction, xlsx, output, model, fmt=format, ops_str=ops)
    logger.error(f"未知子命令: {action}")
    print("可用子命令: diff, convert, dump, clear, qtypes, xlsx")
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
