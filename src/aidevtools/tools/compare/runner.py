"""比数运行器"""
import csv
import zipfile
import numpy as np
from pathlib import Path
from typing import Optional

from aidevtools.core.log import logger
from aidevtools.formats.base import load
from aidevtools.tools.compare.diff import compare_full, compare_block, compare_bit, calc_qsnr, calc_cosine
from aidevtools.tools.compare.report import gen_report, gen_heatmap_svg

def run_compare(csv_path: str, output_dir: str = None,
                mode_filter: str = None, op_filter: str = None,
                block_size: int = 256, atol: float = 1e-5, rtol: float = 1e-5,
                format: str = "raw", dtype=None, shape=None):
    """
    运行比数

    Args:
        csv_path: compare.csv 路径
        output_dir: 输出目录 (默认同 csv 目录)
        mode_filter: 只跑指定 mode (single/chain/full)
        op_filter: 只跑指定算子
        block_size: 分块大小 (bytes)
        atol: 绝对误差阈值
        rtol: 相对误差阈值
        format: 数据格式
        dtype: 数据类型 (raw 格式需要)
        shape: 数据形状 (raw 格式需要)
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / "details"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []

    for row in rows:
        op_name = row["op_name"]

        # 过滤
        if row.get("skip", "").lower() == "true":
            logger.info(f"[{op_name}] 跳过")
            continue
        if mode_filter and row.get("mode") != mode_filter:
            continue
        if op_filter and op_name != op_filter:
            continue

        golden_path = row["golden_bin"]
        result_path = row["result_bin"]

        if not Path(golden_path).exists():
            logger.error(f"[{op_name}] golden 不存在: {golden_path}")
            row["status"] = "ERROR"
            continue

        if not Path(result_path).exists():
            logger.warn(f"[{op_name}] result 不存在: {result_path}")
            row["status"] = "SKIP"
            continue

        # 加载数据 (优先从 csv 读取 dtype/shape)
        row_dtype = row.get("dtype") or dtype
        row_shape = row.get("shape") or shape

        load_kwargs = {}
        if row_dtype:
            import numpy as np
            load_kwargs["dtype"] = getattr(np, row_dtype) if isinstance(row_dtype, str) else row_dtype
        if row_shape:
            if isinstance(row_shape, str):
                load_kwargs["shape"] = tuple(int(x) for x in row_shape.split(","))
            else:
                load_kwargs["shape"] = row_shape

        golden = load(golden_path, format=format, **load_kwargs)
        result = load(result_path, format=format, **load_kwargs)

        # 精确比对
        logger.info(f"[{op_name}] 精确比对...")

        diff_result = compare_full(golden, result, atol=atol, rtol=rtol)
        blocks = compare_block(golden, result, block_size=block_size, threshold=atol)

        detail_path = gen_report(op_name, diff_result, blocks, str(output_dir))
        svg_path = output_dir / op_name / "heatmap.svg"
        gen_heatmap_svg(blocks, str(svg_path))

        row["status"] = "PASS" if diff_result.passed else "FAIL"
        row["max_abs"] = f"{diff_result.max_abs:.6e}"
        row["qsnr"] = f"{diff_result.qsnr:.2f}"
        row["cosine"] = f"{diff_result.cosine:.6f}"
        row["detail_link"] = str(Path(detail_path).relative_to(csv_path.parent))

        status = "PASS" if diff_result.passed else "FAIL"
        logger.info(f"[{op_name}] 精确: {status} qsnr={diff_result.qsnr:.2f}dB")

        # 模糊比对（如果指定了 qtype）
        qtype = row.get("qtype", "").strip()
        if qtype:
            logger.info(f"[{op_name}] 模糊比对 (qtype={qtype})...")

            g_f64 = golden.astype(np.float64).flatten()
            r_f64 = result.astype(np.float64).flatten()

            fuzzy_qsnr = calc_qsnr(golden, result)
            fuzzy_cosine = calc_cosine(golden, result)
            fuzzy_max_abs = np.max(np.abs(g_f64 - r_f64))

            row["fuzzy_max_abs"] = f"{fuzzy_max_abs:.6e}"
            row["fuzzy_qsnr"] = f"{fuzzy_qsnr:.2f}"
            row["fuzzy_cosine"] = f"{fuzzy_cosine:.6f}"

            logger.info(f"[{op_name}] 模糊: qsnr={fuzzy_qsnr:.2f}dB cosine={fuzzy_cosine:.6f}")

        results.append(row)

    # 写结果到单独文件
    if results:
        result_path = csv_path.with_name(csv_path.stem + "_result.csv")
        result_fields = ["op_name", "status", "max_abs", "qsnr", "cosine",
                         "fuzzy_max_abs", "fuzzy_qsnr", "fuzzy_cosine", "detail_link"]

        with open(result_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result_fields)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, "") for k in result_fields})

        logger.info(f"比数完成，结果: {result_path}")

def archive(csv_path: str, output_path: str = None):
    """打包归档"""
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = csv_path.with_suffix(".zip")

    base_dir = csv_path.parent

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # CSV
        zf.write(csv_path, csv_path.name)

        # details 目录
        details_dir = base_dir / "details"
        if details_dir.exists():
            for f in details_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(base_dir))

    logger.info(f"归档完成: {output_path}")
    return str(output_path)
