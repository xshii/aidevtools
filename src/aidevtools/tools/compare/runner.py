"""比数运行器"""
import csv
import zipfile
from pathlib import Path
from typing import Optional

from aidevtools.core.log import logger
from aidevtools.formats.base import load
from aidevtools.tools.compare.diff import compare_full, compare_block, compare_bit
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

        # 加载数据
        load_kwargs = {}
        if dtype:
            load_kwargs["dtype"] = dtype
        if shape:
            load_kwargs["shape"] = shape

        golden = load(golden_path, format=format, **load_kwargs)
        result = load(result_path, format=format, **load_kwargs)

        # 比对
        logger.info(f"[{op_name}] 开始比对...")

        # 完整级
        diff_result = compare_full(golden, result, atol=atol, rtol=rtol)

        # 分块级
        blocks = compare_block(golden, result, block_size=block_size, threshold=atol)

        # 生成报告
        detail_path = gen_report(op_name, diff_result, blocks, str(output_dir))

        # 生成热力图
        svg_path = output_dir / op_name / "heatmap.svg"
        gen_heatmap_svg(blocks, str(svg_path))

        # 更新 row
        row["status"] = "PASS" if diff_result.passed else "FAIL"
        row["max_abs"] = f"{diff_result.max_abs:.6e}"
        row["qsnr"] = f"{diff_result.qsnr:.2f}"
        row["detail_link"] = str(Path(detail_path).relative_to(csv_path.parent))

        status = "PASS" if diff_result.passed else "FAIL"
        logger.info(f"[{op_name}] {status} (qsnr={diff_result.qsnr:.2f}dB)")

        results.append(row)

    # 写回 CSV
    if results:
        fields = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    logger.info(f"比数完成，结果已更新到 {csv_path}")

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
