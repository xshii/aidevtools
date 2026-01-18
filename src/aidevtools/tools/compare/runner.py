"""比数运行器

使用 Pipeline 结构: Load → Compare → Report → Export
"""
import csv
import zipfile
from pathlib import Path
from typing import Optional, List

from aidevtools.core.log import logger
from aidevtools.tools.compare.pipeline import (
    CompareCase, PipelineResult, run_pipeline
)


def run_compare(csv_path: str, output_dir: str = None,
                mode_filter: str = None, op_filter: str = None,
                block_size: int = 256, atol: float = 1e-5, rtol: float = 1e-5,
                format: str = "raw", dtype=None, shape=None,
                export_failed: bool = True) -> PipelineResult:
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
        export_failed: 是否导出失败用例
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / "details"
    output_dir = Path(output_dir)

    # 从 CSV 构建用例
    cases = _load_cases_from_csv(csv_path, mode_filter, op_filter, dtype, shape)

    if not cases:
        logger.warn("没有需要比对的用例")
        return PipelineResult()

    # 执行 pipeline
    result = run_pipeline(
        cases=cases,
        output_dir=str(output_dir),
        format=format,
        atol=atol,
        rtol=rtol,
        block_size=block_size,
        export_failed_cases=export_failed,
    )

    # 写结果 CSV
    _write_result_csv(csv_path, result)

    return result


def _load_cases_from_csv(csv_path: Path, mode_filter: str, op_filter: str,
                         dtype, shape) -> List[CompareCase]:
    """从 CSV 加载用例"""
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    cases = []
    for row in rows:
        op_name = row["op_name"]

        # 过滤
        if row.get("skip", "").lower() == "true":
            logger.info(f"[{op_name}] 跳过 (skip=true)")
            continue
        if mode_filter and row.get("mode") != mode_filter:
            continue
        if op_filter and op_name != op_filter:
            continue

        # 解析 dtype/shape
        row_dtype = row.get("dtype") or dtype
        row_shape = row.get("shape") or shape
        if isinstance(row_shape, str) and row_shape:
            row_shape = tuple(int(x) for x in row_shape.split(","))

        case = CompareCase(
            op_name=op_name,
            golden_path=row["golden_bin"],
            result_path=row["result_bin"],
            dtype=row_dtype,
            shape=row_shape,
            qtype=row.get("qtype", "").strip(),
        )
        cases.append(case)

    return cases


def _write_result_csv(csv_path: Path, result: PipelineResult):
    """写结果 CSV"""
    if not result.cases:
        return

    result_path = csv_path.with_name(csv_path.stem + "_result.csv")
    fields = ["op_name", "status", "max_abs", "qsnr", "cosine", "report", "error"]

    with open(result_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for case in result.cases:
            row = {
                "op_name": case.op_name,
                "status": case.status,
                "max_abs": f"{case.diff.max_abs:.6e}" if case.diff else "",
                "qsnr": f"{case.diff.qsnr:.2f}" if case.diff else "",
                "cosine": f"{case.diff.cosine:.6f}" if case.diff else "",
                "report": Path(case.report_path).name if case.report_path else "",
                "error": case.error,
            }
            writer.writerow(row)

    logger.info(f"结果: {result_path}")


def archive(csv_path: str, output_path: str = None) -> str:
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

    logger.info(f"归档: {output_path}")
    return str(output_path)
