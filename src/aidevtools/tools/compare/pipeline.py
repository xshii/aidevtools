"""比对 Pipeline

Pipeline 结构: Load → Compare → Report → Export
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from aidevtools.tools.compare.diff import DiffResult


@dataclass
class CompareCase:
    """单个比对用例"""
    op_name: str
    golden_path: str
    result_path: str
    dtype: Optional[str] = None
    shape: Optional[tuple] = None
    qtype: str = ""  # 模糊比对类型

    # 加载后的数据
    golden: Optional[np.ndarray] = None
    result: Optional[np.ndarray] = None

    # 比对结果
    diff: Optional[DiffResult] = None
    blocks: List[Dict] = field(default_factory=list)

    # 输出路径
    report_path: str = ""
    heatmap_path: str = ""

    # 状态
    status: str = "PENDING"  # PENDING, PASS, FAIL, SKIP, ERROR
    error: str = ""


@dataclass
class PipelineResult:
    """Pipeline 执行结果"""
    cases: List[CompareCase] = field(default_factory=list)
    output_dir: str = ""

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.cases if c.status == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.cases if c.status == "FAIL")

    @property
    def total(self) -> int:
        return len(self.cases)


# ==================== Pipeline 步骤 ====================

def load_case(case: CompareCase, format: str = "raw") -> CompareCase:
    """加载数据"""
    from aidevtools.formats.base import load
    from aidevtools.core.log import logger

    if not Path(case.golden_path).exists():
        case.status = "ERROR"
        case.error = f"golden 不存在: {case.golden_path}"
        logger.error(f"[{case.op_name}] {case.error}")
        return case

    if not Path(case.result_path).exists():
        case.status = "SKIP"
        case.error = f"result 不存在: {case.result_path}"
        logger.warn(f"[{case.op_name}] {case.error}")
        return case

    load_kwargs = {}
    if case.dtype:
        load_kwargs["dtype"] = getattr(np, case.dtype) if isinstance(case.dtype, str) else case.dtype
    if case.shape:
        load_kwargs["shape"] = case.shape

    try:
        case.golden = load(case.golden_path, format=format, **load_kwargs)
        case.result = load(case.result_path, format=format, **load_kwargs)
    except Exception as e:
        case.status = "ERROR"
        case.error = str(e)
        logger.error(f"[{case.op_name}] 加载失败: {e}")

    return case


def compare_case(case: CompareCase, atol: float = 1e-5, rtol: float = 1e-5,
                 block_size: int = 256) -> CompareCase:
    """执行比对"""
    from aidevtools.tools.compare.diff import compare_full, compare_block
    from aidevtools.core.log import logger

    if case.status in ("ERROR", "SKIP"):
        return case

    if case.golden is None or case.result is None:
        case.status = "ERROR"
        case.error = "数据未加载"
        return case

    case.diff = compare_full(case.golden, case.result, atol=atol, rtol=rtol)
    case.blocks = compare_block(case.golden, case.result, block_size=block_size, threshold=atol)
    case.status = "PASS" if case.diff.passed else "FAIL"

    logger.info(f"[{case.op_name}] {case.status} qsnr={case.diff.qsnr:.2f}dB")
    return case


def report_case(case: CompareCase, output_dir: str) -> CompareCase:
    """生成报告"""
    from aidevtools.tools.compare.report import gen_report, gen_heatmap_svg

    if case.status in ("ERROR", "SKIP") or case.diff is None:
        return case

    out_path = Path(output_dir)
    case.report_path = gen_report(case.op_name, case.diff, case.blocks, str(out_path))

    heatmap_path = out_path / case.op_name / "heatmap.svg"
    gen_heatmap_svg(case.blocks, str(heatmap_path))
    case.heatmap_path = str(heatmap_path)

    return case


def export_failed(case: CompareCase, output_dir: str, qsnr_threshold: float = 20.0) -> CompareCase:
    """导出失败用例"""
    from aidevtools.tools.compare.export import export_failed_cases

    if case.status != "FAIL" or case.golden is None or case.result is None:
        return case

    if not case.blocks:
        return case

    export_failed_cases(
        case.golden, case.result, case.blocks,
        output_dir, case.op_name, qsnr_threshold
    )
    return case


# ==================== Pipeline 执行器 ====================

def run_pipeline(cases: List[CompareCase], output_dir: str,
                 format: str = "raw", atol: float = 1e-5, rtol: float = 1e-5,
                 block_size: int = 256, export_failed_cases: bool = True,
                 qsnr_threshold: float = 20.0) -> PipelineResult:
    """
    执行完整 pipeline

    Pipeline: Load → Compare → Report → Export
    """
    from aidevtools.core.log import logger

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for case in cases:
        # Step 1: Load
        load_case(case, format=format)

        # Step 2: Compare
        compare_case(case, atol=atol, rtol=rtol, block_size=block_size)

        # Step 3: Report
        report_case(case, output_dir)

        # Step 4: Export (可选)
        if export_failed_cases:
            export_failed(case, output_dir, qsnr_threshold)

    result = PipelineResult(cases=cases, output_dir=output_dir)
    logger.info(f"Pipeline 完成: PASS={result.pass_count}, FAIL={result.fail_count}, TOTAL={result.total}")

    return result
