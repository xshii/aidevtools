"""FSM 比对报告生成

本模块提供比对结果的报告生成和可视化功能。
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json

import numpy as np

from aidevtools.core.log import logger
from aidevtools.fsm.hooks import (
    StepResult,
    SegmentResult,
    ModelResult,
    CompareGranularity,
)
from aidevtools.fsm.validator import ValidationResult, ValidationStatus


@dataclass
class FSMCompareReport:
    """FSM 比对报告"""
    # 基本信息
    report_name: str
    model_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    granularity: CompareGranularity = CompareGranularity.SEGMENT

    # 统计信息
    total_ops: int = 0
    passed_ops: int = 0
    failed_ops: int = 0
    quant_issue_ops: int = 0

    # 详细结果
    step_results: List[StepResult] = field(default_factory=list)
    segment_results: List[SegmentResult] = field(default_factory=list)
    model_results: List[ModelResult] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)

    # 误差统计
    max_qsnr: float = 0.0
    min_qsnr: float = float("inf")
    avg_qsnr: float = 0.0
    max_abs_error: float = 0.0
    avg_abs_error: float = 0.0

    # 配置信息
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """通过率"""
        return self.passed_ops / self.total_ops if self.total_ops > 0 else 0.0

    @property
    def summary(self) -> str:
        """生成摘要"""
        return (
            f"Model: {self.model_name}\n"
            f"Pass Rate: {self.pass_rate:.1%} ({self.passed_ops}/{self.total_ops})\n"
            f"QSNR: avg={self.avg_qsnr:.1f}dB, min={self.min_qsnr:.1f}dB, max={self.max_qsnr:.1f}dB\n"
            f"Max Abs Error: {self.max_abs_error:.2e}"
        )


class ReportGenerator:
    """报告生成器"""

    def __init__(self, report_name: str = "fsm_compare"):
        self.report_name = report_name
        self._report: Optional[FSMCompareReport] = None

    def generate(
        self,
        model_name: str,
        step_results: Optional[List[StepResult]] = None,
        segment_results: Optional[List[SegmentResult]] = None,
        model_results: Optional[List[ModelResult]] = None,
        validation_results: Optional[List[ValidationResult]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> FSMCompareReport:
        """生成报告

        Args:
            model_name: 模型名称
            step_results: 单步比对结果
            segment_results: 片段级比对结果
            model_results: 全模型比对结果
            validation_results: 验证结果
            config: 配置信息

        Returns:
            FSMCompareReport
        """
        report = FSMCompareReport(
            report_name=self.report_name,
            model_name=model_name,
            config=config or {},
        )

        # 收集结果
        if step_results:
            report.step_results = step_results
            report.granularity = CompareGranularity.STEP
        if segment_results:
            report.segment_results = segment_results
            report.granularity = CompareGranularity.SEGMENT
        if model_results:
            report.model_results = model_results
            report.granularity = CompareGranularity.MODEL
        if validation_results:
            report.validation_results = validation_results

        # 计算统计
        self._compute_statistics(report)

        self._report = report
        return report

    def _compute_statistics(self, report: FSMCompareReport):
        """计算统计信息"""
        qsnrs = []
        abs_errors = []
        passed = 0
        failed = 0
        quant_issue = 0

        # 从 step_results 统计
        for r in report.step_results:
            qsnrs.append(r.diff.qsnr)
            abs_errors.append(r.diff.max_abs)
            if r.passed:
                passed += 1
            else:
                failed += 1

        # 从 segment_results 统计
        for r in report.segment_results:
            fr = r.full_result
            qsnrs.append(fr.fuzzy_qnt.qsnr)
            abs_errors.append(fr.fuzzy_qnt.max_abs)

            if fr.status == "PERFECT" or fr.status == "PASS":
                passed += 1
            elif fr.status == "QUANT_ISSUE":
                quant_issue += 1
            else:
                failed += 1

        # 从 validation_results 统计
        for r in report.validation_results:
            if r.dut_vs_golden_a:
                qsnrs.append(r.dut_vs_golden_a.qsnr)
                abs_errors.append(r.dut_vs_golden_a.max_abs)

            if r.status == ValidationStatus.DUT_CORRECT:
                passed += 1
            elif r.status == ValidationStatus.QUANTIZATION_ERROR:
                quant_issue += 1
            elif r.status in (ValidationStatus.DUT_FUNCTIONAL_ERROR,
                              ValidationStatus.GOLDEN_MISMATCH):
                failed += 1

        # 汇总
        report.total_ops = passed + failed + quant_issue
        report.passed_ops = passed
        report.failed_ops = failed
        report.quant_issue_ops = quant_issue

        if qsnrs:
            qsnrs_finite = [q for q in qsnrs if q != float("inf")]
            if qsnrs_finite:
                report.max_qsnr = max(qsnrs_finite)
                report.min_qsnr = min(qsnrs_finite)
                report.avg_qsnr = sum(qsnrs_finite) / len(qsnrs_finite)

        if abs_errors:
            report.max_abs_error = max(abs_errors)
            report.avg_abs_error = sum(abs_errors) / len(abs_errors)


def generate_report(
    model_name: str,
    results: List[Any],
    config: Optional[Dict[str, Any]] = None,
) -> FSMCompareReport:
    """便捷函数：生成报告

    自动检测结果类型并生成对应报告。
    """
    generator = ReportGenerator()

    step_results = []
    segment_results = []
    model_results = []
    validation_results = []

    for r in results:
        if isinstance(r, StepResult):
            step_results.append(r)
        elif isinstance(r, SegmentResult):
            segment_results.append(r)
        elif isinstance(r, ModelResult):
            model_results.append(r)
        elif isinstance(r, ValidationResult):
            validation_results.append(r)

    return generator.generate(
        model_name=model_name,
        step_results=step_results or None,
        segment_results=segment_results or None,
        model_results=model_results or None,
        validation_results=validation_results or None,
        config=config,
    )


def print_report(report: FSMCompareReport, detailed: bool = False):
    """打印报告到控制台"""
    print("\n" + "=" * 80)
    print(f"  FSM Golden Comparison Report: {report.report_name}")
    print("=" * 80)

    print(f"\nModel: {report.model_name}")
    print(f"Time:  {report.timestamp}")
    print(f"Granularity: {report.granularity.value}")

    print("\n" + "-" * 40)
    print("  Summary")
    print("-" * 40)
    print(f"  Total Ops:      {report.total_ops}")
    print(f"  Passed:         {report.passed_ops}")
    print(f"  Failed:         {report.failed_ops}")
    print(f"  Quant Issues:   {report.quant_issue_ops}")
    print(f"  Pass Rate:      {report.pass_rate:.1%}")

    print("\n" + "-" * 40)
    print("  Error Statistics")
    print("-" * 40)
    print(f"  QSNR (avg/min/max): {report.avg_qsnr:.1f} / {report.min_qsnr:.1f} / {report.max_qsnr:.1f} dB")
    print(f"  Max Abs Error:      {report.max_abs_error:.2e}")
    print(f"  Avg Abs Error:      {report.avg_abs_error:.2e}")

    if detailed:
        # 打印详细结果
        if report.segment_results:
            print("\n" + "-" * 40)
            print("  Detailed Results (Segment)")
            print("-" * 40)
            print(f"{'Op Name':<20} {'Status':<12} {'QSNR':>10} {'Cosine':>10} {'MaxAbs':>12}")
            print("-" * 70)

            for r in report.segment_results:
                fr = r.full_result
                print(
                    f"{r.segment_name:<20} {fr.status:<12} "
                    f"{fr.fuzzy_qnt.qsnr:>10.1f} "
                    f"{fr.fuzzy_qnt.cosine:>10.6f} "
                    f"{fr.fuzzy_qnt.max_abs:>12.2e}"
                )

        if report.validation_results:
            print("\n" + "-" * 40)
            print("  Detailed Results (Validation)")
            print("-" * 40)

            for r in report.validation_results:
                status_str = r.status.value
                qsnr = r.dut_vs_golden_a.qsnr if r.dut_vs_golden_a else 0.0
                print(f"{r.op_name}_{r.op_id}: {status_str} (QSNR={qsnr:.1f}dB)")
                if r.diagnosis:
                    print(f"    -> {r.diagnosis}")

    print("\n" + "=" * 80 + "\n")


def export_report_json(report: FSMCompareReport, output_path: str):
    """导出报告为 JSON"""
    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "__dict__"):
            return {k: _serialize(v) for k, v in obj.__dict__.items()
                    if not k.startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if hasattr(obj, "value"):  # Enum
            return obj.value
        return obj

    data = _serialize(report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Report exported to {output_path}")


def export_report_html(report: FSMCompareReport, output_path: str):
    """导出报告为 HTML"""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>FSM Compare Report: {report.report_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .stats {{ display: flex; gap: 20px; margin: 10px 0; }}
        .stat {{ background: white; padding: 10px 20px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
        .stat-label {{ color: #666; font-size: 12px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .pass {{ color: #4CAF50; }}
        .fail {{ color: #f44336; }}
        .quant {{ color: #FF9800; }}
    </style>
</head>
<body>
    <h1>FSM Golden Comparison Report</h1>

    <div class="summary">
        <p><strong>Model:</strong> {report.model_name}</p>
        <p><strong>Time:</strong> {report.timestamp}</p>
        <p><strong>Granularity:</strong> {report.granularity.value}</p>
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{report.total_ops}</div>
            <div class="stat-label">Total Ops</div>
        </div>
        <div class="stat">
            <div class="stat-value pass">{report.passed_ops}</div>
            <div class="stat-label">Passed</div>
        </div>
        <div class="stat">
            <div class="stat-value fail">{report.failed_ops}</div>
            <div class="stat-label">Failed</div>
        </div>
        <div class="stat">
            <div class="stat-value quant">{report.quant_issue_ops}</div>
            <div class="stat-label">Quant Issues</div>
        </div>
        <div class="stat">
            <div class="stat-value">{report.pass_rate:.1%}</div>
            <div class="stat-label">Pass Rate</div>
        </div>
    </div>

    <h2>Error Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Avg QSNR</td><td>{report.avg_qsnr:.1f} dB</td></tr>
        <tr><td>Min QSNR</td><td>{report.min_qsnr:.1f} dB</td></tr>
        <tr><td>Max QSNR</td><td>{report.max_qsnr:.1f} dB</td></tr>
        <tr><td>Max Abs Error</td><td>{report.max_abs_error:.2e}</td></tr>
        <tr><td>Avg Abs Error</td><td>{report.avg_abs_error:.2e}</td></tr>
    </table>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Op Name</th>
            <th>Status</th>
            <th>QSNR (dB)</th>
            <th>Cosine</th>
            <th>Max Abs Error</th>
        </tr>
"""

    # 添加详细结果行
    for r in report.segment_results:
        fr = r.full_result
        status_class = "pass" if fr.status in ("PERFECT", "PASS") else "fail" if fr.status == "FAIL" else "quant"
        html += f"""        <tr>
            <td>{r.segment_name}</td>
            <td class="{status_class}">{fr.status}</td>
            <td>{fr.fuzzy_qnt.qsnr:.1f}</td>
            <td>{fr.fuzzy_qnt.cosine:.6f}</td>
            <td>{fr.fuzzy_qnt.max_abs:.2e}</td>
        </tr>
"""

    for r in report.validation_results:
        status_class = "pass" if r.passed else "fail"
        qsnr = r.dut_vs_golden_a.qsnr if r.dut_vs_golden_a else 0.0
        cosine = r.dut_vs_golden_a.cosine if r.dut_vs_golden_a else 0.0
        max_abs = r.dut_vs_golden_a.max_abs if r.dut_vs_golden_a else 0.0
        html += f"""        <tr>
            <td>{r.op_name}_{r.op_id}</td>
            <td class="{status_class}">{r.status.value}</td>
            <td>{qsnr:.1f}</td>
            <td>{cosine:.6f}</td>
            <td>{max_abs:.2e}</td>
        </tr>
"""

    html += """    </table>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report exported to {output_path}")
