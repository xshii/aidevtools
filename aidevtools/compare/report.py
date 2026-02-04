"""
比对报告生成

支持文本报告、JSON 报告和表格输出。
"""

import json
from pathlib import Path
from typing import List, Optional

from .types import CompareResult, CompareStatus


def format_result_row(result: CompareResult) -> str:
    """格式化单行结果"""
    marks = (
        "Y" if result.exact and result.exact.passed else "N",
        "Y" if result.fuzzy_pure and result.fuzzy_pure.passed else "N",
        "Y" if result.fuzzy_qnt and result.fuzzy_qnt.passed else "N",
        "Y" if result.sanity and result.sanity.valid else "N",
    )

    max_abs = "N/A"
    qsnr = "N/A"
    cosine = "N/A"

    if result.fuzzy_qnt:
        max_abs = f"{result.fuzzy_qnt.max_abs:.2e}"
        qsnr = (
            f"{result.fuzzy_qnt.qsnr:.1f}"
            if result.fuzzy_qnt.qsnr != float("inf")
            else "inf"
        )
        cosine = f"{result.fuzzy_qnt.cosine:.6f}"

    name = result.name or f"op_{result.op_id}"
    status = result.status.value if result.status else "UNKNOWN"

    return (
        f"{name:<15} {marks[0]:^6} {marks[1]:^8} {marks[2]:^8} {marks[3]:^8} "
        f"{max_abs:>10} {qsnr:>8} {cosine:>8} {status:^14}"
    )


def print_compare_table(results: List[CompareResult]):
    """打印比对结果表格"""
    print()
    print("=" * 110)
    header = (
        f"{'name':<15} {'exact':^6} {'f_pure':^8} {'f_qnt':^8} {'sanity':^8} "
        f"{'max_abs':>10} {'qsnr':>8} {'cosine':>8} {'status':^14}"
    )
    print(header)
    print("-" * 110)

    for r in results:
        print(format_result_row(r))

    print("=" * 110)

    # 汇总统计
    status_counts = {s: 0 for s in CompareStatus}
    for r in results:
        if r.status in status_counts:
            status_counts[r.status] += 1

    summary = (
        f"\nSummary: "
        f"{status_counts[CompareStatus.PASS]} PASS, "
        f"{status_counts[CompareStatus.GOLDEN_SUSPECT]} GOLDEN_SUSPECT, "
        f"{status_counts[CompareStatus.DUT_ISSUE]} DUT_ISSUE, "
        f"{status_counts[CompareStatus.BOTH_SUSPECT]} BOTH_SUSPECT "
        f"(total: {len(results)})"
    )
    print(summary)
    print()


def generate_text_report(
    results: List[CompareResult],
    output_path: Optional[str] = None,
) -> str:
    """
    生成文本报告

    Args:
        results: 比对结果列表
        output_path: 输出路径 (可选)

    Returns:
        报告内容
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Compare Report")
    lines.append("=" * 80)
    lines.append("")

    for r in results:
        lines.append(f"[{r.name or f'op_{r.op_id}'}]")
        lines.append(f"  Status: {r.status.value if r.status else 'UNKNOWN'}")
        lines.append("")

        if r.exact:
            lines.append("  Exact Compare:")
            lines.append(f"    Passed: {r.exact.passed}")
            lines.append(f"    Mismatch Count: {r.exact.mismatch_count}")
            lines.append(f"    Max Abs Error: {r.exact.max_abs:.6e}")
            lines.append("")

        if r.fuzzy_qnt:
            lines.append("  Fuzzy Compare (Quantized):")
            lines.append(f"    Passed: {r.fuzzy_qnt.passed}")
            lines.append(f"    Max Abs Error: {r.fuzzy_qnt.max_abs:.6e}")
            lines.append(f"    QSNR: {r.fuzzy_qnt.qsnr:.2f} dB")
            lines.append(f"    Cosine: {r.fuzzy_qnt.cosine:.6f}")
            lines.append(f"    Exceed Count: {r.fuzzy_qnt.exceed_count}")
            lines.append("")

        if r.sanity:
            lines.append("  Golden Sanity:")
            lines.append(f"    Valid: {r.sanity.valid}")
            for check, passed in r.sanity.checks.items():
                lines.append(f"    {check}: {passed}")
            for msg in r.sanity.messages:
                lines.append(f"    - {msg}")
            lines.append("")

        lines.append("-" * 80)
        lines.append("")

    # 汇总
    status_counts = {s: 0 for s in CompareStatus}
    for r in results:
        if r.status:
            status_counts[r.status] += 1

    lines.append("Summary:")
    for status, count in status_counts.items():
        lines.append(f"  {status.value}: {count}")
    lines.append(f"  Total: {len(results)}")

    content = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")

    return content


def generate_json_report(
    results: List[CompareResult],
    output_path: Optional[str] = None,
) -> dict:
    """
    生成 JSON 报告

    Args:
        results: 比对结果列表
        output_path: 输出路径 (可选)

    Returns:
        报告字典
    """

    def result_to_dict(r: CompareResult) -> dict:
        d = {
            "name": r.name,
            "op_id": r.op_id,
            "status": r.status.value if r.status else None,
            "dut_passed": r.dut_passed,
            "golden_valid": r.golden_valid,
        }

        if r.exact:
            d["exact"] = {
                "passed": r.exact.passed,
                "mismatch_count": r.exact.mismatch_count,
                "first_diff_offset": r.exact.first_diff_offset,
                "max_abs": r.exact.max_abs,
            }

        if r.fuzzy_pure:
            d["fuzzy_pure"] = {
                "passed": r.fuzzy_pure.passed,
                "max_abs": r.fuzzy_pure.max_abs,
                "qsnr": r.fuzzy_pure.qsnr,
                "cosine": r.fuzzy_pure.cosine,
            }

        if r.fuzzy_qnt:
            d["fuzzy_qnt"] = {
                "passed": r.fuzzy_qnt.passed,
                "max_abs": r.fuzzy_qnt.max_abs,
                "qsnr": r.fuzzy_qnt.qsnr,
                "cosine": r.fuzzy_qnt.cosine,
            }

        if r.sanity:
            d["sanity"] = {
                "valid": r.sanity.valid,
                "checks": r.sanity.checks,
                "messages": r.sanity.messages,
            }

        return d

    report = {
        "results": [result_to_dict(r) for r in results],
        "summary": {},
    }

    # 汇总
    status_counts = {s.value: 0 for s in CompareStatus}
    for r in results:
        if r.status:
            status_counts[r.status.value] += 1

    report["summary"] = {
        "total": len(results),
        "by_status": status_counts,
    }

    if output_path:
        Path(output_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report
