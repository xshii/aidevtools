"""
比对报告生成 (重构版)

支持新的策略模式API（字典格式结果）。
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from .types import CompareStatus, ExactResult, FuzzyResult, SanityResult


# ============================================================================
# 新API - 支持策略模式的字典结果
# ============================================================================


def format_strategy_results(
    results: Dict[str, Any],
    name: str = "",
) -> str:
    """
    格式化策略结果为表格行

    Args:
        results: 策略结果字典 {strategy_name: result}
        name: 算子/比对名称

    Returns:
        格式化的表格行字符串
    """
    # 提取各个策略的结果
    exact = results.get("exact")
    fuzzy_pure = results.get("fuzzy_pure")
    fuzzy_qnt = results.get("fuzzy_qnt")
    sanity = results.get("sanity")

    # 判断通过状态
    marks = (
        "Y" if exact and isinstance(exact, ExactResult) and exact.passed else "N",
        "Y" if fuzzy_pure and isinstance(fuzzy_pure, FuzzyResult) and fuzzy_pure.passed else "N",
        "Y" if fuzzy_qnt and isinstance(fuzzy_qnt, FuzzyResult) and fuzzy_qnt.passed else "N",
        "Y" if sanity and isinstance(sanity, SanityResult) and sanity.valid else "N",
    )

    # 提取指标（优先使用fuzzy_qnt，其次fuzzy_pure）
    max_abs = "N/A"
    qsnr = "N/A"
    cosine = "N/A"

    fuzzy = fuzzy_qnt if fuzzy_qnt else fuzzy_pure
    if fuzzy and isinstance(fuzzy, FuzzyResult):
        max_abs = f"{fuzzy.max_abs:.2e}"
        qsnr = f"{fuzzy.qsnr:.1f}" if fuzzy.qsnr != float("inf") else "inf"
        cosine = f"{fuzzy.cosine:.6f}"

    # 判定最终状态
    dut_pass = (fuzzy_qnt and fuzzy_qnt.passed) if fuzzy_qnt else (fuzzy_pure and fuzzy_pure.passed if fuzzy_pure else False)
    sanity_pass = sanity.valid if sanity else True

    if dut_pass and sanity_pass:
        status = "PASS"
    elif dut_pass and not sanity_pass:
        status = "GOLDEN_SUSPECT"
    elif not dut_pass and sanity_pass:
        status = "DUT_ISSUE"
    else:
        status = "BOTH_SUSPECT"

    display_name = name or "unnamed"
    return (
        f"{display_name:<15} {marks[0]:^6} {marks[1]:^8} {marks[2]:^8} {marks[3]:^8} "
        f"{max_abs:>10} {qsnr:>8} {cosine:>8} {status:^14}"
    )


def print_strategy_table(results_list: List[Dict[str, Any]], names: Optional[List[str]] = None):
    """
    打印策略结果表格

    Args:
        results_list: 结果列表，每个元素是策略结果字典
        names: 名称列表（与results_list对应）
    """
    print()
    print("=" * 110)
    header = (
        f"{'name':<15} {'exact':^6} {'f_pure':^8} {'f_qnt':^8} {'sanity':^8} "
        f"{'max_abs':>10} {'qsnr':>8} {'cosine':>8} {'status':^14}"
    )
    print(header)
    print("-" * 110)

    names = names or [""] * len(results_list)
    for name, results in zip(names, results_list):
        print(format_strategy_results(results, name))

    print("=" * 110)

    # 汇总统计
    status_counts = {
        "PASS": 0,
        "GOLDEN_SUSPECT": 0,
        "DUT_ISSUE": 0,
        "BOTH_SUSPECT": 0,
    }

    for name, results in zip(names, results_list):
        # 重新计算状态
        fuzzy_qnt = results.get("fuzzy_qnt")
        fuzzy_pure = results.get("fuzzy_pure")
        sanity = results.get("sanity")

        dut_pass = (fuzzy_qnt and fuzzy_qnt.passed) if fuzzy_qnt else (fuzzy_pure and fuzzy_pure.passed if fuzzy_pure else False)
        sanity_pass = sanity.valid if sanity else True

        if dut_pass and sanity_pass:
            status_counts["PASS"] += 1
        elif dut_pass and not sanity_pass:
            status_counts["GOLDEN_SUSPECT"] += 1
        elif not dut_pass and sanity_pass:
            status_counts["DUT_ISSUE"] += 1
        else:
            status_counts["BOTH_SUSPECT"] += 1

    summary = (
        f"\nSummary: "
        f"{status_counts['PASS']} PASS, "
        f"{status_counts['GOLDEN_SUSPECT']} GOLDEN_SUSPECT, "
        f"{status_counts['DUT_ISSUE']} DUT_ISSUE, "
        f"{status_counts['BOTH_SUSPECT']} BOTH_SUSPECT "
        f"(total: {len(results_list)})"
    )
    print(summary)
    print()


def generate_strategy_json(
    results: Dict[str, Any],
    name: str = "",
) -> Dict[str, Any]:
    """
    将策略结果转换为JSON格式

    Args:
        results: 策略结果字典
        name: 名称

    Returns:
        JSON序列化友好的字典
    """
    output = {"name": name, "strategies": {}}

    for strategy_name, result in results.items():
        if hasattr(result, "__dict__"):
            # 如果是dataclass，转换为字典
            output["strategies"][strategy_name] = {
                k: (v.value if hasattr(v, "value") else v)
                for k, v in result.__dict__.items()
                if not k.startswith("_")
            }
        elif isinstance(result, list):
            # 如果是列表（如blocked）
            output["strategies"][strategy_name] = [
                {k: v for k, v in item.__dict__.items() if not k.startswith("_")}
                if hasattr(item, "__dict__") else item
                for item in result
            ]
        else:
            output["strategies"][strategy_name] = str(result)

    return output


# ============================================================================
# 旧API - 向后兼容（已废弃，仅供迁移）
# ============================================================================


def format_result_row(result, *args, **kwargs) -> str:
    """
    格式化单行结果（旧API，已废弃）

    注意：此函数仅为向后兼容保留，新代码请使用 format_strategy_results
    """
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


def print_compare_table(results):
    """
    打印比对结果表格（旧API，已废弃）

    注意：此函数仅为向后兼容保留，新代码请使用 print_strategy_table
    """
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


def generate_text_report(results, output_path=None) -> str:
    """生成文本报告（旧API，已废弃）"""
    lines = []
    lines.append("=" * 80)
    lines.append("Compare Report")
    lines.append("=" * 80)
    lines.append("")

    for r in results:
        lines.append(f"Operation: {r.name or f'op_{r.op_id}'}")
        lines.append(f"  Status: {r.status.value if r.status else 'UNKNOWN'}")
        if r.exact:
            lines.append(f"  Exact Compare: {'PASS' if r.exact.passed else 'FAIL'}")
            lines.append(f"    Mismatch: {r.exact.mismatch_count}")
        if r.fuzzy_pure:
            lines.append(
                f"  Fuzzy Compare (pure): QSNR={r.fuzzy_pure.qsnr:.2f}, "
                f"Cosine={r.fuzzy_pure.cosine:.6f}"
            )
        if r.fuzzy_qnt:
            lines.append(
                f"  Fuzzy Compare (qnt): QSNR={r.fuzzy_qnt.qsnr:.2f}, "
                f"Cosine={r.fuzzy_qnt.cosine:.6f}"
            )
        if r.sanity:
            lines.append(f"  Golden Sanity: {'VALID' if r.sanity.valid else 'INVALID'}")
        lines.append("")

    # Summary
    status_counts = {s: 0 for s in CompareStatus}
    for r in results:
        if r.status in status_counts:
            status_counts[r.status] += 1
    lines.append("Summary:")
    lines.append(f"  Total: {len(results)}")
    for s, count in status_counts.items():
        if count > 0:
            lines.append(f"  {s.value}: {count}")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)

    return report


def generate_json_report(results, output_path=None) -> dict:
    """生成JSON报告（旧API，已废弃）"""
    items = []
    for r in results:
        item = {
            "name": r.name or f"op_{r.op_id}",
            "status": r.status.value if r.status else "UNKNOWN",
        }
        if r.exact:
            item["exact"] = {"passed": r.exact.passed}
        if r.fuzzy_pure:
            item["fuzzy_pure"] = {
                "passed": r.fuzzy_pure.passed,
                "qsnr": r.fuzzy_pure.qsnr,
                "cosine": r.fuzzy_pure.cosine,
            }
        if r.fuzzy_qnt:
            item["fuzzy_qnt"] = {
                "passed": r.fuzzy_qnt.passed,
                "qsnr": r.fuzzy_qnt.qsnr,
                "cosine": r.fuzzy_qnt.cosine,
            }
        if r.sanity:
            item["sanity"] = {"valid": r.sanity.valid}
        items.append(item)

    # Summary
    status_counts = {}
    for r in results:
        status_val = r.status.value if r.status else "UNKNOWN"
        status_counts[status_val] = status_counts.get(status_val, 0) + 1

    output = {
        "results": items,
        "summary": {
            "total": len(results),
            "by_status": status_counts,
        },
    }

    if output_path:
        Path(output_path).write_text(json.dumps(output, indent=2))

    return output
