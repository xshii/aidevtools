"""
比对报告生成

支持策略模式API（字典格式结果）。
"""

from typing import Dict, List, Any, Optional

from ..strategy.exact import ExactResult
from ..strategy.fuzzy import FuzzyResult
from ..strategy.sanity import SanityResult


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


def print_joint_report(results: Dict[str, Any], name: str = ""):
    """
    联合文字报告 — 汇总表格 + 各策略详细输出

    自动遍历 results 中所有策略结果，调用对应的 print_result / print_heatmap。
    """
    from ..strategy.bit_analysis import BitAnalysisStrategy, BitAnalysisResult
    from ..strategy.blocked import BlockedStrategy, BlockResult

    # 1. 汇总表格
    print_strategy_table([results], names=[name])

    # 2. Exact bit 统计
    exact = results.get("exact")
    if isinstance(exact, ExactResult) and exact.diff_bits > 0:
        print(f"\n[{name}] Bit Statistics")
        print("=" * 60)
        print(f"  Diff elements:     {exact.mismatch_count:,}/{exact.total_elements:,}")
        print(f"  Diff bits:         {exact.diff_bits:,}/{exact.total_bits:,} ({exact.diff_bit_ratio:.4%})")
        print("=" * 60)

    # 3. Bit Analysis 详细 (key = "bit_analysis_<format>")
    for k, v in results.items():
        if k.startswith("bit_analysis") and isinstance(v, BitAnalysisResult):
            BitAnalysisStrategy.print_result(v, name)
            break

    # 4. Block Heatmap 详细 (key = "blocked_<size>")
    for k, v in results.items():
        if k.startswith("blocked") and isinstance(v, list) and v:
            if isinstance(v[0], BlockResult):
                BlockedStrategy.print_heatmap(v)
                break


def visualize_joint_report(results: Dict[str, Any], name: str = "") -> "Page":
    """
    联合可视化报告 — 将所有策略的图表合并到一个 Page

    Returns:
        pyecharts Page，可 render 为 HTML
    """
    from .visualizer import Visualizer
    from ..strategy.exact import ExactStrategy
    from ..strategy.fuzzy import FuzzyStrategy
    from ..strategy.sanity import SanityStrategy
    from ..strategy.bit_analysis import BitAnalysisStrategy, BitAnalysisResult
    from ..strategy.blocked import BlockedStrategy, BlockResult

    page = Visualizer.create_page(title=f"Joint Report: {name}" if name else "Joint Report")

    # 策略 key → (类型检查, visualize 函数) 的有序列表
    _VIS = [
        ("exact",      ExactResult,        ExactStrategy.visualize),
        ("fuzzy_pure", FuzzyResult,        FuzzyStrategy.visualize),
        ("fuzzy_qnt",  FuzzyResult,        FuzzyStrategy.visualize),
        ("sanity",     SanityResult,       SanityStrategy.visualize),
    ]

    for key, result_type, vis_fn in _VIS:
        result = results.get(key)
        if isinstance(result, result_type):
            try:
                sub_page = vis_fn(result)
                for chart in sub_page:
                    page.add(chart)
            except Exception:
                pass

    # Bit Analysis (key 含格式后缀)
    for k, v in results.items():
        if k.startswith("bit_analysis") and isinstance(v, BitAnalysisResult):
            try:
                sub_page = BitAnalysisStrategy.visualize(v)
                for chart in sub_page:
                    page.add(chart)
            except Exception:
                pass
            break

    # Blocked (key 含 block_size 后缀)
    for k, v in results.items():
        if k.startswith("blocked") and isinstance(v, list) and v:
            if isinstance(v[0], BlockResult):
                try:
                    sub_page = BlockedStrategy.visualize(v)
                    for chart in sub_page:
                        page.add(chart)
                except Exception:
                    pass
                break

    return page


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
