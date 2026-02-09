"""
模型级比对工具

提供全局协调的分级分析能力，适用于大模型（几百个算子）场景。

快速开始:
    analyzer = ModelTieredAnalyzer()
    results = analyzer.progressive_analyze(
        per_op_pairs,
        l1_threshold=0.8,  # 80%通过则不深入
    )
    analyzer.print_summary(results)

设计说明:
    ModelTieredAnalyzer: 全局协调的分级分析
       - L1: 所有算子快速检查
       - L2: 失败算子中度分析（可根据L1通过率跳过）
       - L3: 仍失败的深度分析（可根据L2通过率跳过）
"""
from typing import Dict, Tuple, Optional, Any
from collections import Counter
import numpy as np

from .engine import CompareEngine
from .types import CompareConfig
from .strategy import (
    CompositeStrategy,
    ExactStrategy,
    FuzzyStrategy,
    SanityStrategy,
    BlockedStrategy,
    BitXorStrategy,
    BitAnalysisStrategy,
)


# ============================================================================
# 模型级分析
# ============================================================================


# ============================================================================
# 模型级分级分析器
# ============================================================================


class ModelTieredAnalyzer:
    """
    模型级分级分析器 - 支持全局协调

    特点：
    - 三级渐进式分析（L1快速 → L2中度 → L3深度）
    - 全局协调：根据整体通过率决定是否深入下一级
    - 节省计算：大部分算子通过时跳过深度分析

    适用场景：
        - 大模型调试（几十到几百个算子）
        - 需要快速筛选问题算子
        - 资源敏感场景

    示例:
        analyzer = ModelTieredAnalyzer()
        results = analyzer.progressive_analyze(per_op_pairs)
        analyzer.print_summary(results)
    """

    def __init__(self, config: Optional[CompareConfig] = None):
        """
        Args:
            config: 比对配置（应用于所有级别）
        """
        self.config = config or CompareConfig()

    def progressive_analyze(
        self,
        pairs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        l1_threshold: float = 0.9,
        l2_threshold: float = 0.8,
        block_size: int = 64,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        渐进式模型分析（三级）

        工作流程：
            L1: 所有算子快速检查（Exact + Bitwise）
                ↓ 如果通过率 >= l1_threshold，停止
            L2: 失败算子中度分析（Fuzzy + Sanity）
                ↓ 如果L2通过率 >= l2_threshold，停止
            L3: 仍失败的深度分析（BitwisePro + Blocked）

        Args:
            pairs: {op_name: (golden, dut)} 字典
            l1_threshold: L1通过率阈值（默认0.9，即90%通过则停止）
            l2_threshold: L2通过率阈值（默认0.8，即80%通过则停止）
            block_size: 分块大小（用于L3的BlockedStrategy）
            verbose: 是否打印进度信息

        Returns:
            {op_name: result} 字典，每个result包含：
                - 执行的策略结果（exact, bitwise, fuzzy_pure等）
                - "_levels": ["L1", "L2", "L3"] 执行了哪些级别

        示例:
            results = analyzer.progressive_analyze(
                per_op_pairs,
                l1_threshold=0.85,  # 85%通过则停止
            )

            # 查看某算子执行到了哪一级
            print(results["softmax"]["_levels"])  # ['L1', 'L2']
        """
        total_ops = len(pairs)
        results = {}

        # =====================================================================
        # Level 1: 快速检查所有算子（Exact + Bitwise）
        # =====================================================================
        if verbose:
            print(f"\n{'='*60}")
            print(f"  模型级渐进式分析: {total_ops} 个算子")
            print(f"{'='*60}")
            print(f"\n[L1] 快速检查: Exact + Bitwise")

        engine_l1 = CompareEngine(
            CompositeStrategy([ExactStrategy(), BitXorStrategy()]),
            config=self.config,
        )

        for name, (golden, dut) in pairs.items():
            results[name] = engine_l1.run(dut=dut, golden=golden)
            results[name]["_levels"] = ["L1"]

        # 统计L1通过率
        l1_passed = [
            name for name, r in results.items()
            if r.get("exact", {}).get("passed", False)
        ]
        l1_pass_rate = len(l1_passed) / total_ops

        if verbose:
            print(f"  通过: {len(l1_passed)}/{total_ops} ({l1_pass_rate:.1%})")

        # 如果L1通过率达标，直接返回
        if l1_pass_rate >= l1_threshold:
            if verbose:
                print(f"  ✓ 通过率达标 (>= {l1_threshold:.0%})，跳过深度分析")
            return results

        # =====================================================================
        # Level 2: 失败算子中度分析（Fuzzy + Sanity）
        # =====================================================================
        l1_failed = {name: pairs[name] for name in pairs if name not in l1_passed}

        if verbose:
            print(f"\n[L2] 中度分析: Fuzzy + Sanity ({len(l1_failed)} 个算子)")

        engine_l2 = CompareEngine(
            CompositeStrategy([FuzzyStrategy(), SanityStrategy()]),
            config=self.config,
        )

        for name, (golden, dut) in l1_failed.items():
            l2_result = engine_l2.run(dut=dut, golden=golden)
            results[name].update(l2_result)
            results[name]["_levels"].append("L2")

        # 统计L2通过率（在L1失败的算子中）
        l2_passed = [
            name for name in l1_failed
            if (results[name].get("fuzzy_pure", {}).get("passed", False) or
                results[name].get("fuzzy_qnt", {}).get("passed", False))
        ]
        l2_pass_rate = len(l2_passed) / len(l1_failed) if l1_failed else 0

        if verbose:
            print(f"  新增通过: {len(l2_passed)}/{len(l1_failed)} ({l2_pass_rate:.1%})")

        # 如果L2通过率达标，停止
        if l2_pass_rate >= l2_threshold:
            if verbose:
                print(f"  ✓ L2通过率达标 (>= {l2_threshold:.0%})，跳过L3")
            return results

        # =====================================================================
        # Level 3: 仍失败的深度分析（BitwisePro + Blocked）
        # =====================================================================
        l2_failed = {name: pairs[name] for name in l1_failed if name not in l2_passed}

        if not l2_failed:
            return results

        if verbose:
            print(f"\n[L3] 深度分析: BitAnalysis + Blocked ({len(l2_failed)} 个算子)")

        engine_l3 = CompareEngine(
            CompositeStrategy([
                BitAnalysisStrategy(),
                BlockedStrategy(block_size=block_size),
            ]),
            config=self.config,
        )

        for name, (golden, dut) in l2_failed.items():
            l3_result = engine_l3.run(dut=dut, golden=golden)
            results[name].update(l3_result)
            results[name]["_levels"].append("L3")

        if verbose:
            print(f"  完成深度分析")

        return results

    @staticmethod
    def print_summary(results: Dict[str, Dict[str, Any]]):
        """
        打印分级执行统计

        Args:
            results: progressive_analyze 返回的结果

        输出示例:
            分级执行统计:
              L1: 85 ops (85.0%)
              L2: 10 ops (10.0%)
              L3: 5 ops (5.0%)

            最终状态:
              Exact 通过: 85/100
              Fuzzy 通过: 8/15
              剩余失败: 7
        """
        total = len(results)
        level_counts = Counter()

        for r in results.values():
            max_level = r.get("_levels", ["L1"])[-1]
            level_counts[max_level] += 1

        print(f"\n{'='*60}")
        print("  分级执行统计:")
        for level in ["L1", "L2", "L3"]:
            if level in level_counts:
                count = level_counts[level]
                pct = count / total * 100
                print(f"    {level}: {count} ops ({pct:.1f}%)")

        # 最终状态统计
        exact_passed = sum(
            1 for r in results.values()
            if r.get("exact", {}).get("passed", False)
        )

        # Fuzzy通过（在执行了L2的算子中）
        l2_ops = [r for r in results.values() if "L2" in r.get("_levels", [])]
        fuzzy_passed = sum(
            1 for r in l2_ops
            if (r.get("fuzzy_pure", {}).get("passed", False) or
                r.get("fuzzy_qnt", {}).get("passed", False))
        )

        still_failed = total - exact_passed - fuzzy_passed

        print(f"\n  最终状态:")
        print(f"    Exact 通过: {exact_passed}/{total}")
        if l2_ops:
            print(f"    Fuzzy 通过: {fuzzy_passed}/{len(l2_ops)} (L2算子)")
        print(f"    剩余失败: {still_failed}")
        print(f"{'='*60}\n")


__all__ = [
    "ModelTieredAnalyzer",
]
