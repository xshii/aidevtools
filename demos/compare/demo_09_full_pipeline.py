#!/usr/bin/env python3
"""
Demo 09: 完整比对流水线 - Encoder 模型全量策略检查

展示从基础到深度的渐进式比对策略：
1. Exact - 精确匹配 + bit 统计（快速筛选）
2. Fuzzy - 统计指标（量化感知）
3. Blocked - 块级分析（定位异常区域）
4. BitAnalysis - bit 级深度分析
5. Sanity - Golden 自检

完整可视化：
- 各策略级可视化
- 模型级误差传播分析
"""

import numpy as np
import os


def simulate_encoder_layer():
    """模拟 BERT Encoder Layer 的算子输出"""
    np.random.seed(42)

    # 模拟各算子的 golden 和 dut 输出
    ops = {
        # Multi-Head Attention
        "qkv_proj": {
            "golden": np.random.randn(8, 512, 768).astype(np.float32),
            "scale": 0.01,  # 添加小误差
        },
        "attn_scores": {
            "golden": np.random.randn(8, 12, 512, 512).astype(np.float32),
            "scale": 0.005,
        },
        "attn_output": {
            "golden": np.random.randn(8, 512, 768).astype(np.float32),
            "scale": 0.02,
        },

        # Feed-Forward Network
        "ffn_fc1": {
            "golden": np.random.randn(8, 512, 3072).astype(np.float32),
            "scale": 0.03,  # 误差增大
        },
        "ffn_gelu": {
            "golden": np.random.randn(8, 512, 3072).astype(np.float32),
            "scale": 0.05,  # 误差继续增大（激活函数）
        },
        "ffn_fc2": {
            "golden": np.random.randn(8, 512, 768).astype(np.float32),
            "scale": 0.08,  # 误差累积
        },

        # Layer Norm
        "layernorm1": {
            "golden": np.random.randn(8, 512, 768).astype(np.float32),
            "scale": 0.001,  # LayerNorm 误差小
        },
        "layernorm2": {
            "golden": np.random.randn(8, 512, 768).astype(np.float32),
            "scale": 0.002,
        },
    }

    # 生成 DUT 输出（添加误差）
    for op_name, data in ops.items():
        dut = data["golden"] + np.random.randn(*data["golden"].shape).astype(np.float32) * data["scale"]
        data["dut"] = dut

    return ops


def run_progressive_strategies(ops_data):
    """
    渐进式策略检查

    从快到慢，从粗到细：
    1. Exact - 快速精确匹配 + bit 统计
    2. Fuzzy - 统计指标
    3. Blocked - 块级分析
    4. BitAnalysis - bit 级深度分析
    5. Sanity - Golden 自检
    """
    from aidevtools.compare.strategy import (
        ExactStrategy,
        FuzzyStrategy,
        BlockedStrategy,
        BitAnalysisStrategy,
        SanityStrategy,
        FP32,
    )
    from aidevtools.compare.types import CompareConfig

    print("=" * 80)
    print("渐进式策略检查流水线")
    print("=" * 80)

    # 选择一个算子进行演示（ffn_fc2 - 误差最大）
    op_name = "ffn_fc2"
    golden = ops_data[op_name]["golden"]
    dut = ops_data[op_name]["dut"]

    print(f"\n算子: {op_name}")
    print(f"Shape: {golden.shape}")
    print(f"Total elements: {golden.size}")

    results = {}

    # 1. Exact - 快速筛选
    print("\n" + "-" * 80)
    print("1️⃣  Exact Strategy - 精确匹配（快速筛选）")
    print("-" * 80)

    exact_result = ExactStrategy.compare(golden, dut, max_abs=1e-6, max_count=10)
    results["exact"] = exact_result

    print(f"Passed: {exact_result.passed}")
    print(f"Mismatch: {exact_result.mismatch_count}/{exact_result.total_elements}")
    print(f"Max Abs: {exact_result.max_abs:.6e}")

    # 2. Fuzzy - 统计指标
    print("\n" + "-" * 80)
    print("2️⃣  Fuzzy Strategy - 统计指标（量化感知）")
    print("-" * 80)

    config = CompareConfig(
        fuzzy_min_qsnr=30.0,
        fuzzy_min_cosine=0.999,
        fuzzy_atol=1e-5,
        fuzzy_rtol=1e-3,
    )

    fuzzy_result = FuzzyStrategy.compare(golden, dut, config=config)
    results["fuzzy"] = fuzzy_result

    print(f"Passed: {fuzzy_result.passed}")
    print(f"QSNR: {fuzzy_result.qsnr:.2f} dB")
    print(f"Cosine: {fuzzy_result.cosine:.6f}")
    print(f"Max Abs: {fuzzy_result.max_abs:.6e}")

    # 3. Blocked - 块级分析
    print("\n" + "-" * 80)
    print("3️⃣  Blocked Strategy - 块级分析（定位异常区域）")
    print("-" * 80)

    blocked_result = BlockedStrategy.compare(golden, dut, block_size=4096, min_qsnr=25.0)
    results["blocked"] = blocked_result

    passed_blocks = sum(1 for b in blocked_result if b.passed)
    failed_blocks = len(blocked_result) - passed_blocks

    print(f"Total blocks: {len(blocked_result)}")
    print(f"Passed: {passed_blocks}, Failed: {failed_blocks}")

    if failed_blocks > 0:
        worst = sorted(blocked_result, key=lambda b: b.qsnr)[:3]
        print(f"Worst 3 blocks:")
        for b in worst:
            print(f"  - Block@{b.offset}: QSNR={b.qsnr:.2f} dB, Cosine={b.cosine:.4f}")

    # 4. BitAnalysis - 深度 bit 级分析
    print("\n" + "-" * 80)
    print("4️⃣  BitAnalysis Strategy - bit 级深度分析")
    print("-" * 80)

    # 对前 1000 个元素进行 bit 分析（演示）
    sample_size = min(1000, golden.size)
    golden_sample = golden.flatten()[:sample_size]
    dut_sample = dut.flatten()[:sample_size]

    bitwise_result = BitAnalysisStrategy.compare(golden_sample, dut_sample, fmt=FP32)
    results["bitwise"] = bitwise_result

    print(f"Total: {bitwise_result.summary.total_elements}")
    print(f"Diff: {bitwise_result.summary.diff_elements}")
    print(f"Sign Flip: {bitwise_result.summary.sign_flip_count}")
    print(f"Exponent Diff: {bitwise_result.summary.exponent_diff_count}")
    print(f"Mantissa Diff: {bitwise_result.summary.mantissa_diff_count}")
    print(f"Has Critical: {bitwise_result.has_critical}")

    # 5. Sanity - Golden 自检
    print("\n" + "-" * 80)
    print("5️⃣  Sanity Strategy - Golden 自检")
    print("-" * 80)

    sanity_result = SanityStrategy.check_data(golden_sample, name="golden")
    results["sanity"] = sanity_result

    print(f"Valid: {sanity_result.valid}")
    if sanity_result.checks:
        for check_name, passed in sanity_result.checks.items():
            print(f"  {check_name}: {'✅' if passed else '❌'}")

    return results


def generate_all_visualizations(results, ops_data):
    """生成所有可视化报告"""

    print("\n" + "=" * 80)
    print("生成可视化报告")
    print("=" * 80)

    try:
        import pyecharts
    except ImportError:
        print("❌ pyecharts not installed. Skip visualization.")
        print("   Install: pip install pyecharts")
        return

    output_dir = "/tmp/compare_demo09"
    os.makedirs(output_dir, exist_ok=True)

    from aidevtools.compare.strategy import (
        ExactStrategy,
        FuzzyStrategy,
        BlockedStrategy,
        BitAnalysisStrategy,
        SanityStrategy,
    )
    from aidevtools.compare.types import CompareConfig

    # 1. Exact 可视化
    if "exact" in results:
        print("\n📊 Generating Exact visualization...")
        page = ExactStrategy.visualize(results["exact"])
        path = f"{output_dir}/01_exact_report.html"
        page.render(path)
        print(f"   ✅ {path}")

    # 2. Fuzzy 可视化
    if "fuzzy" in results:
        print("\n📊 Generating Fuzzy visualization...")
        page = FuzzyStrategy.visualize(results["fuzzy"], config=CompareConfig())
        path = f"{output_dir}/02_fuzzy_report.html"
        page.render(path)
        print(f"   ✅ {path}")

    # 3. Blocked 可视化
    if "blocked" in results:
        print("\n📊 Generating Blocked visualization...")
        page = BlockedStrategy.visualize(results["blocked"], threshold=25.0, cols=8)
        path = f"{output_dir}/03_blocked_report.html"
        page.render(path)
        print(f"   ✅ {path}")

    # 4. BitAnalysis 可视化
    if "bitwise" in results:
        print("\n📊 Generating BitAnalysis visualization...")
        page = BitAnalysisStrategy.visualize(results["bitwise"])
        path = f"{output_dir}/04_bitwise_report.html"
        page.render(path)
        print(f"   ✅ {path}")

    # 5. Sanity 可视化
    if "sanity" in results:
        print("\n📊 Generating Sanity visualization...")
        page = SanityStrategy.visualize(results["sanity"])
        path = f"{output_dir}/06_sanity_report.html"
        page.render(path)
        print(f"   ✅ {path}")

    # 7. 模型级可视化（误差传播分析）
    print("\n📊 Generating Model-level visualization...")
    generate_model_visualization(ops_data, output_dir)


def generate_model_visualization(ops_data, output_dir):
    """生成模型级误差传播可视化"""
    from aidevtools.compare.report.model_visualizer import (
        ModelVisualizer,
        ModelCompareResult,
        OpCompareResult,
        OpStatus,
    )
    from aidevtools.compare.strategy import FuzzyStrategy
    from aidevtools.compare.types import CompareConfig

    config = CompareConfig()

    # 计算每个算子的指标
    ops = []
    for i, (op_name, data) in enumerate(ops_data.items()):
        golden = data["golden"]
        dut = data["dut"]

        # 计算 Fuzzy 指标
        fuzzy_result = FuzzyStrategy.compare(golden, dut, config=config)

        ops.append(OpCompareResult(
            op_name=op_name,
            op_id=i,
            status=OpStatus.HAS_DATA,
            qsnr=fuzzy_result.qsnr,
            cosine=fuzzy_result.cosine,
            max_abs=fuzzy_result.max_abs,
            passed=fuzzy_result.passed,
        ))

    # 构建模型结果
    model_result = ModelCompareResult(
        model_name="BERT-Encoder-Layer",
        ops=ops,
        total_ops=len(ops),
        ops_with_data=len(ops),
        ops_missing_dut=0,
        passed_ops=sum(1 for op in ops if op.passed),
        failed_ops=sum(1 for op in ops if not op.passed),
    )

    # 生成可视化
    page = ModelVisualizer.visualize(model_result)
    path = f"{output_dir}/07_model_report.html"
    page.render(path)
    print(f"   ✅ {path}")


def main():
    """运行完整流水线"""
    print("\n" + "=" * 80)
    print("Demo 09: Encoder 模型全量策略检查")
    print("=" * 80)

    # 1. 模拟 Encoder 层
    print("\n🔧 模拟 BERT Encoder Layer...")
    ops_data = simulate_encoder_layer()
    print(f"   生成 {len(ops_data)} 个算子的输出")

    # 2. 运行渐进式策略
    results = run_progressive_strategies(ops_data)

    # 3. 生成所有可视化
    generate_all_visualizations(results, ops_data)

    print("\n" + "=" * 80)
    print("✅ 全部完成！")
    print("=" * 80)
    print("\n报告位置: /tmp/compare_demo09/")
    print("  01_exact_report.html      - 精确匹配 + bit 统计")
    print("  02_fuzzy_report.html      - 统计指标")
    print("  03_blocked_report.html    - 块级分析")
    print("  04_bitwise_report.html    - bit 级分析")
    print("  05_sanity_report.html     - Golden 自检")
    print("  06_model_report.html      - 模型级误差传播")
    print()


if __name__ == "__main__":
    main()
