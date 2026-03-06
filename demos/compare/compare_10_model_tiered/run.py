#!/usr/bin/env python
"""Compare Demo 10: 模型级渐进式分析 (ModelTieredAnalyzer)

演示模型级全局协调分级比数:
  L1: 所有算子快速检查 (Exact)
      -> 通过率 >= 阈值? 停止
  L2: 失败算子中度分析 (Fuzzy + Sanity)
      -> 通过率 >= 阈值? 停止
  L3: 仍失败的深度分析 (BitAnalysis + Blocked)

模型: 小 Transformer (2 layers, 20 个算子)
场景: 大部分算子 bit-exact 或 fuzzy 通过, 少数算子注入较大噪声模拟 DUT 缺陷

运行: python demos/compare/compare_10_model_tiered/run.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np
from aidevtools.compare import CompareEngine, CompareConfig


# ============================================================================
# 配置
# ============================================================================

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
N_LAYERS = 2
SEED = 42

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=25.0,
    fuzzy_min_cosine=0.995,
    fuzzy_max_exceed_ratio=0.01,
    fuzzy_atol=1e-4,
    fuzzy_rtol=1e-3,
)


# ============================================================================
# 数据生成
# ============================================================================


def build_op_pairs(rng: np.random.RandomState):
    """构建 {op_name: (golden, dut)} 字典

    模拟场景:
    - 大部分算子: dut = golden + 微小噪声 (fuzzy 可过)
    - 少数算子:   dut = golden + 较大噪声 (fuzzy 不过, 需深度分析)
    - 1 个算子:   dut = golden (bit-exact)
    """
    pairs = {}

    # 需要深度分析的算子 (注入大噪声)
    bad_ops = {"L1_ffn_gelu", "L0_attn_softmax"}

    for layer_idx in range(N_LAYERS):
        op_defs = [
            (f"L{layer_idx}_Q_proj",       (BATCH, SEQ, HIDDEN)),
            (f"L{layer_idx}_K_proj",       (BATCH, SEQ, HIDDEN)),
            (f"L{layer_idx}_V_proj",       (BATCH, SEQ, HIDDEN)),
            (f"L{layer_idx}_attn_softmax", (BATCH, SEQ, SEQ)),
            (f"L{layer_idx}_O_proj",       (BATCH, SEQ, HIDDEN)),
            (f"L{layer_idx}_layernorm_1",  (BATCH, SEQ, HIDDEN)),
            (f"L{layer_idx}_ffn_up",       (BATCH, SEQ, FFN)),
            (f"L{layer_idx}_ffn_gelu",     (BATCH, SEQ, FFN)),
            (f"L{layer_idx}_ffn_down",     (BATCH, SEQ, HIDDEN)),
            (f"L{layer_idx}_layernorm_2",  (BATCH, SEQ, HIDDEN)),
        ]

        for name, shape in op_defs:
            golden = rng.randn(*shape).astype(np.float32)

            if name == "L0_layernorm_1":
                # bit-exact: dut = golden
                dut = golden.copy()
            elif name in bad_ops:
                # 大噪声: QSNR ~10 dB, fuzzy 不过
                noise_scale = float(np.std(golden)) * 0.3
                dut = golden + rng.randn(*shape).astype(np.float32) * noise_scale
            else:
                # 微小噪声: QSNR ~40 dB, fuzzy 可过
                noise_scale = float(np.std(golden)) * 0.005
                dut = golden + rng.randn(*shape).astype(np.float32) * noise_scale

            pairs[name] = (golden, dut)

    return pairs


# ============================================================================
# 主流程
# ============================================================================


def main():
    print("=" * 70)
    print("  Demo 10: 模型级渐进式分析 (ModelTieredAnalyzer)")
    print(f"  模型: Transformer ({N_LAYERS} layers, 20 ops)")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    pairs = build_op_pairs(rng)
    print(f"\n  算子总数: {len(pairs)}")
    print(f"  预期 bit-exact: 1 (L0_layernorm_1)")
    print(f"  预期 fuzzy 通过: 17")
    print(f"  预期需深度分析: 2 (L0_attn_softmax, L1_ffn_gelu)")

    # =====================================================================
    # 方式 1: ModelTieredAnalyzer — 全局协调分级
    # =====================================================================
    print("\n" + "=" * 70)
    print("  [方式 1] ModelTieredAnalyzer.progressive_analyze()")
    print("=" * 70)

    analyzer = CompareEngine.model_progressive(config=COMPARE_CFG)
    results = analyzer.progressive_analyze(
        pairs,
        l1_threshold=0.9,   # L1 通过率 >= 90% 则跳过 L2
        l2_threshold=0.8,   # L2 通过率 >= 80% 则跳过 L3
        block_size=64,
        verbose=True,
    )
    analyzer.print_summary(results)

    # =====================================================================
    # 逐算子结果展示
    # =====================================================================
    print("\n" + "-" * 70)
    print("  逐算子结果:")
    print("-" * 70)
    print(f"  {'算子':<25} {'级别':<15} {'Exact':^8} {'Fuzzy':^8} {'Sanity':^8}")
    print("  " + "-" * 64)

    for name, r in results.items():
        levels = ", ".join(r.get("_levels", []))
        exact_str = "PASS" if getattr(r.get("exact"), "passed", False) else "FAIL"
        fuzzy = r.get("fuzzy_pure") or r.get("fuzzy_qnt")
        fuzzy_str = "PASS" if fuzzy and fuzzy.passed else ("FAIL" if fuzzy else "-")
        sanity = r.get("sanity")
        sanity_str = "PASS" if sanity and sanity.valid else ("FAIL" if sanity else "-")
        print(f"  {name:<25} {levels:<15} {exact_str:^8} {fuzzy_str:^8} {sanity_str:^8}")

    # =====================================================================
    # 深度分析结果 (L3 算子)
    # =====================================================================
    l3_ops = [name for name, r in results.items() if "L3" in r.get("_levels", [])]
    if l3_ops:
        print(f"\n" + "-" * 70)
        print(f"  L3 深度分析详情 ({len(l3_ops)} 个算子):")
        print("-" * 70)

        for name in l3_ops:
            r = results[name]
            print(f"\n  [{name}]")

            # Fuzzy 指标
            fuzzy = r.get("fuzzy_pure")
            if fuzzy:
                print(f"    QSNR:   {fuzzy.qsnr:.2f} dB")
                print(f"    Cosine: {fuzzy.cosine:.6f}")
                print(f"    MaxAbs: {fuzzy.max_abs:.4e}")

            # Blocked 结果
            for k, v in r.items():
                if k.startswith("blocked") and isinstance(v, list) and v:
                    passed_blocks = sum(1 for b in v if b.passed)
                    print(f"    Blocks: {passed_blocks}/{len(v)} passed")
                    worst = sorted(v, key=lambda b: b.qsnr)[:3]
                    for b in worst:
                        print(f"      Block@{b.offset}: QSNR={b.qsnr:.1f} dB")

            # BitAnalysis 结果
            for k, v in r.items():
                if k.startswith("bit_analysis") and hasattr(v, "summary"):
                    s = v.summary
                    print(f"    BitAnalysis: {s.diff_elements}/{s.total_elements} diff elements")
                    print(f"      Sign flips:     {s.sign_flip_count}")
                    print(f"      Exponent diffs: {s.exponent_diff_count}")
                    print(f"      Mantissa diffs: {s.mantissa_diff_count}")
                    if v.has_critical:
                        print(f"      ** HAS CRITICAL WARNINGS **")

    # =====================================================================
    # 方式 2: 单算子引擎 (对比)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  [方式 2] 单算子 CompareEngine.progressive() 对比")
    print("=" * 70)

    engine = CompareEngine.progressive(config=COMPARE_CFG, deep=True)

    # 挑一个好算子和一个坏算子对比
    for name in ["L0_Q_proj", "L1_ffn_gelu"]:
        golden, dut = pairs[name]
        r = engine.run(dut=dut, golden=golden)
        levels = r.get("_executed_levels", [])
        stopped = r.get("_stopped_at", "?")
        exact = r.get("exact")
        fuzzy = r.get("fuzzy_pure")
        print(f"\n  {name}:")
        print(f"    Levels: {levels}, stopped at: {stopped}")
        if exact:
            print(f"    Exact: {'PASS' if exact.passed else 'FAIL'}, "
                  f"diff_bits: {exact.diff_bits}/{exact.total_bits}")
        if fuzzy:
            print(f"    Fuzzy: {'PASS' if fuzzy.passed else 'FAIL'}, "
                  f"QSNR={fuzzy.qsnr:.1f} dB, cosine={fuzzy.cosine:.6f}")

    # =====================================================================
    # 结果断言
    # =====================================================================
    print("\n" + "=" * 70)
    print("  结果验证:")
    print("=" * 70)

    # 1) 所有算子都有结果
    assert len(results) == 20, f"应有 20 个算子结果, 实际 {len(results)}"

    # 2) L0_layernorm_1 是 bit-exact (dut = golden.copy())
    ln1 = results["L0_layernorm_1"]
    assert ln1.get("exact") is not None, "L0_layernorm_1 应有 exact 结果"
    assert ln1["exact"].passed, "L0_layernorm_1 应该 bit-exact 通过"

    # 3) 坏算子 (大噪声) exact 不通过
    for bad_name in ["L0_attn_softmax", "L1_ffn_gelu"]:
        bad_r = results[bad_name]
        assert bad_r.get("exact") is not None, f"{bad_name} 应有 exact 结果"
        assert not bad_r["exact"].passed, f"{bad_name} 不应该 exact 通过"
        bad_fuzzy = bad_r.get("fuzzy_pure") or bad_r.get("fuzzy_qnt")
        if bad_fuzzy:
            assert not bad_fuzzy.passed, f"{bad_name} 不应该 fuzzy 通过 (QSNR={bad_fuzzy.qsnr:.1f})"

    # 4) 单算子引擎: 好算子 vs 坏算子
    good_r = engine.run(dut=pairs["L0_Q_proj"][1], golden=pairs["L0_Q_proj"][0])
    bad_r = engine.run(dut=pairs["L1_ffn_gelu"][1], golden=pairs["L1_ffn_gelu"][0])
    assert not good_r["exact"].passed, "好算子有微小噪声, exact 不通过"
    good_fuzzy = good_r.get("fuzzy_pure")
    assert good_fuzzy is not None, "好算子应有 fuzzy 结果"
    assert good_fuzzy.qsnr > bad_r.get("fuzzy_pure", good_fuzzy).qsnr, \
        "好算子 QSNR 应高于坏算子"

    # 5) sanity 检查: 所有有 sanity 结果的算子都应 valid
    for name, r in results.items():
        sanity = r.get("sanity")
        if sanity:
            assert sanity.valid, f"{name} sanity 应该 valid"

    print("  所有断言通过!")

    # =====================================================================
    # 总结
    # =====================================================================
    print("\n" + "=" * 70)
    print("  总结:")
    print(f"    ModelTieredAnalyzer: 全局协调, 根据通过率决定分级深度")
    print(f"    L1 (Exact): 快速筛选 bit-exact 算子")
    print(f"    L2 (Fuzzy+Sanity): 失败算子做统计比对")
    print(f"    L3 (BitAnalysis+Blocked): 仍失败的做深度定位")
    print(f"    适用场景: 大模型几十到几百个算子的快速筛查")
    print("=" * 70)
    print("\nDemo 10 完成!")


if __name__ == "__main__":
    main()
