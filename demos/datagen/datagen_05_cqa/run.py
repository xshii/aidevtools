#!/usr/bin/env python
"""CQA (Chain Quantization Aware) 链式量化感知 Demo

对比三种 Golden 计算模式在长链路下的精度差异:
  1. Pure:  纯 fp32 链 (基准)
  2. HW:   单步量化 golden — 各层独立 simulate_quantize，不链式传递
  3. CQA:  链式量化 golden — 每层输出量化后传给下层，模拟硬件真实数据流

Encoder 结构 (10 个算子):
    Q/K/V proj -> Softmax -> O proj -> LayerNorm
    -> FFN_up -> GELU -> FFN_down -> LayerNorm

运行: python demos/datagen/datagen_05_cqa/run.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np

from aidevtools.datagen import Model
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.compare import FuzzyStrategy, CompareConfig

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
SEED = 42

PRECISION = PrecisionConfig(
    input_dtype="bfpp8",
    weight_dtype="bfpp4",
    compute_dtype="fp32",
    output_dtype="bfpp8",
)

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=1.0,
    fuzzy_min_cosine=0.85,
    fuzzy_max_exceed_ratio=0.15,
    fuzzy_atol=0.5,
    fuzzy_rtol=0.5,
)


def build_encoder(m: Model):
    """构建 Encoder，返回各层输出"""
    x = m.input((BATCH, SEQ, HIDDEN))

    # Self-Attention
    q = m.linear(x, out_features=HIDDEN)
    k = m.linear(x, out_features=HIDDEN)
    v = m.linear(x, out_features=HIDDEN)
    attn = m.softmax(q)
    out = m.linear(attn, out_features=HIDDEN)
    ln1 = m.layernorm(out)

    # FFN
    up = m.linear(ln1, out_features=FFN)
    act = m.gelu(up)
    down = m.linear(act, out_features=HIDDEN)
    output = m.layernorm(down)

    return output


def main():
    print("=" * 70)
    print("  CQA (Chain Quantization Aware) Demo")
    print(f"  精度: input={PRECISION.input_dtype}, weight={PRECISION.weight_dtype}, "
          f"output={PRECISION.output_dtype}")
    print(f"  模型: Encoder ({BATCH}x{SEQ}x{HIDDEN}, FFN={FFN})")
    print("=" * 70)

    # =====================================================================
    # 模式 1: Pure fp32 (基准)
    # =====================================================================
    with Model(seed=SEED, precision=PRECISION, cqa=False) as m_pure:
        out_pure = build_encoder(m_pure)
    golden_pure = out_pure.golden
    outputs_pure = [o.golden for o in m_pure.outputs]

    # =====================================================================
    # 模式 2: CQA 链式量化
    # =====================================================================
    with Model(seed=SEED, precision=PRECISION, cqa=True) as m_cqa:
        out_cqa = build_encoder(m_cqa)
    golden_cqa = out_cqa.golden
    outputs_cqa = [o.golden for o in m_cqa.outputs]

    # =====================================================================
    # 逐层对比: Pure vs CQA
    # =====================================================================
    op_names = [
        "Q_proj", "K_proj", "V_proj", "Softmax", "O_proj",
        "LayerNorm_1", "FFN_up", "GELU", "FFN_down", "LayerNorm_2",
    ]

    print(f"\n  逐层精度对比 (Pure fp32 vs CQA):")
    print(f"  {'层':<15} {'QSNR(dB)':>10} {'Cosine':>10} {'MaxAbs':>12}")
    print(f"  {'-'*50}")

    for i, name in enumerate(op_names):
        r = FuzzyStrategy.compare(outputs_pure[i], outputs_cqa[i], config=COMPARE_CFG)
        print(f"  {name:<15} {r.qsnr:>10.2f} {r.cosine:>10.6f} {r.max_abs:>12.4e}")

    # =====================================================================
    # 最终输出对比
    # =====================================================================
    r_final = FuzzyStrategy.compare(golden_pure, golden_cqa, config=COMPARE_CFG)

    print(f"\n  最终输出 (Pure vs CQA):")
    print(f"    QSNR:   {r_final.qsnr:.2f} dB")
    print(f"    Cosine: {r_final.cosine:.6f}")
    print(f"    MaxAbs: {r_final.max_abs:.4e}")

    # =====================================================================
    # 误差传播分析: 逐层 QSNR 变化趋势
    # =====================================================================
    print(f"\n  误差传播趋势 (逐层 QSNR):")
    qsnrs = []
    for i, name in enumerate(op_names):
        r = FuzzyStrategy.compare(outputs_pure[i], outputs_cqa[i], config=COMPARE_CFG)
        qsnrs.append(r.qsnr)

    # 简单 ASCII 趋势图
    max_q = max(q for q in qsnrs if q != float('inf'))
    min_q = min(q for q in qsnrs if q != float('-inf'))
    width = 40
    for i, (name, q) in enumerate(zip(op_names, qsnrs)):
        if q == float('inf'):
            bar = "#" * width
        else:
            bar_len = max(1, int((q - min_q) / (max_q - min_q + 1e-9) * width))
            bar = "#" * bar_len
        print(f"  L{i:02d} {name:<14} {q:>7.1f} dB |{bar}")

    # =====================================================================
    # CQA 的价值: 对比 DUT 时更接近
    # =====================================================================
    print(f"\n  CQA 的价值:")
    print(f"    Pure golden 与硬件 DUT 的差异包含:")
    print(f"      1. 每层权重量化误差")
    print(f"      2. 每层激活值量化误差")
    print(f"      3. 误差逐层累积放大")
    print(f"    CQA golden 已内含量化链路误差,")
    print(f"    与 DUT 对比时只剩计算精度差异 (MAC 舍入等)")

    # =====================================================================
    # 结果断言
    # =====================================================================
    assert len(outputs_pure) == 10, f"应有 10 层输出, 实际 {len(outputs_pure)}"
    assert len(outputs_cqa) == 10, f"应有 10 层输出, 实际 {len(outputs_cqa)}"
    assert golden_pure.shape == golden_cqa.shape, "Pure 和 CQA 输出 shape 应一致"

    # CQA 与 Pure 应有差异 (量化引入了误差)
    assert not np.allclose(golden_pure, golden_cqa, atol=1e-6), \
        "CQA 应与 Pure 有差异 (量化链路)"

    # CQA 与 Pure 有显著差异 (bfpp4 权重 + 10 层累积, cosine 可低至 0.7)
    # 但不应完全无关 (cosine > 0.5)
    assert r_final.cosine > 0.5, \
        f"CQA 与 Pure 的 Cosine 应 > 0.5, 实际 {r_final.cosine:.4f}"

    # 逐层: 越后面的层 QSNR 应越低 (误差累积)
    # 取首尾对比 (跳过 inf)
    finite_qsnrs = [(i, q) for i, q in enumerate(qsnrs) if q != float('inf')]
    if len(finite_qsnrs) >= 2:
        first_q = finite_qsnrs[0][1]
        last_q = finite_qsnrs[-1][1]
        assert first_q > last_q - 5, \
            f"后层 QSNR 不应显著高于前层 (首层 {first_q:.1f}, 末层 {last_q:.1f})"

    print(f"\n  所有断言通过!")
    print("=" * 70)
    print("\nDemo 完成!")


if __name__ == "__main__":
    main()
