#!/usr/bin/env python
"""QA vs CQA 精度对比 Demo

对比四种 Golden 模式与模拟 DUT 的精度差异:
  1. Pure:     纯 fp32 链 (基准)
  2. QA:       量化感知随机数 (改数据分布, 非链式)
  3. CQA:      链式量化 (模拟硬件数据流, 标准随机数)
  4. QA + CQA: 两者结合

DUT 模拟: CQA golden + 微小 MAC 舍入噪声 (模拟硬件计算误差)

核心结论:
  - QA 降低「单步」量化误差 (改善数据分布)
  - CQA 消除「链路分叉」误差 (对齐量化路径)
  - 两者解决不同层面的问题, 可以结合使用

运行: python demos/datagen/datagen_06_qa_vs_cqa/run.py
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

# 标准精度配置
PC_STD = PrecisionConfig(
    input_dtype="bfpp8",
    weight_dtype="bfpp4",
    compute_dtype="fp32",
    output_dtype="bfpp8",
)

# QA 精度配置
PC_QA = PrecisionConfig(
    input_dtype="bfpp8",
    weight_dtype="bfpp4",
    compute_dtype="fp32",
    output_dtype="bfpp8",
    qa_aware=True,
    qa_center=1.0,
    qa_amplitude=0.5,
)

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=1.0,
    fuzzy_min_cosine=0.5,
    fuzzy_max_exceed_ratio=0.5,
    fuzzy_atol=1.0,
    fuzzy_rtol=1.0,
)

OP_NAMES = [
    "Q_proj", "K_proj", "V_proj", "Softmax", "O_proj",
    "LayerNorm_1", "FFN_up", "GELU", "FFN_down", "LayerNorm_2",
]


def build_encoder(m: Model):
    """构建 Encoder (10 个算子)"""
    x = m.input((BATCH, SEQ, HIDDEN))
    q = m.linear(x, out_features=HIDDEN)
    k = m.linear(x, out_features=HIDDEN)
    v = m.linear(x, out_features=HIDDEN)
    attn = m.softmax(q)
    out = m.linear(attn, out_features=HIDDEN)
    ln1 = m.layernorm(out)
    up = m.linear(ln1, out_features=FFN)
    act = m.gelu(up)
    down = m.linear(act, out_features=HIDDEN)
    return m.layernorm(down)


def main():
    print("=" * 70)
    print("  QA vs CQA 精度对比")
    print(f"  精度: input=bfpp8, weight=bfpp4, output=bfpp8")
    print(f"  模型: Encoder ({BATCH}x{SEQ}x{HIDDEN}, FFN={FFN})")
    print("=" * 70)

    # =================================================================
    # 构建四种 Golden
    # =================================================================

    # 1. Pure fp32
    with Model(seed=SEED, precision=PC_STD) as m:
        out = build_encoder(m)
    golden_pure = out.golden
    outputs_pure = [o.golden for o in m.outputs]

    # 2. QA (量化感知随机数, 非链式)
    with Model(seed=SEED, precision=PC_QA) as m:
        out = build_encoder(m)
    golden_qa = out.golden
    outputs_qa = [o.golden for o in m.outputs]

    # 3. CQA (链式量化, 标准随机数)
    with Model(seed=SEED, precision=PC_STD, cqa=True) as m:
        out = build_encoder(m)
    golden_cqa = out.golden
    outputs_cqa = [o.golden for o in m.outputs]

    # 4. QA + CQA (结合)
    with Model(seed=SEED, precision=PC_QA, cqa=True) as m:
        out = build_encoder(m)
    golden_qa_cqa = out.golden
    outputs_qa_cqa = [o.golden for o in m.outputs]

    # =================================================================
    # 模拟 DUT: CQA golden + 微小 MAC 舍入噪声
    # =================================================================
    rng = np.random.RandomState(99)
    mac_noise_scale = 0.001
    dut_std = golden_cqa + rng.randn(*golden_cqa.shape).astype(np.float32) * mac_noise_scale

    # QA 版 DUT
    rng2 = np.random.RandomState(99)
    dut_qa = golden_qa_cqa + rng2.randn(*golden_qa_cqa.shape).astype(np.float32) * mac_noise_scale

    # =================================================================
    # 对比 1: 各 Golden vs 标准 DUT
    # =================================================================
    print(f"\n  [对比 1] 各 Golden vs 标准 DUT (CQA + MAC 噪声)")
    print(f"  {'Golden 类型':<25} {'QSNR(dB)':>10} {'Cosine':>10} {'MaxAbs':>10}")
    print(f"  {'-'*57}")

    results = {}
    for name, g in [("Pure (fp32)", golden_pure),
                    ("QA (量化感知)", golden_qa),
                    ("CQA (链式量化)", golden_cqa)]:
        r = FuzzyStrategy.compare(g, dut_std, config=COMPARE_CFG)
        results[name] = r
        print(f"  {name:<25} {r.qsnr:>10.2f} {r.cosine:>10.6f} {r.max_abs:>10.4e}")

    # =================================================================
    # 对比 2: QA+CQA vs QA DUT
    # =================================================================
    print(f"\n  [对比 2] QA+CQA Golden vs QA DUT (QA_CQA + MAC 噪声)")
    print(f"  {'Golden 类型':<25} {'QSNR(dB)':>10} {'Cosine':>10} {'MaxAbs':>10}")
    print(f"  {'-'*57}")

    for name, g in [("QA (非链式)", golden_qa),
                    ("QA + CQA (结合)", golden_qa_cqa)]:
        r = FuzzyStrategy.compare(g, dut_qa, config=COMPARE_CFG)
        results[name + "_qa_dut"] = r
        print(f"  {name:<25} {r.qsnr:>10.2f} {r.cosine:>10.6f} {r.max_abs:>10.4e}")

    # =================================================================
    # 逐层误差传播对比
    # =================================================================
    print(f"\n  [对比 3] 逐层 QSNR — Pure vs CQA vs QA+CQA (对比各自基准)")
    print(f"  {'层':<14} {'Pure→CQA':>12} {'Pure→QA':>12} {'QA→QA_CQA':>12}")
    print(f"  {'-'*52}")

    for i, name in enumerate(OP_NAMES):
        r_cqa = FuzzyStrategy.compare(outputs_pure[i], outputs_cqa[i], config=COMPARE_CFG)
        r_qa = FuzzyStrategy.compare(outputs_pure[i], outputs_qa[i], config=COMPARE_CFG)
        r_qa_cqa = FuzzyStrategy.compare(outputs_qa[i], outputs_qa_cqa[i], config=COMPARE_CFG)
        print(f"  {name:<14} {r_cqa.qsnr:>10.1f}dB {r_qa.qsnr:>10.1f}dB {r_qa_cqa.qsnr:>10.1f}dB")

    # =================================================================
    # 结论
    # =================================================================
    print(f"\n" + "=" * 70)
    print(f"  结论:")
    print(f"  " + "-" * 66)
    print(f"  QA  解决: 数据分布问题 — 让「单步量化误差」更小")
    print(f"       原理: 受控动态范围, block 内值量级相近, 共享指数高效")
    print(f"       局限: 不影响链路传递, 各层独立")
    print(f"  " + "-" * 66)
    print(f"  CQA 解决: 链路分叉问题 — 让 Golden 与 DUT 走相同量化路径")
    print(f"       原理: 每层输出量化后传下层, 与硬件数据流一致")
    print(f"       效果: Golden vs DUT 差异 = MAC 舍入, 不含量化链路分叉")
    print(f"  " + "-" * 66)
    print(f"  QA+CQA:  两者正交, 可结合使用")
    print(f"       QA 降低每步误差 + CQA 对齐链路 = 最接近硬件的 Golden")
    print("=" * 70)

    # =================================================================
    # 结果断言
    # =================================================================

    # CQA vs DUT 应该非常接近 (只有 MAC 噪声)
    r_cqa_dut = results["CQA (链式量化)"]
    assert r_cqa_dut.qsnr > 40, f"CQA vs DUT QSNR 应 > 40 dB, 实际 {r_cqa_dut.qsnr:.1f}"
    assert r_cqa_dut.cosine > 0.999, f"CQA vs DUT Cosine 应 > 0.999, 实际 {r_cqa_dut.cosine:.6f}"

    # Pure vs DUT 差异应远大于 CQA vs DUT
    r_pure_dut = results["Pure (fp32)"]
    assert r_pure_dut.qsnr < r_cqa_dut.qsnr, \
        f"Pure vs DUT ({r_pure_dut.qsnr:.1f}) 应差于 CQA vs DUT ({r_cqa_dut.qsnr:.1f})"

    # QA+CQA vs QA_DUT 也应非常接近
    r_qacqa = results["QA + CQA (结合)_qa_dut"]
    assert r_qacqa.qsnr > 40, f"QA+CQA vs DUT QSNR 应 > 40 dB, 实际 {r_qacqa.qsnr:.1f}"

    print(f"\n  所有断言通过!")
    print("\nDemo 完成!")


if __name__ == "__main__":
    main()
