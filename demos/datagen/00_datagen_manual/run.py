#!/usr/bin/env python
"""方式 1: DataGenerator 手动 API 构建 Encoder

手动创建每个张量 (randn / xavier / zeros)，手动调用算子 cpu_golden，
输入 bfp8，权重 bfp4 量化。适合需要精细控制每个参数的场景。

Encoder 结构:
    Q/K/V proj → Softmax → O proj → LayerNorm
    → FFN_up → GELU → FFN_down → LayerNorm

运行: python demos/datagen/00_datagen_manual/run.py
"""
import numpy as np

from aidevtools.datagen import DataGenerator
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.ops.registry import get_op_instance
from aidevtools.formats.quantize import simulate_quantize
from aidevtools.compare.fuzzy import compare_fuzzy
from aidevtools.compare.types import CompareConfig

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
QTYPE_INPUT = "bfp8"
QTYPE_WEIGHT = "bfp4"


def main():
    print("=" * 70)
    print("  方式 1: DataGenerator 手动 API (input:bfp8, weight:bfp4)")
    print("=" * 70)

    gen = DataGenerator(seed=42, qtype=QTYPE_INPUT)
    config = CompareConfig(fuzzy_min_qsnr=1.0, fuzzy_min_cosine=0.85,
                           fuzzy_max_exceed_ratio=0.15,
                           fuzzy_atol=0.5, fuzzy_rtol=0.5)
    results = []

    # 输入
    x = gen.randn((BATCH, SEQ, HIDDEN), name="input", qtype=QTYPE_INPUT)
    hidden = x.array

    # Q/K/V projections — weight: [out_features, in_features] (PyTorch 格式)
    linear_op = get_op_instance("linear")
    for name in ["Q", "K", "V"]:
        w = gen.xavier((HIDDEN, HIDDEN), name=f"w_{name}", qtype=QTYPE_WEIGHT)
        b = gen.uniform((HIDDEN,), low=-0.1, high=0.1, name=f"b_{name}", qtype=QTYPE_INPUT)
        golden_pure = linear_op.cpu_golden(hidden, w.array, b.array)

        # 量化版: input bfp8, weight bfp4
        h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
        w_q = simulate_quantize(w.array.astype(np.float32), QTYPE_WEIGHT)
        golden_hw = linear_op.cpu_golden(h_q, w_q, b.array)

        r = compare_fuzzy(golden_pure, golden_hw, config)
        results.append((f"{name}_proj", golden_pure.shape, r))
        if name == "Q":
            hidden = golden_pure  # 简化: 用 Q 输出继续

    # Softmax
    softmax_op = get_op_instance("softmax")
    golden_pure = softmax_op.cpu_golden(hidden)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    golden_hw = softmax_op.cpu_golden(h_q)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("softmax", golden_pure.shape, r))
    hidden = golden_pure

    # O proj
    w_o = gen.xavier((HIDDEN, HIDDEN), name="w_O", qtype=QTYPE_WEIGHT)
    golden_pure = linear_op.cpu_golden(hidden, w_o.array)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    w_q = simulate_quantize(w_o.array.astype(np.float32), QTYPE_WEIGHT)
    golden_hw = linear_op.cpu_golden(h_q, w_q)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("O_proj", golden_pure.shape, r))
    hidden = golden_pure

    # LayerNorm 1
    ln_op = get_op_instance("layernorm")
    gamma = gen.ones((HIDDEN,), name="gamma1", qtype=QTYPE_INPUT)
    beta = gen.zeros((HIDDEN,), name="beta1", qtype=QTYPE_INPUT)
    golden_pure = ln_op.cpu_golden(hidden, (HIDDEN,), weight=gamma.array, bias=beta.array)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    golden_hw = ln_op.cpu_golden(h_q, (HIDDEN,), weight=gamma.array, bias=beta.array)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("layernorm_1", golden_pure.shape, r))
    hidden = golden_pure

    # FFN up — weight: [FFN, HIDDEN]
    w_up = gen.xavier((FFN, HIDDEN), name="w_ffn_up", qtype=QTYPE_WEIGHT)
    golden_pure = linear_op.cpu_golden(hidden, w_up.array)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    w_q = simulate_quantize(w_up.array.astype(np.float32), QTYPE_WEIGHT)
    golden_hw = linear_op.cpu_golden(h_q, w_q)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("ffn_up", golden_pure.shape, r))
    hidden = golden_pure

    # GELU
    gelu_op = get_op_instance("gelu")
    golden_pure = gelu_op.cpu_golden(hidden)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    golden_hw = gelu_op.cpu_golden(h_q)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("gelu", golden_pure.shape, r))
    hidden = golden_pure

    # FFN down — weight: [HIDDEN, FFN]
    w_down = gen.xavier((HIDDEN, FFN), name="w_ffn_down", qtype=QTYPE_WEIGHT)
    golden_pure = linear_op.cpu_golden(hidden, w_down.array)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    w_q = simulate_quantize(w_down.array.astype(np.float32), QTYPE_WEIGHT)
    golden_hw = linear_op.cpu_golden(h_q, w_q)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("ffn_down", golden_pure.shape, r))
    hidden = golden_pure

    # LayerNorm 2
    gamma2 = gen.ones((HIDDEN,), name="gamma2", qtype=QTYPE_INPUT)
    beta2 = gen.zeros((HIDDEN,), name="beta2", qtype=QTYPE_INPUT)
    golden_pure = ln_op.cpu_golden(hidden, (HIDDEN,), weight=gamma2.array, bias=beta2.array)
    h_q = simulate_quantize(hidden.astype(np.float32), QTYPE_INPUT)
    golden_hw = ln_op.cpu_golden(h_q, (HIDDEN,), weight=gamma2.array, bias=beta2.array)
    r = compare_fuzzy(golden_pure, golden_hw, config)
    results.append(("layernorm_2", golden_pure.shape, r))

    # 报告
    print(f"\n  Encoder 比数 (Pure fp32 vs HW input:{QTYPE_INPUT} weight:{QTYPE_WEIGHT})")
    print(f"  batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, ffn={FFN}")
    print(f"\n  {'算子':<15} {'Shape':<20} {'QSNR':>10} {'Cosine':>10} {'Status':>8}")
    print(f"  {'-'*65}")
    for name, shape, r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {name:<15} {str(shape):<20} {r.qsnr:>10.2f} {r.cosine:>10.6f} {status:>8}")

    print(f"\n  L2 总量: {gen.total_size / 1024:.1f} KB, 张量数: {len(gen._tensors)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
