#!/usr/bin/env python
"""方式 3: Model DSL 构建 Encoder

使用类 PyTorch 语法的 Model DSL 构建 Encoder，
自动生成权重和 golden，输入 fp16，权重 bfp4 精度。

Encoder 结构:
    Q/K/V proj → Softmax → O proj → LayerNorm
    → FFN_up → GELU → FFN_down → LayerNorm

运行: python demos/datagen/02_model_dsl/run.py
"""
import numpy as np

from aidevtools.datagen import Model
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.formats.quantize import simulate_quantize
from aidevtools.compare.fuzzy import compare_fuzzy
from aidevtools.compare.types import CompareConfig

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256

PRECISION = PrecisionConfig(
    input_dtype="fp16",
    weight_dtype="bfp4",
    compute_dtype="fp32",
    output_dtype="bfp8",
)


def main():
    print("=" * 70)
    print("  方式 3: Model DSL (input:fp16, weight:bfp4)")
    print("=" * 70)

    with Model(seed=42, precision=PRECISION) as m:
        x = m.input((BATCH, SEQ, HIDDEN))

        # Self-Attention
        q = m.linear(x, out_features=HIDDEN)
        k = m.linear(x, out_features=HIDDEN)
        v = m.linear(x, out_features=HIDDEN)
        attn = m.softmax(q)  # 简化: 直接对 Q softmax
        out = m.linear(attn, out_features=HIDDEN)
        ln1 = m.layernorm(out)

        # FFN
        ffn_up = m.linear(ln1, out_features=FFN)
        ffn_act = m.gelu(ffn_up)
        ffn_down = m.linear(ffn_act, out_features=HIDDEN)
        output = m.layernorm(ffn_down)

    # 报告
    print(f"\n  Encoder 结构: {len(m.outputs)} 层")
    print(f"  张量数: {len(m.tensors)}")
    print(f"  最终输出: {m.final_output.shape}")

    print(f"\n  {'张量名':<35} {'Shape':<20} {'QType':<8}")
    print(f"  {'-'*65}")
    for tname, t in m.tensors.items():
        print(f"  {tname:<35} {str(t.shape):<20} {t.qtype:<8}")

    # bfp8 量化比对 (对最终输出)
    config = CompareConfig(fuzzy_min_qsnr=1.0, fuzzy_min_cosine=0.85,
                           fuzzy_max_exceed_ratio=0.15,
                           fuzzy_atol=0.5, fuzzy_rtol=0.5)
    pure = m.final_output.astype(np.float32)
    hw = simulate_quantize(pure, "bfp8")
    r = compare_fuzzy(pure, hw, config)

    print(f"\n  最终输出 bfp8 量化比对:")
    print(f"    QSNR: {r.qsnr:.2f} dB")
    print(f"    Cosine: {r.cosine:.6f}")
    print(f"    MaxAbs: {r.max_abs:.6e}")
    print(f"    Status: {'PASS' if r.passed else 'FAIL'}")

    # 四种比数 (用 generate_four_track)
    print(f"\n  四种比数 (relu 示例):")
    tracks = m.generate_four_track("relu", input_shape=(4, 16), precision=PRECISION)
    for label, g in tracks.all_goldens.items():
        print(f"    {label}: shape={g.shape}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
