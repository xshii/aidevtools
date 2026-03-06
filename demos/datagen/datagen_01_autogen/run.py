#!/usr/bin/env python
"""方式 2: 算子自动生成 API 构建 Encoder

使用 gen.generate_four_track() 根据 @register_op 元信息自动生成数据，
无需手动创建权重。输入 fp16，权重 bfpp4 精度。

Encoder 结构:
    Q/K/V proj → Softmax → O proj → LayerNorm
    → FFN_up → GELU → FFN_down → LayerNorm

运行: python demos/datagen/01_datagen_autogen/run.py
"""
import numpy as np

from aidevtools.datagen import DataGenerator
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.compare import FuzzyStrategy
from aidevtools.compare import CompareConfig

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256

PRECISION = PrecisionConfig(
    input_dtype="fp16",
    weight_dtype="bfpp4",
    compute_dtype="fp32",
    output_dtype="bfpp8",
)

ENCODER_OPS = [
    ("Q_proj",      "linear",    (BATCH, SEQ, HIDDEN),     {"out_features": HIDDEN}),
    ("K_proj",      "linear",    (BATCH, SEQ, HIDDEN),     {"out_features": HIDDEN}),
    ("V_proj",      "linear",    (BATCH, SEQ, HIDDEN),     {"out_features": HIDDEN}),
    ("attn_softmax","softmax",   (BATCH, SEQ, HIDDEN),     {}),
    ("O_proj",      "linear",    (BATCH, SEQ, HIDDEN),     {"out_features": HIDDEN}),
    ("layernorm_1", "layernorm", (BATCH, SEQ, HIDDEN),     {}),
    ("ffn_up",      "linear",    (BATCH, SEQ, HIDDEN),     {"out_features": FFN}),
    ("ffn_gelu",    "gelu",      (BATCH, SEQ, FFN),        {}),
    ("ffn_down",    "linear",    (BATCH, SEQ, FFN),        {"out_features": HIDDEN}),
    ("layernorm_2", "layernorm", (BATCH, SEQ, HIDDEN),     {}),
]


def main():
    print("=" * 70)
    print("  方式 2: 算子自动生成 API (input:fp16, weight:bfpp4)")
    print("=" * 70)

    gen = DataGenerator(seed=42, precision=PRECISION)
    config = CompareConfig(fuzzy_min_qsnr=1.0, fuzzy_min_cosine=0.85,
                           fuzzy_max_exceed_ratio=0.15,
                           fuzzy_atol=0.5, fuzzy_rtol=0.5)

    print(f"\n  精度: input={PRECISION.input_dtype}, weight={PRECISION.weight_dtype}, "
          f"compute={PRECISION.compute_dtype}")
    print(f"  Encoder: {len(ENCODER_OPS)} 个算子\n")

    # 逐算子生成四种比数
    print(f"  {'算子':<15} {'Shape':<20} {'QSNR(hw)':>10} {'Cos(hw)':>10} {'QSNR(local)':>12}")
    print(f"  {'-'*69}")

    all_tracks = []
    for name, op_name, shape, kwargs in ENCODER_OPS:
        tracks = gen.generate_four_track(
            op_name, input_shape=shape, precision=PRECISION, **kwargs,
        )
        all_tracks.append((name, tracks))

        # HW bfpp4 比数
        hw_qsnr, hw_cos = "N/A", "N/A"
        if tracks.golden_hw is not None:
            r_hw = FuzzyStrategy.compare(tracks.golden_pure, tracks.golden_hw, config=config)
            hw_qsnr = f"{r_hw.qsnr:.2f}"
            hw_cos = f"{r_hw.cosine:.6f}"

        # Local fp16 比数
        local_qsnr = "N/A"
        if tracks.golden_local is not None:
            r_local = FuzzyStrategy.compare(tracks.golden_pure, tracks.golden_local, config=config)
            local_qsnr = f"{r_local.qsnr:.2f}"

        print(f"  {name:<15} {str(tracks.golden_pure.shape):<20} "
              f"{hw_qsnr:>10} {hw_cos:>10} {local_qsnr:>12}")

    # 结果断言
    assert len(all_tracks) == 10, f"应有 10 个算子, 实际 {len(all_tracks)}"
    for name, tracks in all_tracks:
        assert tracks.golden_pure is not None, f"{name}: golden_pure 不应为 None"
        assert tracks.golden_pure.size > 0, f"{name}: golden_pure 不应为空"
    print("  所有断言通过!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
