#!/usr/bin/env python
"""Encoder Bit 级比对 Demo (BFP8) — Model DSL

使用 Model DSL 构建 Encoder，生成 BFP8 golden 后随机篡改 16 字节，
一键式逐算子 + 全局 bit 比对。

Encoder 结构:
    Q/K/V proj → Softmax → O proj → LayerNorm
    → FFN_up → GELU → FFN_down → LayerNorm

输出:
    1. 一键式: 逐算子 + 全局 bit 级分析 (compare_model_bitwise)
    2. 全模型 bit 级热力图 (文本 + SVG)
    3. Per-bit 错误分布条形图 SVG

运行: python -m demos.compare.08_bitwise_encoder_bfp8.run
"""
import numpy as np
from pathlib import Path

from aidevtools.datagen import Model
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.formats.quantize import simulate_quantize
from aidevtools.compare.bitwise import (
    FP32,
    compare_model_bitwise,
    print_model_bit_analysis,
    print_bit_analysis,
    print_bit_heatmap,
    gen_bit_heatmap_svg,
    gen_perbit_bar_svg,
)

# ---- 全局参数 ----
BATCH, SEQ, HIDDEN = 2, 16, 64
FFN_DIM = HIDDEN * 4  # 256
QTYPE = "bfp8"
SEED = 42
CORRUPT_BYTES = 16

PRECISION = PrecisionConfig(
    input_dtype="fp16",
    weight_dtype="bfp4",
    compute_dtype="fp32",
    output_dtype="bfp8",
)


# =========================================================
# Model DSL 构建 Encoder
# =========================================================
def build_encoder_dsl():
    """
    用 Model DSL 构建 Encoder:
    - Q/K/V 投影 (linear)
    - Softmax → O 投影 → LayerNorm
    - FFN: up → GELU → down → LayerNorm
    """
    with Model(seed=SEED, precision=PRECISION, qtype=QTYPE) as m:
        x = m.input((BATCH, SEQ, HIDDEN))

        # === Self-Attention ===
        q = m.linear(x, out_features=HIDDEN)
        k = m.linear(x, out_features=HIDDEN)
        v = m.linear(x, out_features=HIDDEN)
        attn = m.softmax(q)
        o_proj = m.linear(attn, out_features=HIDDEN)
        ln1 = m.layernorm(o_proj)

        # === FFN ===
        ffn_up = m.linear(ln1, out_features=FFN_DIM)
        ffn_act = m.gelu(ffn_up)
        ffn_down = m.linear(ffn_act, out_features=HIDDEN)
        output = m.layernorm(ffn_down)

    # 收集逐算子 golden
    per_op_goldens = {}
    op_names = [
        "Q_proj", "K_proj", "V_proj", "Softmax",
        "O_proj", "LayerNorm_1",
        "FFN_up", "GELU", "FFN_down", "LayerNorm_2",
    ]
    for tensor, name in zip(m.outputs, op_names):
        per_op_goldens[name] = tensor.golden.astype(np.float32)

    final_output = m.final_output.astype(np.float32)
    return final_output, per_op_goldens


# =========================================================
# 篡改: 随机翻转 N 字节
# =========================================================
def corrupt_bytes(data: np.ndarray, n_bytes: int, seed: int = 0) -> np.ndarray:
    """在 BFP8 数据中随机篡改 n_bytes 个字节"""
    corrupted = np.ascontiguousarray(data).copy()
    original_shape = corrupted.shape
    flat = corrupted.reshape(-1)
    raw = flat.view(np.uint8)

    rng = np.random.RandomState(seed)
    total_bytes = len(raw)
    positions = rng.choice(total_bytes, size=min(n_bytes, total_bytes), replace=False)
    for pos in positions:
        flip_mask = rng.randint(1, 256, dtype=np.uint8)
        raw[pos] ^= flip_mask

    return flat.reshape(original_shape)


# =========================================================
# 主函数
# =========================================================
def main():
    workspace = Path(__file__).parent / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  Encoder Bit 级比对 Demo (BFP8) — Model DSL")
    print(f"  batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, ffn={FFN_DIM}")
    print(f"  格式: {QTYPE}, 篡改: {CORRUPT_BYTES} 字节")
    print("=" * 75)

    # --- 第一部分: Model DSL 构建 Encoder ---
    print("\n" + "-" * 75)
    print("  第一部分: Model DSL 构建 Encoder")
    print("-" * 75)

    final_output, per_op_goldens = build_encoder_dsl()
    print(f"\n  Encoder 结构:")
    print(f"    输入:   ({BATCH}, {SEQ}, {HIDDEN})")
    print(f"    FFN:    {HIDDEN} → {FFN_DIM} → {HIDDEN}")
    print(f"    输出:   {final_output.shape}")
    print(f"    算子数: {len(per_op_goldens)}")

    # --- 第二部分: 一键式 bit 比对 (逐算子 + 全局) ---
    print("\n" + "-" * 75)
    print(f"  第二部分: 一键式 bit 比对 (compare_model_bitwise)")
    print("-" * 75)

    # 构建 per_op_pairs: {name: (golden_bfp8, corrupted_bfp8)}
    per_op_pairs = {}
    for op_name, golden_np in per_op_goldens.items():
        g_bfp8 = simulate_quantize(golden_np, QTYPE)
        c_bfp8 = corrupt_bytes(g_bfp8, max(1, CORRUPT_BYTES // len(per_op_goldens)),
                               seed=int.from_bytes(op_name.encode(), 'little') % 2**31)
        per_op_pairs[op_name] = (g_bfp8, c_bfp8)

    # 全局最终输出
    golden_bfp8 = simulate_quantize(final_output, QTYPE)
    corrupted_bfp8 = corrupt_bytes(golden_bfp8, CORRUPT_BYTES, seed=123)

    # 一键比对
    model_result = compare_model_bitwise(
        per_op_pairs=per_op_pairs,
        fmt=FP32,
        final_pair=(golden_bfp8, corrupted_bfp8),
    )
    print_model_bit_analysis(model_result, name="Encoder")

    # --- 第三部分: 全局热力图 + SVG ---
    print("-" * 75)
    print("  第三部分: 全局热力图 + Per-bit 分布")
    print("-" * 75)

    print_bit_heatmap(golden_bfp8, corrupted_bfp8, fmt=FP32,
                      block_size=64, cols=32)

    svg_heatmap = str(workspace / "encoder_bit_heatmap.svg")
    gen_bit_heatmap_svg(golden_bfp8, corrupted_bfp8, svg_heatmap,
                        fmt=FP32, block_size=64)
    print(f"  SVG 热力图: {svg_heatmap}")

    svg_perbit = str(workspace / "encoder_perbit_bar.svg")
    gen_perbit_bar_svg(model_result.global_result, svg_perbit)
    print(f"  Per-bit 分布图: {svg_perbit}")

    # --- 第四部分: Softmax 详细分析 ---
    print("\n" + "-" * 75)
    print("  第四部分: Softmax 输出详细 bit 分析")
    print("-" * 75)

    softmax_result = model_result.per_op["Softmax"]
    print_bit_analysis(softmax_result, name="Softmax")

    svg_softmax = str(workspace / "softmax_perbit_bar.svg")
    gen_perbit_bar_svg(softmax_result, svg_softmax)
    print(f"  Softmax Per-bit 分布图: {svg_softmax}")

    # --- 汇总 ---
    print("\n" + "=" * 75)
    s = model_result.global_result.summary
    print(f"  汇总:")
    print(f"    全模型 bit 差异率: {s.diff_ratio:.4%}")
    print(f"    符号位翻转:       {s.sign_flip_count} 个 ({s.sign_flip_ratio:.4%})")
    print(f"    指数域偏移:       {s.exponent_diff_count} 个 (max shift={s.max_exponent_diff})")
    print(f"    尾数差异:         {s.mantissa_diff_count} 个")
    print(f"    CRITICAL 告警:    {'YES' if model_result.has_critical else 'NO'}")
    print(f"\n  生成文件:")
    print(f"    {svg_heatmap}")
    print(f"    {svg_perbit}")
    print(f"    {svg_softmax}")
    print("=" * 75)


if __name__ == "__main__":
    main()
