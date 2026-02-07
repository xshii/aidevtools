#!/usr/bin/env python
"""Autoload Replay — 导出 → 篡改 → 自动加载 → 比对

演示 Model data_dir 自动回放能力:
1. 用 Model DSL (QA-aware) 构建 Encoder，export 为 DUT bin
2. 复制目录后篡改最后两步 (FFN_down, LayerNorm_2) 的权重
3. Model(data_dir=...) 自动加载篡改数据，通过 cpu_golden 重算
4. 逐算子 + 全局 bit 级比对，发现最后两步异常

Encoder 结构:
    Q/K/V proj → Softmax → O proj → LayerNorm_1
    → FFN_up → GELU → FFN_down → LayerNorm_2

运行: python -m demos.compare.09_autoload_replay.run
"""
import shutil
import numpy as np
from pathlib import Path

from aidevtools.datagen import Model
from aidevtools.frontend.types import PrecisionConfig
from aidevtools.compare.bitwise import (
    FP32,
    compare_model_bitwise,
    print_model_bit_analysis,
    gen_bit_heatmap_svg,
    gen_perbit_bar_svg,
)

# ---- 全局参数 ----
BATCH, SEQ, HIDDEN = 1, 8, 32
FFN_DIM = HIDDEN * 4  # 128
QTYPE = "bfp8"
SEED = 42

PRECISION = PrecisionConfig(
    input_dtype="fp16",
    weight_dtype="bfp4",
    compute_dtype="fp32",
    output_dtype="bfp8",
    qa_aware=True,
    qa_center=1.0,
    qa_amplitude=0.5,
)

OP_NAMES = [
    "Q_proj", "K_proj", "V_proj", "Softmax",
    "O_proj", "LayerNorm_1",
    "FFN_up", "GELU", "FFN_down", "LayerNorm_2",
]


# =========================================================
# Encoder 构建 (Model DSL)
# =========================================================
def build_encoder(m):
    """构建 Encoder 结构"""
    x = m.input((BATCH, SEQ, HIDDEN))
    # Self-Attention
    q = m.linear(x, out_features=HIDDEN)
    k = m.linear(x, out_features=HIDDEN)
    v = m.linear(x, out_features=HIDDEN)
    attn = m.softmax(q)
    o_proj = m.linear(attn, out_features=HIDDEN)
    ln1 = m.layernorm(o_proj)
    # FFN
    ffn_up = m.linear(ln1, out_features=FFN_DIM)
    ffn_act = m.gelu(ffn_up)
    ffn_down = m.linear(ffn_act, out_features=HIDDEN)
    output = m.layernorm(ffn_down)
    return output


# =========================================================
# 篡改: 随机翻转 bin 文件中的字节
# =========================================================
def corrupt_bin_file(path: Path, n_bytes: int = 8, seed: int = 0):
    """在 bin 文件中随机翻转 n_bytes 个字节"""
    raw = bytearray(path.read_bytes())
    rng = np.random.RandomState(seed)
    positions = rng.choice(len(raw), size=min(n_bytes, len(raw)), replace=False)
    for pos in positions:
        flip = rng.randint(1, 256)
        raw[pos] ^= flip
    path.write_bytes(bytes(raw))
    return len(positions)


# =========================================================
# 主函数
# =========================================================
def main():
    workspace = Path(__file__).parent / "workspace"
    golden_dir = workspace / "golden_original"
    corrupt_dir = workspace / "golden_corrupted"
    BM = "enc"

    # 清理
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    print("=" * 75)
    print("  Autoload Replay Demo — 导出 → 篡改 → 自动加载 → 比对")
    print(f"  shape=({BATCH},{SEQ},{HIDDEN}), ffn={FFN_DIM}, qtype={QTYPE}")
    print(f"  QA-aware: center={PRECISION.qa_center}, amplitude={PRECISION.qa_amplitude}")
    print("=" * 75)

    # ---- Phase 1: 生成 + 导出 ----
    print("\n" + "-" * 75)
    print("  Phase 1: Model DSL (QA-aware) 生成 Encoder → export DUT bin")
    print("-" * 75)

    with Model(seed=SEED, precision=PRECISION, qtype=QTYPE) as m:
        build_encoder(m)

    exported = m.export(str(golden_dir), bm=BM)
    print(f"\n  导出 {len(exported)} 个 tensor 到 {golden_dir}")
    for name, path in exported.items():
        print(f"    {path.name}")

    # ---- Phase 2: 复制 + 篡改最后两步权重 ----
    print("\n" + "-" * 75)
    print("  Phase 2: 复制目录，篡改 FFN_down + LayerNorm_2 的权重")
    print("-" * 75)

    shutil.copytree(golden_dir, corrupt_dir)

    # 找到最后两步的 weight + bias 文件
    # linear_5=FFN_down, layernorm_1=LN2
    corrupt_targets = ["linear_5", "layernorm_1"]
    corrupted_files = []
    for f in sorted(corrupt_dir.glob("*.bin")):
        for target in corrupt_targets:
            if target in f.name:
                n = corrupt_bin_file(f, n_bytes=max(16, f.stat().st_size // 4),
                                     seed=int.from_bytes(f.name.encode(), 'little') % 2**31)
                corrupted_files.append((f.name, n))
                print(f"  篡改: {f.name} ({n} bytes / {f.stat().st_size} total)")

    if not corrupted_files:
        print("  警告: 未找到目标文件, 列出所有文件:")
        for f in sorted(corrupt_dir.glob("*.bin")):
            print(f"    {f.name}")
        return

    # ---- Phase 3: Autoload 回放 ----
    print("\n" + "-" * 75)
    print("  Phase 3: Autoload 回放 (原始 vs 篡改)")
    print("-" * 75)

    # 回放原始数据
    with Model(data_dir=str(golden_dir), bm=BM, qtype=QTYPE, precision=PRECISION) as m_orig:
        build_encoder(m_orig)
    original_outputs = {name: t.golden.astype(np.float32) for name, t in zip(OP_NAMES, m_orig.outputs)}

    # 回放篡改数据
    with Model(data_dir=str(corrupt_dir), bm=BM, qtype=QTYPE, precision=PRECISION) as m_corr:
        build_encoder(m_corr)
    corrupted_outputs = {name: t.golden.astype(np.float32) for name, t in zip(OP_NAMES, m_corr.outputs)}

    print(f"\n  原始回放: {len(original_outputs)} ops")
    print(f"  篡改回放: {len(corrupted_outputs)} ops")

    # ---- Phase 4: 逐算子 + 全局 bit 比对 ----
    print("\n" + "-" * 75)
    print("  Phase 4: 逐算子 + 全局 bit 级比对")
    print("-" * 75)

    per_op_pairs = {}
    for name in OP_NAMES:
        per_op_pairs[name] = (original_outputs[name], corrupted_outputs[name])

    # 全局: 最终输出
    final_orig = m_orig.final_output.astype(np.float32)
    final_corr = m_corr.final_output.astype(np.float32)

    result = compare_model_bitwise(
        per_op_pairs=per_op_pairs,
        fmt=FP32,
        final_pair=(final_orig, final_corr),
    )
    print_model_bit_analysis(result, name="Encoder (Original vs Corrupted)")

    # ---- SVG ----
    if result.global_result:
        svg_path = str(workspace / "global_perbit_bar.svg")
        gen_perbit_bar_svg(result.global_result, svg_path)
        print(f"\n  Per-bit 分布图: {svg_path}")

    svg_heatmap = str(workspace / "global_heatmap.svg")
    gen_bit_heatmap_svg(final_orig, final_corr, svg_heatmap, fmt=FP32, block_size=32)
    print(f"  全局热力图:     {svg_heatmap}")

    # ---- 汇总 ----
    print("\n" + "=" * 75)
    print("  汇总:")
    n_critical = sum(1 for r in result.per_op.values() if r.has_critical)
    n_diff = sum(1 for r in result.per_op.values() if r.summary.diff_elements > 0)
    print(f"    总算子:       {len(result.per_op)}")
    print(f"    有差异算子:   {n_diff}")
    print(f"    CRITICAL 算子: {n_critical}")
    if result.global_result:
        s = result.global_result.summary
        print(f"    全局 diff:    {s.diff_ratio:.4%}")
    print(f"    全局 CRITICAL: {'YES' if result.has_critical else 'NO'}")

    # 验证: 最后两步应有异常
    last_two = OP_NAMES[-2:]  # FFN_down, LayerNorm_2
    for name in last_two:
        r = result.per_op[name]
        status = "CRITICAL" if r.has_critical else ("DIFF" if r.summary.diff_elements > 0 else "OK")
        print(f"    {name}: {status}")

    print("=" * 75)


if __name__ == "__main__":
    main()
