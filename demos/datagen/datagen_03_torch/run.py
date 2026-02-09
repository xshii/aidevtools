#!/usr/bin/env python
"""方式 4: PyTorch 劫持 (Golden Mode) 构建 Encoder

通过 import aidevtools.golden 劫持 torch.nn.functional，
用户写标准 PyTorch 代码，框架自动计算 golden。输入 bfp8，权重 bfp4 精度。

Encoder 结构:
    Q/K/V proj → Softmax → O proj → LayerNorm
    → FFN_up → GELU → FFN_down → LayerNorm

运行: python demos/datagen/03_torch_golden/run.py
"""
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
QTYPE_INPUT = "bfp8"
QTYPE_WEIGHT = "bfp4"


def main():
    print("=" * 70)
    print("  方式 4: PyTorch 劫持 (Golden Mode) (input:bfp8, weight:bfp4)")
    print("=" * 70)

    if not HAS_TORCH:
        print("\n  torch 未安装, 跳过本 demo")
        print("  安装: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("=" * 70)
        return

    import aidevtools.golden as golden
    golden.clear()

    torch.manual_seed(42)

    # ========== Self-Attention ==========
    print("\n  [Self-Attention]")
    x = torch.randn(BATCH, SEQ, HIDDEN)
    print(f"    Input: {tuple(x.shape)}")

    # Q/K/V projections
    w_q = torch.randn(HIDDEN, HIDDEN) * 0.02
    w_k = torch.randn(HIDDEN, HIDDEN) * 0.02
    w_v = torch.randn(HIDDEN, HIDDEN) * 0.02

    q = F.linear(x, w_q)
    k = F.linear(x, w_k)
    v = F.linear(x, w_v)
    print(f"    Q/K/V: {tuple(q.shape)}")

    # Softmax
    attn = F.softmax(q, dim=-1)
    print(f"    Attention: {tuple(attn.shape)}")

    # Output projection
    w_o = torch.randn(HIDDEN, HIDDEN) * 0.02
    out = F.linear(attn, w_o)
    print(f"    O proj: {tuple(out.shape)}")

    # LayerNorm 1
    ln1 = F.layer_norm(out, normalized_shape=(HIDDEN,))
    print(f"    LayerNorm 1: {tuple(ln1.shape)}")

    # ========== FFN ==========
    print("\n  [FFN]")
    w_up = torch.randn(FFN, HIDDEN) * 0.02
    ffn_up = F.linear(ln1, w_up)
    print(f"    FFN up: {tuple(ffn_up.shape)}")

    ffn_act = F.gelu(ffn_up)
    print(f"    GELU: {tuple(ffn_act.shape)}")

    w_down = torch.randn(HIDDEN, FFN) * 0.02
    ffn_down = F.linear(ffn_act, w_down)
    print(f"    FFN down: {tuple(ffn_down.shape)}")

    output = F.layer_norm(ffn_down, normalized_shape=(HIDDEN,))
    print(f"    LayerNorm 2: {tuple(output.shape)}")

    # 报告
    records = golden.records()
    print(f"\n  Golden 记录: {len(records)} 个算子")
    if records:
        print(f"\n  {'算子':<20} {'输入 Shape':<20} {'输出 Shape':<20}")
        print(f"  {'-'*60}")
        for r in records:
            in_shape = str(tuple(r.get("input", np.array([])).shape))
            out_shape = str(tuple(r.get("golden", np.array([])).shape))
            print(f"  {r['name']:<20} {in_shape:<20} {out_shape:<20}")

    # 量化比对: input bfp8, weight bfp4
    from aidevtools.formats.quantize import simulate_quantize
    from aidevtools.compare.fuzzy import compare_fuzzy
    from aidevtools.compare.types import CompareConfig

    config = CompareConfig(fuzzy_min_qsnr=1.0, fuzzy_min_cosine=0.85,
                           fuzzy_max_exceed_ratio=0.15,
                           fuzzy_atol=0.5, fuzzy_rtol=0.5)
    out_np = output.detach().numpy().astype(np.float32)
    dut = simulate_quantize(out_np, QTYPE_INPUT)
    r = compare_fuzzy(out_np, dut, config)
    print(f"\n  最终输出 {QTYPE_INPUT} 比对: QSNR={r.qsnr:.2f}, Cosine={r.cosine:.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
