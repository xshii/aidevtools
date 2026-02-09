#!/usr/bin/env python
"""Compare Demo 01: Encoder - Pure fp32 vs fp16 计算精度

测试 fp16 计算引入的精度损失（No DUT）

模型: Encoder
  - hidden_dim: 64
  - ffn_dim: 256
  - seq_len: 16
  - batch: 2
  - 10个算子

前端: PyTorch 劫持

对比:
  - Pure: 相同随机数 → fp32 计算 → fp32 输出
  - FP16: 相同随机数 → 转 fp16 → fp16 计算 → 转 fp32 输出

策略: FuzzyStrategy (QSNR + Cosine)

运行: python demos/compare/compare_01_encoder_pure_vs_fp16/run.py
"""
import sys
import numpy as np

# 确保能找到 aidevtools
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aidevtools.compare import CompareEngine, CompareConfig, print_strategy_table

# 全局参数
BATCH, SEQ, HIDDEN, FFN = 2, 16, 64, 256
SEED = 42

COMPARE_CFG = CompareConfig(
    fuzzy_min_qsnr=30.0,
    fuzzy_min_cosine=0.999,
    fuzzy_max_exceed_ratio=0.01,
    fuzzy_atol=1e-5,
    fuzzy_rtol=1e-3,
)

ENCODER_OPS = [
    "Q_proj", "K_proj", "V_proj", "attn_softmax",
    "O_proj", "layernorm_1",
    "ffn_up", "ffn_gelu", "ffn_down", "layernorm_2",
]


def generate_encoder_torch_pure():
    """PyTorch 生成 Encoder - 纯 fp32"""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("错误: 需要安装 PyTorch")
        print("  pip install torch")
        sys.exit(1)

    torch.manual_seed(SEED)

    outputs = {}
    x = torch.randn(BATCH, SEQ, HIDDEN)

    # Q/K/V projections
    wq = torch.randn(HIDDEN, HIDDEN) * 0.02
    q = F.linear(x, wq)
    outputs["Q_proj"] = q.detach().numpy().astype(np.float32)

    wk = torch.randn(HIDDEN, HIDDEN) * 0.02
    k = F.linear(x, wk)
    outputs["K_proj"] = k.detach().numpy().astype(np.float32)

    wv = torch.randn(HIDDEN, HIDDEN) * 0.02
    v = F.linear(x, wv)
    outputs["V_proj"] = v.detach().numpy().astype(np.float32)

    # Attention
    attn = F.softmax(q, dim=-1)
    outputs["attn_softmax"] = attn.detach().numpy().astype(np.float32)

    # Output projection
    wo = torch.randn(HIDDEN, HIDDEN) * 0.02
    out = F.linear(attn, wo)
    outputs["O_proj"] = out.detach().numpy().astype(np.float32)

    # LayerNorm 1
    ln1 = F.layer_norm(out, normalized_shape=(HIDDEN,))
    outputs["layernorm_1"] = ln1.detach().numpy().astype(np.float32)

    # FFN
    w_up = torch.randn(FFN, HIDDEN) * 0.02
    up = F.linear(ln1, w_up)
    outputs["ffn_up"] = up.detach().numpy().astype(np.float32)

    act = F.gelu(up)
    outputs["ffn_gelu"] = act.detach().numpy().astype(np.float32)

    w_down = torch.randn(HIDDEN, FFN) * 0.02
    down = F.linear(act, w_down)
    outputs["ffn_down"] = down.detach().numpy().astype(np.float32)

    # LayerNorm 2
    output = F.layer_norm(down, normalized_shape=(HIDDEN,))
    outputs["layernorm_2"] = output.detach().numpy().astype(np.float32)

    return outputs


def generate_encoder_torch_fp16():
    """PyTorch 生成 Encoder - fp16 精度计算"""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("错误: 需要安装 PyTorch")
        print("  pip install torch")
        sys.exit(1)

    torch.manual_seed(SEED)

    outputs = {}
    # 先生成 fp32，再转 fp16（与 pure 使用相同的随机数）
    x = torch.randn(BATCH, SEQ, HIDDEN).half()

    # Q/K/V projections
    wq = (torch.randn(HIDDEN, HIDDEN) * 0.02).half()
    q = F.linear(x, wq)
    outputs["Q_proj"] = q.detach().numpy().astype(np.float32)

    wk = (torch.randn(HIDDEN, HIDDEN) * 0.02).half()
    k = F.linear(x, wk)
    outputs["K_proj"] = k.detach().numpy().astype(np.float32)

    wv = (torch.randn(HIDDEN, HIDDEN) * 0.02).half()
    v = F.linear(x, wv)
    outputs["V_proj"] = v.detach().numpy().astype(np.float32)

    # Attention
    attn = F.softmax(q, dim=-1)
    outputs["attn_softmax"] = attn.detach().numpy().astype(np.float32)

    # Output projection
    wo = (torch.randn(HIDDEN, HIDDEN) * 0.02).half()
    out = F.linear(attn, wo)
    outputs["O_proj"] = out.detach().numpy().astype(np.float32)

    # LayerNorm 1 (fp16 不支持，需要转fp32)
    ln1 = F.layer_norm(out.float(), normalized_shape=(HIDDEN,)).half()
    outputs["layernorm_1"] = ln1.detach().numpy().astype(np.float32)

    # FFN
    w_up = (torch.randn(FFN, HIDDEN) * 0.02).half()
    up = F.linear(ln1, w_up)
    outputs["ffn_up"] = up.detach().numpy().astype(np.float32)

    act = F.gelu(up)
    outputs["ffn_gelu"] = act.detach().numpy().astype(np.float32)

    w_down = (torch.randn(HIDDEN, FFN) * 0.02).half()
    down = F.linear(act, w_down)
    outputs["ffn_down"] = down.detach().numpy().astype(np.float32)

    # LayerNorm 2 (fp16 不支持，需要转fp32)
    output = F.layer_norm(down.float(), normalized_shape=(HIDDEN,)).half()
    outputs["layernorm_2"] = output.detach().numpy().astype(np.float32)

    return outputs


def main():
    print("=" * 75)
    print("  Compare Demo 01: Encoder - Pure fp32 vs fp16 计算精度")
    print(f"  模型: Encoder (batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, ffn={FFN})")
    print(f"  策略: StandardStrategy (L1: Exact+Bitwise, L2: Fuzzy+Sanity)")
    print("=" * 75)

    # 生成两组数据
    print("\n[1/3] 生成 Pure fp32 数据 (PyTorch)")
    pure_outputs = generate_encoder_torch_pure()
    print(f"  ✓ 生成完成，{len(pure_outputs)} 个算子")

    print("\n[2/3] 生成 FP16 计算数据 (PyTorch, 相同随机数)")
    fp16_outputs = generate_encoder_torch_fp16()
    print(f"  ✓ 生成完成，{len(fp16_outputs)} 个算子")

    # 比对
    print("\n[3/3] 使用 StandardStrategy 比对")
    engine = CompareEngine.standard(config=COMPARE_CFG)

    results = {}
    for op_name in ENCODER_OPS:
        if op_name not in pure_outputs or op_name not in fp16_outputs:
            print(f"  跳过 {op_name}: 数据缺失")
            continue

        golden = pure_outputs[op_name]
        dut = fp16_outputs[op_name]
        results[op_name] = engine.run(dut=dut, golden=golden)

    # 转换为 print_strategy_table 需要的格式
    results_list = [results[name] for name in ENCODER_OPS if name in results]
    names_list = [name for name in ENCODER_OPS if name in results]

    # 使用标准报告工具打印结果
    print_strategy_table(results_list, names_list)

    # 简单总结
    print("\n" + "=" * 75)
    print("  总结:")
    print("    - L1 (Exact): 预期失败，因为 fp16 有量化误差")
    print("    - L2 (Fuzzy): 预期通过，QSNR ~67dB 在可接受范围")
    print("    - 展示了分级策略的价值：快速失败 → 深度分析")
    print("=" * 75)

    # 断言验证
    print("\n[验证] 检查结果是否符合预期...")

    # 1. L1 (Exact) 应该全部失败（因为 fp16 有量化误差）
    exact_all_fail = all(
        not r["exact"].passed if r.get("exact") else True
        for r in results_list
    )
    assert exact_all_fail, "Exact 策略应该全部失败（fp16 有量化误差）"
    print("  ✓ Exact 策略全部失败（符合预期）")

    # 2. Sanity 应该全部通过（无 NaN/Inf）
    sanity_all_pass = all(
        r["sanity"].valid if r.get("sanity") else True
        for r in results_list
    )
    assert sanity_all_pass, "Sanity 策略应该全部通过（无 NaN/Inf）"
    print("  ✓ Sanity 策略全部通过（无异常值）")

    # 3. QSNR 应该在合理范围内（>60 dB 说明精度损失很小）
    qsnrs = [
        r["fuzzy_pure"].qsnr
        for r in results_list
        if r.get("fuzzy_pure") and r["fuzzy_pure"].qsnr != float("inf")
    ]
    if qsnrs:
        avg_qsnr = np.mean(qsnrs)
        assert avg_qsnr > 60, f"平均 QSNR 应该 >60 dB，实际 {avg_qsnr:.1f} dB"
        print(f"  ✓ 平均 QSNR {avg_qsnr:.1f} dB（精度损失很小）")

    # 4. Cosine 相似度应该接近 1.0
    cosines = [
        r["fuzzy_pure"].cosine
        for r in results_list
        if r.get("fuzzy_pure")
    ]
    if cosines:
        avg_cosine = np.mean(cosines)
        assert avg_cosine > 0.999, f"平均 Cosine 应该 >0.999，实际 {avg_cosine:.6f}"
        print(f"  ✓ 平均 Cosine {avg_cosine:.6f}（高度相似）")

    # 5. 不应该有 GOLDEN_SUSPECT 或 BOTH_SUSPECT（golden 是纯 fp32，应该是正确的）
    from aidevtools.compare import CompareStatus
    bad_status = [
        name for name, r in zip(names_list, results_list)
        if r.get("status") in [CompareStatus.GOLDEN_SUSPECT, CompareStatus.BOTH_SUSPECT]
    ]
    assert len(bad_status) == 0, f"不应该怀疑 golden 数据: {bad_status}"
    print("  ✓ Golden 数据无异常")

    print("\n✓ 所有验证通过！")


if __name__ == "__main__":
    main()
