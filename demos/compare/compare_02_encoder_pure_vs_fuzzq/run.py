#!/usr/bin/env python
"""Compare Demo 02: Encoder - Pure vs FuzzQ

测试量化反量化（FuzzQ）引入的数据误差（No DUT）

模型: Encoder
  - hidden_dim: 64
  - ffn_dim: 256
  - seq_len: 16
  - batch: 2
  - 10个算子

前端: PyTorch 劫持

对比:
  - Pure: 纯随机 fp32 → fp32 计算
  - FuzzQ: 纯随机 fp32 → 量化到 bfp8 → 反量化 → fp32 计算

策略: StandardStrategy (L1: Exact+Bitwise, L2: Fuzzy+Sanity)

运行: python demos/compare/compare_02_encoder_pure_vs_fuzzq/run.py
"""
import sys
import numpy as np

# 确保能找到 aidevtools
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from aidevtools.compare import CompareEngine, CompareConfig, print_strategy_table, CompareStatus
from aidevtools.datagen import DataGenerator
from aidevtools.formats.quantize import simulate_quantize

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


def generate_encoder_pure():
    """使用 DataGenerator 生成 Encoder - 纯随机 fp32"""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("错误: 需要安装 PyTorch")
        print("  pip install torch")
        sys.exit(1)

    # 纯随机生成
    gen = DataGenerator(seed=SEED)

    outputs = {}
    x = torch.from_numpy(gen.randn((BATCH, SEQ, HIDDEN)).array)

    # Q/K/V projections
    wq = torch.from_numpy(gen.randn((HIDDEN, HIDDEN)).array * 0.02)
    q = F.linear(x, wq)
    outputs["Q_proj"] = q.detach().numpy().astype(np.float32)

    wk = torch.from_numpy(gen.randn((HIDDEN, HIDDEN)).array * 0.02)
    k = F.linear(x, wk)
    outputs["K_proj"] = k.detach().numpy().astype(np.float32)

    wv = torch.from_numpy(gen.randn((HIDDEN, HIDDEN)).array * 0.02)
    v = F.linear(x, wv)
    outputs["V_proj"] = v.detach().numpy().astype(np.float32)

    # Attention
    attn = F.softmax(q, dim=-1)
    outputs["attn_softmax"] = attn.detach().numpy().astype(np.float32)

    # Output projection
    wo = torch.from_numpy(gen.randn((HIDDEN, HIDDEN)).array * 0.02)
    out = F.linear(attn, wo)
    outputs["O_proj"] = out.detach().numpy().astype(np.float32)

    # LayerNorm 1
    ln1 = F.layer_norm(out, normalized_shape=(HIDDEN,))
    outputs["layernorm_1"] = ln1.detach().numpy().astype(np.float32)

    # FFN
    w_up = torch.from_numpy(gen.randn((FFN, HIDDEN)).array * 0.02)
    up = F.linear(ln1, w_up)
    outputs["ffn_up"] = up.detach().numpy().astype(np.float32)

    act = F.gelu(up)
    outputs["ffn_gelu"] = act.detach().numpy().astype(np.float32)

    w_down = torch.from_numpy(gen.randn((HIDDEN, FFN)).array * 0.02)
    down = F.linear(act, w_down)
    outputs["ffn_down"] = down.detach().numpy().astype(np.float32)

    # LayerNorm 2
    output = F.layer_norm(down, normalized_shape=(HIDDEN,))
    outputs["layernorm_2"] = output.detach().numpy().astype(np.float32)

    return outputs


def generate_encoder_fuzzq():
    """使用 DataGenerator 生成 Encoder - 量化反量化（FuzzQ）"""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("错误: 需要安装 PyTorch")
        print("  pip install torch")
        sys.exit(1)

    # 量化反量化生成
    gen = DataGenerator(seed=SEED)

    outputs = {}
    # 输入数据：fp32 → 量化到 bfp8 → 反量化
    x_pure = gen.randn((BATCH, SEQ, HIDDEN)).array
    x_fuzzq = simulate_quantize(x_pure, "bfp8")
    x = torch.from_numpy(x_fuzzq)

    # Q/K/V projections（权重也量化反量化）
    wq_pure = gen.randn((HIDDEN, HIDDEN)).array * 0.02
    wq_fuzzq = simulate_quantize(wq_pure, "bfp8")
    wq = torch.from_numpy(wq_fuzzq)
    q = F.linear(x, wq)
    outputs["Q_proj"] = q.detach().numpy().astype(np.float32)

    wk_pure = gen.randn((HIDDEN, HIDDEN)).array * 0.02
    wk_fuzzq = simulate_quantize(wk_pure, "bfp8")
    wk = torch.from_numpy(wk_fuzzq)
    k = F.linear(x, wk)
    outputs["K_proj"] = k.detach().numpy().astype(np.float32)

    wv_pure = gen.randn((HIDDEN, HIDDEN)).array * 0.02
    wv_fuzzq = simulate_quantize(wv_pure, "bfp8")
    wv = torch.from_numpy(wv_fuzzq)
    v = F.linear(x, wv)
    outputs["V_proj"] = v.detach().numpy().astype(np.float32)

    # Attention
    attn = F.softmax(q, dim=-1)
    outputs["attn_softmax"] = attn.detach().numpy().astype(np.float32)

    # Output projection
    wo_pure = gen.randn((HIDDEN, HIDDEN)).array * 0.02
    wo_fuzzq = simulate_quantize(wo_pure, "bfp8")
    wo = torch.from_numpy(wo_fuzzq)
    out = F.linear(attn, wo)
    outputs["O_proj"] = out.detach().numpy().astype(np.float32)

    # LayerNorm 1
    ln1 = F.layer_norm(out, normalized_shape=(HIDDEN,))
    outputs["layernorm_1"] = ln1.detach().numpy().astype(np.float32)

    # FFN
    w_up_pure = gen.randn((FFN, HIDDEN)).array * 0.02
    w_up_fuzzq = simulate_quantize(w_up_pure, "bfp8")
    w_up = torch.from_numpy(w_up_fuzzq)
    up = F.linear(ln1, w_up)
    outputs["ffn_up"] = up.detach().numpy().astype(np.float32)

    act = F.gelu(up)
    outputs["ffn_gelu"] = act.detach().numpy().astype(np.float32)

    w_down_pure = gen.randn((HIDDEN, FFN)).array * 0.02
    w_down_fuzzq = simulate_quantize(w_down_pure, "bfp8")
    w_down = torch.from_numpy(w_down_fuzzq)
    down = F.linear(act, w_down)
    outputs["ffn_down"] = down.detach().numpy().astype(np.float32)

    # LayerNorm 2
    output = F.layer_norm(down, normalized_shape=(HIDDEN,))
    outputs["layernorm_2"] = output.detach().numpy().astype(np.float32)

    return outputs


def main():
    print("=" * 75)
    print("  Compare Demo 02: Encoder - Pure vs FuzzQ")
    print(f"  模型: Encoder (batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, ffn={FFN})")
    print(f"  策略: StandardStrategy (L1: Exact+Bitwise, L2: Fuzzy+Sanity)")
    print("=" * 75)

    # 生成两组数据
    print("\n[1/3] 生成 Pure fp32 数据 (DataGenerator)")
    pure_outputs = generate_encoder_pure()
    print(f"  ✓ 生成完成，{len(pure_outputs)} 个算子")

    print("\n[2/3] 生成 FuzzQ 数据 (量化反量化 bfp8)")
    fuzzq_outputs = generate_encoder_fuzzq()
    print(f"  ✓ 生成完成，{len(fuzzq_outputs)} 个算子")

    # 比对
    print("\n[3/3] 使用 StandardStrategy 比对")
    engine = CompareEngine.standard(config=COMPARE_CFG)

    results = {}
    for op_name in ENCODER_OPS:
        if op_name not in pure_outputs or op_name not in fuzzq_outputs:
            print(f"  跳过 {op_name}: 数据缺失")
            continue

        golden = pure_outputs[op_name]
        dut = fuzzq_outputs[op_name]
        results[op_name] = engine.run(dut=dut, golden=golden)

    # 转换为 print_strategy_table 需要的格式
    results_list = [results[name] for name in ENCODER_OPS if name in results]
    names_list = [name for name in ENCODER_OPS if name in results]

    # 使用标准报告工具打印结果
    print_strategy_table(results_list, names_list)

    # 简单总结
    print("\n" + "=" * 75)
    print("  总结:")
    print("    - L1 (Exact): 预期失败，因为 bfp8 量化引入误差")
    print("    - L2 (Fuzzy): QSNR ~17dB，bfp8 低精度格式有明显精度损失")
    print("    - 展示了 FuzzQ (量化反量化) 数据生成策略的特点")
    print("    - bfp8: 块浮点格式，低精度但保持数值范围")
    print("=" * 75)

    # 断言验证
    print("\n[验证] 检查结果是否符合预期...")

    # 1. L1 (Exact) 应该全部失败（因为量化反量化有误差）
    exact_all_fail = all(
        not r["exact"].passed if r.get("exact") else True
        for r in results_list
    )
    assert exact_all_fail, "Exact 策略应该全部失败（量化反量化有误差）"
    print("  ✓ Exact 策略全部失败（符合预期）")

    # 2. Sanity 应该全部通过（无 NaN/Inf）
    sanity_all_pass = all(
        r["sanity"].valid if r.get("sanity") else True
        for r in results_list
    )
    assert sanity_all_pass, "Sanity 策略应该全部通过（无 NaN/Inf）"
    print("  ✓ Sanity 策略全部通过（无异常值）")

    # 3. QSNR 应该在合理范围内（bfp8 低精度格式，预期 >12 dB）
    qsnrs = [
        r["fuzzy_pure"].qsnr
        for r in results_list
        if r.get("fuzzy_pure") and r["fuzzy_pure"].qsnr != float("inf")
    ]
    if qsnrs:
        avg_qsnr = np.mean(qsnrs)
        assert avg_qsnr > 12, f"平均 QSNR 应该 >12 dB (bfp8 低精度)，实际 {avg_qsnr:.1f} dB"
        print(f"  ✓ 平均 QSNR {avg_qsnr:.1f} dB（bfp8 量化误差可见）")

    # 4. Cosine 相似度应该较高（bfp8 有量化误差，但整体方向一致）
    cosines = [
        r["fuzzy_pure"].cosine
        for r in results_list
        if r.get("fuzzy_pure")
    ]
    if cosines:
        avg_cosine = np.mean(cosines)
        assert avg_cosine > 0.97, f"平均 Cosine 应该 >0.97 (bfp8)，实际 {avg_cosine:.6f}"
        print(f"  ✓ 平均 Cosine {avg_cosine:.6f}（方向基本一致）")

    # 5. 不应该有 GOLDEN_SUSPECT（golden 是纯 fp32）
    bad_status = [
        name for name, r in zip(names_list, results_list)
        if r.get("status") in [CompareStatus.GOLDEN_SUSPECT, CompareStatus.BOTH_SUSPECT]
    ]
    assert len(bad_status) == 0, f"不应该怀疑 golden 数据: {bad_status}"
    print("  ✓ Golden 数据无异常")

    print("\n✓ 所有验证通过！")


if __name__ == "__main__":
    main()
