#!/usr/bin/env python
"""MiniTransformer Demo - 完整比对流程演示

演示完整的 golden 生成与比对流程：
1. 使用 PyTorch 风格 F API 定义算子序列
2. 执行 cpu_golden (via subprocess)
3. 构造假的 DUT 数据（模拟 bfp 格式处理）
4. 比对 golden 与 DUT
"""
import numpy as np
from pathlib import Path

from aidevtools import ops
from aidevtools.ops import _functional as F
from aidevtools.ops import get_records
from aidevtools.tools.compare.diff import compare_full
from aidevtools.formats.quantize import generate_fake_dut

# 设置 cpu golden dtype 并使用 cpp golden
from aidevtools.ops.cpu_golden import set_cpu_golden_dtype
set_cpu_golden_dtype("gfp16")
ops.set_golden_mode("cpp")


def run_model():
    """运行模型，返回每层的 golden"""
    ops.seed(42)
    ops.clear()

    # 配置
    batch, seq, hidden, out_hidden = 2, 8, 64, 32
    np.random.seed(42)

    # MatMul: (2, 8, 64) @ (64, 32) -> (2, 8, 32)
    x = np.random.randn(batch, seq, hidden).astype(np.float32)
    w = np.random.randn(hidden, out_hidden).astype(np.float32)
    y = F.matmul(x, w)
    print(f"  MatMul: {x.shape} @ {w.shape} -> {y.shape}")

    # LayerNorm
    y = F.layer_norm(y, normalized_shape=(out_hidden,))
    print(f"  LayerNorm: {y.shape}")

    # Softmax
    y = F.softmax(y, dim=-1)
    print(f"  Softmax: {y.shape}")

    return get_records()


def main():
    print(f"\n{'=' * 70}")
    print("  MiniTransformer Demo - PyTorch 风格 API")
    print(f"{'=' * 70}")
    print("  golden_mode: cpp (via subprocess)")
    print("  quantization: gfp16 (cpp)")

    # 1. 运行模型
    print("\n[1] 运行模型 (MatMul -> LayerNorm -> Softmax)")
    print("    框架执行 cpu_golden via subprocess")
    records = run_model()

    for r in records:
        print(f"    {r['name']}: input={r['input'].shape}, golden={r['golden'].shape}")

    # 2. 生成假的 DUT 数据
    print("\n[2] 生成假的 DUT 数据 (模拟 bfp8 格式处理)")
    print("    流程: golden -> bfp8 量化/反量化 -> 加小噪声")
    np.random.seed(123)
    dut_outputs = []
    for r in records:
        # 使用 golden 作为基准生成假 DUT
        dut = generate_fake_dut(r["golden"], qtype="bfp8", noise_level=0.001)
        dut_outputs.append(dut)
        print(f"    {r['name']}: dut={dut.shape}")

    # 3. 使用框架比对工具
    print("\n[3] 比对 (golden vs DUT)")
    print(f"    {'Op':15} {'Max Abs':>12} {'QSNR':>10} {'Cosine':>10} {'Status':>8}")
    print("    " + "-" * 60)
    for i, r in enumerate(records):
        diff = compare_full(r["golden"], dut_outputs[i])
        status = "PASS" if diff.passed else "FAIL"
        print(f"    {r['name']:15} {diff.max_abs:12.6e} {diff.qsnr:10.2f} {diff.cosine:10.6f} {status:>8}")

    # 4. 导出 bin
    print("[4] 导出 bin 文件")
    output_dir = Path(__file__).parent / "workspace"
    ops.dump(str(output_dir))

    # 同时导出 dut
    from aidevtools.formats.base import save as save_data
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        save_data(str(output_dir / f"{r['name']}_dut.bin"), dut_outputs[i])

    print(f"\n    输出目录: {output_dir}")
    print("    文件列表:")
    for f in sorted(output_dir.glob("*.bin")):
        print(f"      {f.name}")


if __name__ == "__main__":
    main()
