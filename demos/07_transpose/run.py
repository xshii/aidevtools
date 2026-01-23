#!/usr/bin/env python
"""Transpose Demo - linear + transpose

演示 PyTorch 风格 F API:
1. F.linear (matmul + bias)
2. Transpose (4维矩阵转置)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import F, ops
from aidevtools.ops import get_records
from aidevtools.tools.compare.diff import compare_3col, print_compare_table
from aidevtools.formats.quantize import generate_fake_dut


def run_model():
    """使用 F API 运行 linear + transpose"""
    ops.seed(42)
    ops.clear()
    np.random.seed(42)

    # Input: (batch, heads, seq, d_k) = (2, 4, 8, 64)
    x = np.random.randn(2, 4, 8, 64).astype(np.float32)

    # Linear: weight [out_features, in_features] = [32, 64]
    w = np.random.randn(32, 64).astype(np.float32) * 0.02
    b = np.random.randn(32).astype(np.float32) * 0.01

    # Linear: (2, 4, 8, 64) @ (64, 32) -> (2, 4, 8, 32)
    y = F.linear(x, w, b)
    print(f"  Linear: {x.shape} -> {y.shape}")

    # Transpose: (2, 4, 8, 32) -> (2, 4, 32, 8)
    y = np.transpose(y, axes=(0, 1, 3, 2))
    print(f"  Transpose: -> {y.shape}")

    return get_records()


def main():
    print(f"\n{'=' * 80}")
    print(f"  Transpose Demo - Linear + Transpose (4D)")
    print(f"  PyTorch 风格 F API")
    print(f"{'=' * 80}")

    # 运行模型
    print(f"\n[1] 运行模型 (Linear -> Transpose)")
    records = run_model()

    for r in records:
        print(f"    {r['name']}: input={r['input'].shape}, output={r['golden'].shape}")

    # 生成假 DUT
    print(f"\n[2] 生成假的 DUT 数据")
    np.random.seed(123)
    dut_outputs = [generate_fake_dut(r["reference"], qtype="bfp8", noise_level=0.0005) for r in records]

    for i, r in enumerate(records):
        print(f"    {r['name']}: dut={dut_outputs[i].shape}")

    # 比对
    print(f"\n[3] 比对结果")
    results = []
    for i, r in enumerate(records):
        result = compare_3col(
            op_name=r["op"],
            op_id=i,
            result=dut_outputs[i],
            golden_pure=r["reference"],
            golden_qnt=r["golden"],
        )
        results.append(result)

    print_compare_table(results)

    # 导出
    output_dir = Path(__file__).parent / "workspace"
    ops.dump(str(output_dir))
    print(f"    导出到: {output_dir}")


if __name__ == "__main__":
    main()
