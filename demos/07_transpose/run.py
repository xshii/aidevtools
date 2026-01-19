#!/usr/bin/env python
"""Transpose Demo - matmul+bias 后 transpose

演示：
1. Linear (matmul + bias)
2. Transpose (4维矩阵转置)
3. 同时测试 bfp4/bfp8/bfp16 三种量化类型
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import ops
from aidevtools.ops.base import get_records
from aidevtools.tools.compare.diff import compare_3col, print_compare_table


def run_with_dtype(dtype: str):
    """使用指定 dtype 运行 linear + transpose"""
    ops.seed(42)
    ops.clear()

    # Linear: (batch, heads, seq, d_k) -> (batch, heads, seq, d_v)
    # 输入: (2, 4, 8, 64), 输出: (2, 4, 8, 32)
    y = ops.linear((2, 4, 8, 64), 32, dtype=dtype)

    # Transpose: (2, 4, 8, 32) -> (2, 4, 32, 8)
    # 交换最后两个维度
    y = ops.transpose(y, axes=(0, 1, 3, 2), dtype=dtype)

    return get_records()


def generate_fake_dut(golden: np.ndarray, noise_level: float = 0.0005) -> np.ndarray:
    """生成假的 DUT 数据"""
    noise = np.random.randn(*golden.shape).astype(np.float32) * noise_level
    return golden + noise


def main():
    print(f"\n{'=' * 80}")
    print(f"  Transpose Demo - Linear + Transpose (4D)")
    print(f"{'=' * 80}")

    # 测试三种 dtype
    dtypes = ["bfp4", "bfp8", "bfp16"]

    for dtype in dtypes:
        print(f"\n{'=' * 80}")
        print(f"  dtype = {dtype}")
        print(f"{'=' * 80}")

        # 运行模型
        print(f"\n[1] 运行模型 (Linear -> Transpose)")
        records = run_with_dtype(dtype)

        for r in records:
            print(f"    {r['name']}: input={r['input'].shape}, output={r['golden'].shape}")

        # 生成假 DUT
        print(f"\n[2] 生成假的 DUT 数据")
        np.random.seed(123)
        dut_outputs = [generate_fake_dut(r["golden"]) for r in records]

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
        output_dir = Path(__file__).parent / "workspace" / dtype
        ops.dump(str(output_dir))
        print(f"    导出到: {output_dir}")


if __name__ == "__main__":
    main()
