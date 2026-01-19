#!/usr/bin/env python
"""MiniTransformer Demo - 使用简化 ops API

演示：
1. 使用 ops API 定义算子序列（自动生成数据）
2. 执行 golden + reference
3. 构造假的 DUT 数据
4. 使用框架比对工具比对
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from aidevtools import ops
from aidevtools.ops.base import get_records
from aidevtools.tools.compare.diff import compare_3col, print_compare_table


def run_model():
    """运行模型，返回每层的 golden 和 reference"""
    ops.seed(42)
    ops.clear()

    # 定义模型: Linear -> LayerNorm -> Softmax
    y = ops.linear((2, 8, 64), 32, dtype="bfp8")
    y = ops.layernorm(y, 32, dtype="bfp8")
    y = ops.softmax(y, dtype="bfp8")

    return get_records()


def generate_fake_dut(golden: np.ndarray, noise_level: float = 0.001) -> np.ndarray:
    """生成假的 DUT 数据（在 golden 基础上加噪声）"""
    noise = np.random.randn(*golden.shape).astype(np.float32) * noise_level
    return golden + noise


def main():
    print(f"\n{'=' * 70}")
    print(f"  MiniTransformer Demo")
    print(f"{'=' * 70}")

    # 1. 运行模型
    print("\n[1] 运行模型 (Linear -> LayerNorm -> Softmax)")
    records = run_model()

    for r in records:
        print(f"    {r['name']}: input={r['input'].shape}, golden={r['golden'].shape}")

    # 2. 生成假的 DUT 数据
    print("\n[2] 生成假的 DUT 数据 (golden + noise)")
    np.random.seed(123)
    dut_outputs = []
    for r in records:
        dut = generate_fake_dut(r["golden"], noise_level=0.001)
        dut_outputs.append(dut)
        print(f"    {r['name']}: dut={dut.shape}")

    # 3. 使用框架比对工具
    print("\n[3] 比对 (使用框架 compare_3col)")
    results = []
    for i, r in enumerate(records):
        # golden_pure = reference (fp64), golden_qnt = golden (带量化)
        result = compare_3col(
            op_name=r["op"],
            op_id=i,
            result=dut_outputs[i],
            golden_pure=r["reference"],
            golden_qnt=r["golden"],
        )
        results.append(result)

    print_compare_table(results)

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
    print(f"    文件列表:")
    for f in sorted(output_dir.glob("*.bin")):
        print(f"      {f.name}")


if __name__ == "__main__":
    main()
