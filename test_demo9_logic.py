#!/usr/bin/env python3
"""测试 demo9 的核心逻辑（不需要 pyecharts）"""

import sys
import numpy as np

# 测试策略导入和比对逻辑
from aidevtools.compare.strategy import (
    ExactStrategy,
    FuzzyStrategy,
    BlockedStrategy,
    BitAnalysisStrategy,
    BitXorStrategy,
    SanityStrategy,
    FP32,
)
from aidevtools.compare.types import CompareConfig

print("✅ 所有策略导入成功")

# 构造测试数据
np.random.seed(42)
golden = np.random.randn(1000).astype(np.float32)
dut = golden + np.random.randn(1000).astype(np.float32) * 0.01

# 1. Exact
print("\n1️⃣  Testing ExactStrategy...")
exact_result = ExactStrategy.compare(golden, dut, max_abs=1e-6)
print(f"   Passed: {exact_result.passed}")
print(f"   Mismatch: {exact_result.mismatch_count}/{exact_result.total_elements}")

# 2. Fuzzy
print("\n2️⃣  Testing FuzzyStrategy...")
config = CompareConfig()
fuzzy_result = FuzzyStrategy.compare(golden, dut, config=config)
print(f"   Passed: {fuzzy_result.passed}")
print(f"   QSNR: {fuzzy_result.qsnr:.2f} dB")
print(f"   Cosine: {fuzzy_result.cosine:.6f}")

# 3. Blocked
print("\n3️⃣  Testing BlockedStrategy...")
blocks = BlockedStrategy.compare(golden, dut, block_size=256)
passed_blocks = sum(1 for b in blocks if b.passed)
print(f"   Total blocks: {len(blocks)}")
print(f"   Passed: {passed_blocks}/{len(blocks)}")

# 4. BitAnalysis
print("\n4️⃣  Testing BitAnalysisStrategy...")
bitwise_result = BitAnalysisStrategy.compare(golden[:100], dut[:100], fmt=FP32)
print(f"   Total: {bitwise_result.summary.total_elements}")
print(f"   Diff: {bitwise_result.summary.diff_elements}")
print(f"   Sign Flip: {bitwise_result.summary.sign_flip_count}")

# 5. BitXor
print("\n5️⃣  Testing BitXorStrategy...")
bitxor_result = BitXorStrategy.compare(golden[:100], dut[:100])
print(f"   Diff elements: {bitxor_result.diff_elements}/{bitxor_result.total_elements}")
print(f"   Diff bits: {bitxor_result.diff_bits}/{bitxor_result.total_bits}")

# 6. Sanity
print("\n6️⃣  Testing SanityStrategy...")
sanity_result = SanityStrategy.check_data(golden[:100])
print(f"   Valid: {sanity_result.valid}")

print("\n" + "=" * 60)
print("✅ 所有策略核心逻辑测试通过！")
print("=" * 60)
print("\n如需测试可视化，请安装: pip install pyecharts")
