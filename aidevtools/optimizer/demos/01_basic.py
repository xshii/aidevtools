"""
Demo: 基础用法

展示 PyTorch 劫持 → Benchmark 提取 → 时延评估

流程:
1. import aidevtools.golden 启用劫持
2. 执行 PyTorch 代码 (F.linear, F.gelu, ...)
3. extract_benchmark() 自动提取 Benchmark
4. FusionEvaluator 评估时延
"""

import torch
import torch.nn.functional as F

# 启用 PyTorch 劫持
import aidevtools.golden as golden
from aidevtools.optimizer import extract_benchmark, extract_and_evaluate, FusionEvaluator
from aidevtools.ops import clear as ops_clear


def demo_simple_linear():
    """演示简单 Linear"""
    print("=" * 60)
    print("1. 简单 Linear")
    print("=" * 60)

    ops_clear()  # 清空之前的计算图

    # PyTorch 代码 (自动被劫持)
    x = torch.randn(512, 768)
    w = torch.randn(768, 3072)
    y = F.linear(x, w)

    # 提取 Benchmark
    bm = extract_benchmark("simple_linear")

    print(f"Benchmark: {bm.name}")
    print(f"算子数: {len(bm.ops)}")
    for op in bm.ops:
        print(f"  - {op.name}: {op.op_type.value}, shapes={op.shapes}")


def demo_ffn():
    """演示 FFN (Linear → GELU → Linear)"""
    print("\n" + "=" * 60)
    print("2. FFN (Linear → GELU → Linear)")
    print("=" * 60)

    ops_clear()

    # FFN 结构
    x = torch.randn(512, 768)
    w1 = torch.randn(3072, 768)  # (out_features, in_features)
    w2 = torch.randn(768, 3072)

    y = F.linear(x, w1)   # (512, 768) @ (768, 3072) -> (512, 3072)
    y = F.gelu(y)         # (512, 3072)
    y = F.linear(y, w2)   # (512, 3072) @ (3072, 768) -> (512, 768)

    # 提取并评估
    bm = extract_benchmark("ffn")

    print(f"Benchmark: {bm.name}")
    print(f"算子数: {len(bm.ops)}")
    for op in bm.ops:
        print(f"  - {op.name}: {op.op_type.value}, shapes={op.shapes}")

    # 获取融合组
    fusion_groups = bm.get_fusion_groups()
    print(f"\n可融合组: {len(fusion_groups)} 个")
    for ops, pattern, speedup in fusion_groups:
        print(f"  - {ops}: pattern={pattern}, speedup={speedup:.2f}x")


def demo_evaluate():
    """演示时延评估"""
    print("\n" + "=" * 60)
    print("3. 时延评估")
    print("=" * 60)

    ops_clear()

    # 执行 PyTorch
    x = torch.randn(512, 768)
    w1 = torch.randn(3072, 768)
    w2 = torch.randn(768, 3072)

    y = F.linear(x, w1)
    y = F.gelu(y)
    y = F.linear(y, w2)

    # 一键提取 + 评估
    result = extract_and_evaluate("ffn_eval")

    print(result.summary())


def demo_strategy_compare():
    """演示策略比较"""
    print("\n" + "=" * 60)
    print("4. 策略比较")
    print("=" * 60)

    ops_clear()

    # Attention Projection
    x = torch.randn(512, 768)
    wq = torch.randn(768, 768)
    wk = torch.randn(768, 768)
    wv = torch.randn(768, 768)

    q = F.linear(x, wq)
    k = F.linear(x, wk)
    v = F.linear(x, wv)

    # 提取
    bm = extract_benchmark("attention_proj")

    # 比较不同策略
    evaluator = FusionEvaluator()
    compare_result = evaluator.compare(
        bm,
        strategies=["baseline", "efficiency_aware", "fuse_speedup"],
    )

    print(compare_result.summary())


def demo_nn_module():
    """演示使用 nn.Module"""
    print("\n" + "=" * 60)
    print("5. 使用 nn.Module")
    print("=" * 60)

    ops_clear()

    # 定义模型
    class FFN(torch.nn.Module):
        def __init__(self, hidden_size, intermediate_size):
            super().__init__()
            self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
            self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)

        def forward(self, x):
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            return x

    # 创建模型并执行
    model = FFN(hidden_size=768, intermediate_size=3072)
    x = torch.randn(512, 768)
    y = model(x)

    # 提取
    bm = extract_benchmark("nn_ffn")

    print(f"Benchmark: {bm.name}")
    print(f"算子数: {len(bm.ops)}")
    for op in bm.ops:
        print(f"  - {op.name}: {op.op_type.value}, shapes={op.shapes}")


if __name__ == "__main__":
    demo_simple_linear()
    demo_ffn()
    demo_evaluate()
    demo_strategy_compare()
    demo_nn_module()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
