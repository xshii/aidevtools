"""
Demo: 基础用法

展示 Benchmark 定义、评估和策略比较
"""

from aidevtools.optimizer import (
    Benchmark,
    BenchmarkSuite,
    FusionEvaluator,
    get_fusion_rules,
)


def demo_benchmark_builder():
    """演示 Benchmark 链式构建"""
    print("=" * 60)
    print("1. Benchmark 链式构建")
    print("=" * 60)

    # 创建一个简单的 FFN benchmark
    bm = (
        Benchmark("my_ffn")
        .add_op("mm1", "matmul", M=512, N=3072, K=768)
        .add_op("gelu", "gelu", M=512, N=3072)
        .add_op("mm2", "matmul", M=512, N=768, K=3072)
    )

    print(f"Benchmark: {bm.name}")
    print(f"算子数量: {len(bm.ops)}")
    print("\n算子列表:")
    for op in bm.ops:
        print(f"  - {op.name}: {op.op_type.value}, shapes={op.shapes}")

    # 获取融合组
    fusion_groups = bm.get_fusion_groups()
    print(f"\n可融合组: {len(fusion_groups)} 个")
    for ops, pattern, speedup in fusion_groups:
        print(f"  - {ops}: pattern={pattern}, speedup={speedup:.2f}x")

    return bm


def demo_fusion_rules():
    """演示全局融合规则"""
    print("\n" + "=" * 60)
    print("2. 全局融合规则")
    print("=" * 60)

    rules = get_fusion_rules()
    print(rules.summary())

    # 查询特定规则
    rule = rules.get_rule("matmul", "gelu")
    if rule:
        print(f"\nmatmul + gelu 规则:")
        print(f"  - ratio: {rule.ratio}")
        print(f"  - speedup: {rule.fuse_speedup}x")


def demo_evaluator():
    """演示评估器"""
    print("\n" + "=" * 60)
    print("3. FusionEvaluator 评估")
    print("=" * 60)

    evaluator = FusionEvaluator()

    # 使用预定义 suite
    result = evaluator.evaluate_suite(
        "bert_ffn",
        seq_len=512,
        hidden=768,
        intermediate=3072,
    )

    print(result.summary())


def demo_strategy_compare():
    """演示策略比较"""
    print("\n" + "=" * 60)
    print("4. 策略比较")
    print("=" * 60)

    evaluator = FusionEvaluator()

    # 自定义 benchmark
    bm = (
        Benchmark("attention_proj")
        .add_op("q_proj", "matmul", M=512, N=768, K=768)
        .add_op("k_proj", "matmul", M=512, N=768, K=768)
        .add_op("v_proj", "matmul", M=512, N=768, K=768)
    )

    # 比较不同策略
    compare_result = evaluator.compare(
        bm,
        strategies=["baseline", "efficiency_aware", "fuse_speedup"],
    )

    print(compare_result.summary())


def demo_suite():
    """演示预定义 BenchmarkSuite"""
    print("\n" + "=" * 60)
    print("5. 预定义 BenchmarkSuite")
    print("=" * 60)

    suite = BenchmarkSuite()

    print("可用的预定义 benchmarks:")
    for name in suite.list_benchmarks()[:10]:  # 只显示前 10 个
        print(f"  - {name}")

    # 获取一个具体的 benchmark
    bm = suite.get("bert_ffn_512")
    if bm:
        print(f"\nbert_ffn_512 详情:")
        print(f"  - 算子数: {len(bm.ops)}")
        print(f"  - 类别: {bm.category}")


if __name__ == "__main__":
    demo_benchmark_builder()
    demo_fusion_rules()
    demo_evaluator()
    demo_strategy_compare()
    demo_suite()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
