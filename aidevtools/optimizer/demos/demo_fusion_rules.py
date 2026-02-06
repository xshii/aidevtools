"""
Demo: 融合规则配置

展示如何配置和使用全局融合规则
"""

from aidevtools.optimizer import (
    Benchmark,
    FusionRule,
    FusionPattern,
    FusionRules,
    FusionHyperParams,
    get_fusion_rules,
    can_fuse,
    get_fuse_ratio,
    get_fuse_speedup,
)


def demo_global_rules():
    """演示全局融合规则"""
    print("=" * 60)
    print("1. 全局融合规则")
    print("=" * 60)

    rules = get_fusion_rules()
    print(rules.summary())


def demo_add_custom_rule():
    """演示添加自定义规则"""
    print("\n" + "=" * 60)
    print("2. 添加自定义规则")
    print("=" * 60)

    rules = get_fusion_rules()

    # 添加一个自定义算子的融合规则
    custom_rule = FusionRule(
        op_type_a="custom_conv",
        op_type_b="batch_norm",
        ratio=0.85,          # 融合后访存比例
        fuse_speedup=1.4,    # 融合加速比
        description="Conv + BN 融合",
    )

    rules.add_rule(custom_rule)
    print(f"添加规则: {custom_rule.op_type_a} + {custom_rule.op_type_b}")

    # 验证
    print(f"\n查询 can_fuse('custom_conv', 'batch_norm'): {can_fuse('custom_conv', 'batch_norm')}")
    print(f"查询 get_fuse_speedup('custom_conv', 'batch_norm'): {get_fuse_speedup('custom_conv', 'batch_norm')}x")


def demo_fusion_patterns():
    """演示多算子融合模式"""
    print("\n" + "=" * 60)
    print("3. 多算子融合模式")
    print("=" * 60)

    rules = get_fusion_rules()

    # 查看已有的融合模式
    print("预定义融合模式:")
    for name, pattern in rules._patterns.items():
        print(f"  - {name}: {' -> '.join(pattern.op_sequence)}")
        print(f"    speedup: {pattern.speedup}x, priority: {pattern.priority}")

    # 添加自定义模式
    custom_pattern = FusionPattern(
        name="conv_bn_relu",
        op_sequence=["conv", "batch_norm", "relu"],
        speedup=1.8,
        priority=10,
        description="卷积 + BN + ReLU 三合一",
    )

    rules.add_pattern(custom_pattern)
    print(f"\n添加模式: {custom_pattern.name}")


def demo_auto_composition():
    """演示两两规则自动组合"""
    print("\n" + "=" * 60)
    print("4. 两两规则自动组合")
    print("=" * 60)

    rules = get_fusion_rules()

    # 创建一个 3 算子的 benchmark
    bm = (
        Benchmark("three_ops")
        .add_op("mm1", "matmul", M=512, N=768, K=768)
        .add_op("gelu", "gelu", M=512, N=768)
        .add_op("mm2", "matmul", M=512, N=768, K=768)
    )

    # 获取可融合组
    op_types = [op.op_type.value for op in bm.ops]
    shapes_list = [op.shapes for op in bm.ops]

    groups = rules.find_fusable_groups(op_types, shapes_list)

    print(f"算子序列: {op_types}")
    print(f"\n找到的融合组:")
    for indices, pattern_name, speedup in groups:
        print(f"  - 索引 {indices}: pattern={pattern_name}, speedup={speedup:.2f}x")


def demo_hyper_params():
    """演示超参数配置"""
    print("\n" + "=" * 60)
    print("5. 超参数配置")
    print("=" * 60)

    rules = get_fusion_rules()
    params = rules.hyper_params

    print("当前超参数:")
    print(f"\n组合加速比参数:")
    print(f"  - decay_base: {params.decay_base}")
    print(f"  - decay_rate: {params.decay_rate}")
    print(f"  - speedup_scale: {params.speedup_scale}")

    print(f"\n底噪开销参数:")
    print(f"  - op_submit_base: {params.op_submit_base} cycles")
    print(f"  - op_launch_latency: {params.op_launch_latency} cycles")
    print(f"  - dma_submit_base: {params.dma_submit_base} cycles")

    print(f"\n融合节省参数:")
    print(f"  - fuse_submit_save: {params.fuse_submit_save * 100:.0f}%")
    print(f"  - fuse_dma_save: {params.fuse_dma_save * 100:.0f}%")

    # 转换为向量 (用于 ML)
    vec = params.to_vector()
    print(f"\n参数向量维度: {len(vec)}")
    print(f"参数边界数: {len(params.param_bounds())}")


def demo_override_pair():
    """演示 Benchmark 级别的规则覆盖"""
    print("\n" + "=" * 60)
    print("6. Benchmark 级别规则覆盖")
    print("=" * 60)

    # 创建 benchmark 并覆盖特定算子对的规则
    bm = (
        Benchmark("custom_override")
        .add_op("mm1", "matmul", M=512, N=768, K=768)
        .add_op("act", "gelu", M=512, N=768)
        # 覆盖 mm1 + act 的融合规则
        .override_pair("mm1", "act", ratio=0.88, fuse_speedup=1.5)
    )

    print(f"Benchmark: {bm.name}")
    print(f"覆盖规则: {bm._pair_overrides}")

    # 获取融合组时会使用覆盖的规则
    groups = bm.get_fusion_groups()
    for ops, pattern, speedup in groups:
        print(f"融合组: {ops}, speedup={speedup:.2f}x")


if __name__ == "__main__":
    demo_global_rules()
    demo_add_custom_rule()
    demo_fusion_patterns()
    demo_auto_composition()
    demo_hyper_params()
    demo_override_pair()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
