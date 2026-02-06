"""
Demo: PyTorch 劫持 → Benchmark 桥接

展示如何从 PyTorch 代码自动提取 Benchmark，替代手动构建
"""

# 注意: 实际运行需要先 import aidevtools.golden 启用劫持
# 这里用模拟数据演示


def demo_concept():
    """演示概念 (伪代码)"""
    print("=" * 60)
    print("PyTorch 劫持 → Benchmark 桥接")
    print("=" * 60)

    print("""
之前 (手动构建):
    bm = (
        Benchmark("ffn")
        .add_op("mm1", "matmul", M=512, N=3072, K=768)
        .add_op("gelu", "gelu", M=512, N=3072)
        .add_op("mm2", "matmul", M=512, N=768, K=3072)
    )

现在 (PyTorch 劫持自动提取):
    import aidevtools.golden
    import torch.nn.functional as F

    y = F.linear(x, w1)      # 被劫持，记录到计算图
    y = F.gelu(y)            # 被劫持
    y = F.linear(y, w2)      # 被劫持

    bm = extract_benchmark("ffn")  # 自动从计算图提取
    """)


def demo_api():
    """演示 API"""
    print("\n" + "=" * 60)
    print("API 说明")
    print("=" * 60)

    print("""
1. extract_benchmark(name) -> Benchmark
   从当前计算图提取 Benchmark

2. extract_and_evaluate(name) -> EvalResult
   一键: 提取 + 评估

3. trace_and_extract(func, *args) -> TracedBenchmark
   执行函数并提取 Benchmark

使用示例:

    # 方式1: 先执行后提取
    import aidevtools.golden
    y = F.linear(x, w)
    y = F.gelu(y)
    bm = extract_benchmark("my_model")

    # 方式2: 一键提取+评估
    result = extract_and_evaluate("my_model")
    print(result.summary())

    # 方式3: 包装函数
    def my_ffn(x, w1, w2):
        y = F.linear(x, w1)
        y = F.gelu(y)
        y = F.linear(y, w2)
        return y

    traced = trace_and_extract(my_ffn, x, w1, w2)
    print(traced.summary())
    """)


def demo_workflow():
    """演示完整工作流"""
    print("\n" + "=" * 60)
    print("完整工作流")
    print("=" * 60)

    print("""
┌──────────────────────────────────────────────────────────┐
│  Step 1: 启用 PyTorch 劫持                               │
│                                                          │
│  import aidevtools.golden as golden                      │
│  golden.set_mode("python")                               │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 2: 执行 PyTorch 代码                               │
│                                                          │
│  x = torch.randn(512, 768)                               │
│  y = F.linear(x, w1)  # 劫持 → 记录 OpNode               │
│  y = F.gelu(y)        # 劫持 → 记录 OpNode               │
│  y = F.linear(y, w2)  # 劫持 → 记录 OpNode               │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 3: 自动提取 Benchmark                              │
│                                                          │
│  from aidevtools.optimizer import extract_benchmark      │
│  bm = extract_benchmark("my_ffn")                        │
│                                                          │
│  # bm.ops = [                                            │
│  #   OpSpec("linear_0", MATMUL, {M=512, N=3072, K=768}), │
│  #   OpSpec("gelu_0", GELU, {M=512, N=3072}),            │
│  #   OpSpec("linear_1", MATMUL, {M=512, N=768, K=3072}), │
│  # ]                                                     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Step 4: 时延评估                                        │
│                                                          │
│  evaluator = FusionEvaluator()                           │
│  result = evaluator.evaluate(bm)                         │
│  print(result.summary())                                 │
└──────────────────────────────────────────────────────────┘
    """)


def demo_comparison():
    """演示两种方式对比"""
    print("\n" + "=" * 60)
    print("两种前端方式对比")
    print("=" * 60)

    print("""
┌─────────────────────────┬─────────────────────────────────┐
│     手动构建 (旧)       │     PyTorch 劫持 (新)           │
├─────────────────────────┼─────────────────────────────────┤
│ bm = Benchmark("ffn")   │ import aidevtools.golden        │
│   .add_op("mm1",        │                                 │
│           "matmul",     │ y = F.linear(x, w1)             │
│           M=512,        │ y = F.gelu(y)                   │
│           N=3072,       │ y = F.linear(y, w2)             │
│           K=768)        │                                 │
│   .add_op("gelu",       │ bm = extract_benchmark("ffn")   │
│           "gelu",       │                                 │
│           M=512,        │                                 │
│           N=3072)       │                                 │
│   .add_op("mm2", ...)   │                                 │
├─────────────────────────┼─────────────────────────────────┤
│ 需要手动指定 shape      │ 自动从 tensor 推断 shape        │
│ 需要手动指定 op_type    │ 自动从函数名推断 op_type        │
│ 容易出错                │ 与实际执行一致                  │
└─────────────────────────┴─────────────────────────────────┘
    """)


if __name__ == "__main__":
    demo_concept()
    demo_api()
    demo_workflow()
    demo_comparison()

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)
