#!/usr/bin/env python
"""
算子 API 使用示例

展示两种模式:
1. reference 模式: 使用内置 numpy 实现（模糊比对）
2. golden 模式: 使用用户注册的 Golden 实现（精确比对）
"""
import numpy as np

from aidevtools.ops import set_mode, get_mode, register_golden
from aidevtools.ops.nn import linear, relu, gelu, softmax, layernorm, attention
from aidevtools.trace import dump, gen_csv, clear


def demo_reference_mode():
    """模糊比对模式演示"""
    print("=" * 50)
    print("Reference 模式（模糊比对）")
    print("=" * 50)

    set_mode("reference")
    print(f"当前模式: {get_mode()}")

    # 使用内置 numpy 实现
    x = np.random.randn(2, 4, 64).astype(np.float32)
    w = np.random.randn(64, 128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)

    y = linear(x, w, b)
    print(f"linear: {x.shape} -> {y.shape}")

    y = relu(x)
    print(f"relu: {x.shape} -> {y.shape}")

    y = gelu(x)
    print(f"gelu: {x.shape} -> {y.shape}")

    y = softmax(x)
    print(f"softmax: {x.shape} -> {y.shape}, sum={y.sum(axis=-1)[0,0]:.4f}")

    gamma = np.ones(64, dtype=np.float32)
    beta = np.zeros(64, dtype=np.float32)
    y = layernorm(x, gamma, beta)
    print(f"layernorm: {x.shape} -> {y.shape}")


def demo_golden_mode():
    """精确比对模式演示"""
    print("\n" + "=" * 50)
    print("Golden 模式（精确比对）")
    print("=" * 50)

    # 先注册 Golden 实现
    # 实际使用时，这里调用 C++ binding
    @register_golden("linear")
    def golden_linear(x, weight, bias=None):
        """模拟 Golden 实现（实际应调用 C++ 库）"""
        print("  [调用 Golden linear]")
        y = np.matmul(x, weight)
        if bias is not None:
            y = y + bias
        return y

    @register_golden("relu")
    def golden_relu(x):
        """模拟 Golden 实现"""
        print("  [调用 Golden relu]")
        return np.maximum(0, x)

    # 切换到 golden 模式
    set_mode("golden")
    print(f"当前模式: {get_mode()}")

    x = np.random.randn(2, 4, 64).astype(np.float32)
    w = np.random.randn(64, 128).astype(np.float32)
    b = np.random.randn(128).astype(np.float32)

    y = linear(x, w, b)
    print(f"linear: {x.shape} -> {y.shape}")

    y = relu(x)
    print(f"relu: {x.shape} -> {y.shape}")

    # gelu 没有注册 Golden，会报错
    print("\n尝试调用未注册的 gelu:")
    try:
        y = gelu(x)
    except ValueError as e:
        print(f"  错误: {e}")


def demo_workflow():
    """完整工作流演示"""
    print("\n" + "=" * 50)
    print("完整工作流")
    print("=" * 50)

    clear()  # 清空之前的 trace 记录

    # 1. Reference 模式快速验证
    print("\n[1] Reference 模式快速验证流程")
    set_mode("reference")

    x = np.random.randn(2, 8, 64).astype(np.float32)
    w = np.random.randn(64, 64).astype(np.float32)

    y = linear(x, w)
    y = relu(y)
    y = softmax(y)
    print(f"    输出: {y.shape}")

    # 2. 导出
    print("\n[2] 导出数据")
    dump("./workspace", format="raw")
    csv_path = gen_csv("./workspace", model_name="demo")
    print(f"    CSV: {csv_path}")

    # 3. 切换到 Golden 模式精确验证
    print("\n[3] Golden 模式精确验证")
    print("    (需要先注册 Golden 实现，然后重新运行)")


def main():
    demo_reference_mode()
    demo_golden_mode()
    demo_workflow()

    print("\n" + "=" * 50)
    print("使用说明")
    print("=" * 50)
    print("""
1. Reference 模式（默认）:
   - 使用内置 numpy 实现
   - 适合快速验证流程、模糊比对

2. Golden 模式:
   - 需要用户注册 Golden 实现
   - 适合精确比对

切换模式:
    from aidevtools.ops import set_mode
    set_mode("reference")  # 或 "golden"

注册 Golden:
    from aidevtools.ops import register_golden

    @register_golden("linear")
    def my_golden_linear(x, weight, bias=None):
        return my_cpp_lib.linear(x, weight, bias)
""")


if __name__ == "__main__":
    main()
