#!/usr/bin/env python
"""
算子 API 使用示例

调用算子时，自动同时执行:
1. reference 实现 (numpy) -> 保存为 golden（参考标准）
2. golden 实现 (用户注册的 C++ binding) -> 保存为 result（待验证）
"""
import numpy as np

from aidevtools.ops import register_golden, clear, dump, gen_csv
from aidevtools.ops.nn import linear, relu, softmax


def demo_without_golden():
    """未注册 golden 实现时"""
    print("=" * 50)
    print("场景 1: 未注册 Golden 实现")
    print("=" * 50)

    clear()

    x = np.random.randn(2, 8, 64).astype(np.float32)
    w = np.random.randn(64, 64).astype(np.float32)

    y = linear(x, w)
    y = relu(y)
    y = softmax(y)

    print(f"  linear -> relu -> softmax")
    print(f"  输入: {x.shape}, 输出: {y.shape}")
    print(f"  softmax sum (should be 1.0): {y.sum(axis=-1)[0, 0]:.4f}")

    # 导出
    dump("./workspace_no_golden", format="raw")
    csv_path = gen_csv("./workspace_no_golden", model_name="no_golden")
    print(f"  CSV: {csv_path}")
    print("  注意: result_bin 列为空，需要后续填充模拟器输出")


def demo_with_golden():
    """注册 golden 实现后"""
    print("\n" + "=" * 50)
    print("场景 2: 注册 Golden 实现")
    print("=" * 50)

    # 注册 golden 实现（模拟 C++ binding）
    @register_golden("linear")
    def golden_linear(x, weight, bias=None):
        """模拟的 Golden 实现（实际应调用 C++ 库）"""
        # 这里故意加一点噪声，模拟硬件误差
        y = np.matmul(x, weight)
        if bias is not None:
            y = y + bias
        y = y + np.random.randn(*y.shape).astype(np.float32) * 1e-6
        return y

    @register_golden("relu")
    def golden_relu(x):
        return np.maximum(0, x)

    @register_golden("softmax")
    def golden_softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    clear()

    x = np.random.randn(2, 8, 64).astype(np.float32)
    w = np.random.randn(64, 64).astype(np.float32)

    y = linear(x, w)
    y = relu(y)
    y = softmax(y)

    print(f"  linear -> relu -> softmax")
    print(f"  输入: {x.shape}, 输出: {y.shape}")

    # 导出
    dump("./workspace_with_golden", format="raw")
    csv_path = gen_csv("./workspace_with_golden", model_name="with_golden")
    print(f"  CSV: {csv_path}")
    print("  注意: result_bin 列已填充，可直接比对")


def main():
    demo_without_golden()
    demo_with_golden()

    print("\n" + "=" * 50)
    print("工作流说明")
    print("=" * 50)
    print("""
1. 未注册 Golden:
   - 只执行 reference (numpy) 实现
   - 导出 golden.bin（标准答案）、input.bin、weight.bin
   - result_bin 留空，等待模拟器运行后填充

2. 已注册 Golden:
   - 同时执行 reference 和 golden 两种实现
   - 导出 golden.bin（numpy）、result.bin（用户实现）
   - 可直接用 compare 工具比对两者差异

使用方法:
    from aidevtools.ops import register_golden, clear, dump, gen_csv
    from aidevtools.ops.nn import linear, relu, softmax

    # 1. 注册 Golden 实现
    @register_golden("linear")
    def my_linear(x, weight, bias=None):
        return my_cpp_lib.linear(x, weight, bias)

    # 2. 执行算子流程
    clear()
    y = linear(x, w, b)
    y = relu(y)

    # 3. 导出
    dump("./output")
    gen_csv("./output", model_name="my_model")
""")


if __name__ == "__main__":
    main()
