"""算子 API

使用方法:
    from aidevtools.ops import register_golden, clear, dump, gen_csv
    from aidevtools.ops.nn import linear, relu, softmax

    # 1. 注册用户的 golden 实现（如 C++ binding）
    @register_golden("linear")
    def my_linear(x, weight, bias=None):
        return my_cpp_lib.linear(x, weight, bias)

    # 2. 调用算子（自动同时执行 reference 和 golden）
    clear()  # 清空之前的记录
    y = linear(x, w, b)
    y = relu(y)
    y = softmax(y)

    # 3. 导出数据和 CSV
    dump("./workspace")
    gen_csv("./workspace", model_name="my_model")
"""
from aidevtools.ops.base import (
    Op,
    register_golden,
    has_golden,
    get_records,
    clear,
    dump,
    gen_csv,
)
from aidevtools.ops import nn  # 触发算子实例化
