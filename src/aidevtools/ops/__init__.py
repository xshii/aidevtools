"""算子 API

使用方法:
    from aidevtools.ops import set_golden_mode, register_golden_cpp, clear, dump
    from aidevtools.ops.nn import linear, relu, softmax

    # 1. 设置 golden 模式 (默认 "python"，可选 "cpp")
    set_golden_mode("python")

    # 2. 注册 C++ golden 实现（可选）
    @register_golden_cpp("linear")
    def my_linear(x, weight, bias=None):
        return my_cpp_lib.linear(x, weight, bias)

    # 3. 调用算子（自动执行 golden + reference + profile）
    clear()  # 清空之前的记录和 profiles
    y = linear(x, w, b)
    y = relu(y)
    y = softmax(y)

    # 4. 导出数据
    dump("./workspace")

    # 5. 获取 profiles 用于 Paper Analysis
    from aidevtools.analysis import PaperAnalyzer
    profiles = get_profiles()
    analyzer = PaperAnalyzer(chip="npu_910")
    analyzer.add_profiles(profiles)
    result = analyzer.analyze()

算子包含3种计算形式:
    - golden_cpp: C++ Golden 实现（通过 @register_golden_cpp 注册）
    - golden_python: Python Golden 实现（内置）
    - reference: 高精度参考实现（fp64，用于 fuzzy 比对）
"""
from aidevtools.ops.base import (
    Op,
    register_golden_cpp,
    has_golden_cpp,
    get_records,
    clear,
    dump,
    set_golden_mode,
    get_golden_mode,
    set_compute_golden,
    get_compute_golden,
    # Profile API (用于 Paper Analysis)
    get_profiles,
    set_profile_enabled,
    get_profile_enabled,
    set_profile_only,
    get_profile_only,
    profile_only,  # 上下文管理器
)
from aidevtools.ops import nn  # 触发算子实例化
