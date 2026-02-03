"""算子 API

推荐用法 - 通过 PyTorch 劫持:
    import aidevtools.golden  # 导入即启用劫持

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden

工具函数:
    from aidevtools import ops

    ops.seed(42)       # 设置随机种子
    ops.clear()        # 清空记录
    ops.dump("./out")  # 导出数据

比对模式:
    from aidevtools.ops import CompareMode

    ops.set_compare_mode(CompareMode.SINGLE_OP)   # 每个算子都比对
    ops.set_compare_mode(CompareMode.FULL_GRAPH)  # 只比对最终输出
    ops.set_compare_mode(CompareMode.MIXED)       # 自动生成测试

TracedTensor:
    from aidevtools import ops

    x = ops.traced(input_data, "gfp16")
    y = F.matmul(x, w)  # 输出也是 TracedTensor
    result = y.numpy()  # 获取 numpy 数组

混合模式测试:
    ops.set_compare_mode(CompareMode.MIXED)
    ops.clear()

    x = ops.traced(input_data, "gfp16")
    y = F.matmul(x, w)
    y = F.gelu(y)

    test_cases = ops.generate_test_cases()
    results = ops.run_test_cases(test_cases)
"""
# 工具函数
from aidevtools.ops.auto import get_seed, seed
from aidevtools.ops.traced_tensor import (
    TracedTensor,
    ensure_traced,
    traced,
    wrap_traced_output,
)
from aidevtools.ops.base import (
    CompareMode,
    Op,
    OpNode,
    clear,
    compare_final,
    dump,
    get_compare_mode,
    get_compute_golden,
    get_golden_mode,
    get_graph,
    get_graph_ops,
    get_profile_enabled,
    get_profile_only,
    # Profile API
    get_profiles,
    get_records,
    has_golden_cpp,
    mark_compare_point,
    profile_only,
    register_golden_cpp,
    set_compare_mode,
    set_compute_golden,
    set_golden_mode,
    set_profile_enabled,
    set_profile_only,
)
from aidevtools.ops.test_generator import (
    TestCase,
    TestResult,
    generate_test_cases,
    print_test_summary,
    run_test_cases,
)

# 内部模块，仅用于触发算子注册，不对外暴露
from aidevtools.ops import _functional as _F  # noqa: F401

__all__ = [
    # 工具函数
    "seed",
    "get_seed",
    "clear",
    "dump",
    # TracedTensor
    "TracedTensor",
    "traced",
    "ensure_traced",
    "wrap_traced_output",
    # 比对模式
    "CompareMode",
    "set_compare_mode",
    "get_compare_mode",
    "mark_compare_point",
    "compare_final",
    # 计算图
    "OpNode",
    "get_graph",
    "get_graph_ops",
    # 测试生成
    "TestCase",
    "TestResult",
    "generate_test_cases",
    "run_test_cases",
    "print_test_summary",
    # Golden 配置
    "set_golden_mode",
    "get_golden_mode",
    "set_compute_golden",
    "get_compute_golden",
    # 记录
    "get_records",
    "get_profiles",
    "set_profile_enabled",
    "get_profile_enabled",
    "set_profile_only",
    "get_profile_only",
    "profile_only",
    # 高级
    "Op",
    "register_golden_cpp",
    "has_golden_cpp",
]
