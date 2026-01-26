"""Golden 模式 - 导入即启用

导入此模块后，torch.nn.functional 的调用自动走 CPU golden。

用法:
    import aidevtools.golden  # 导入即启用

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden

配置:
    import aidevtools.golden as golden

    golden.set_mode("cpp")      # 切换到 C++ golden
    golden.set_compare("fuzzy") # 启用模糊比对
    golden.set_quantize("gfp16") # 启用量化
    golden.set_profile(True)    # 启用 Paper Analysis
    golden.set_backward(True)   # 启用反向传播 golden

    golden.disable()  # 禁用
    golden.enable()   # 重新启用

    golden.report()   # 打印比对报告
    golden.profiles() # 获取 profiles
"""

from aidevtools.torch_backend import TorchGoldenBackend, TorchBackendConfig

# 全局后端实例
_backend = TorchGoldenBackend(TorchBackendConfig(
    golden_mode="python",
    verbose=False,
))

# 导入时自动启用
_backend.enable()


def enable():
    """启用 golden 模式"""
    _backend.enable()


def disable():
    """禁用 golden 模式"""
    _backend.disable()


def set_mode(mode: str):
    """设置 golden 模式: "cpp" | "python" | "none" """
    _backend.configure(golden_mode=mode)


def set_compare(mode: str):
    """设置比对模式: "exact" | "fuzzy" | "quantized" | "none" """
    _backend.configure(compare_mode=mode)


def set_quantize(qtype: str):
    """设置量化类型: "gfp16" | "gfp8" | "bfp16" | "bfp8" | None"""
    _backend.configure(quantize_type=qtype)


def set_profile(enabled: bool):
    """启用/禁用 Paper Analysis"""
    _backend.configure(profile_enabled=enabled)


def set_backward(enabled: bool):
    """启用/禁用反向传播 golden"""
    _backend.configure(backward_enabled=enabled)


def clear():
    """清空记录"""
    _backend.clear()


def report():
    """打印比对报告"""
    _backend.print_comparison_report()


def profiles():
    """获取 Paper Analysis profiles"""
    return _backend.get_profiles()


def results():
    """获取比对结果"""
    return _backend.get_compare_results()


def records():
    """获取执行记录"""
    return _backend.get_records()
