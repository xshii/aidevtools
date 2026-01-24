"""BFP 自定义格式模块

提供 Python Golden 和 C++ Golden 两种实现。

使用方式:
    # Python Golden
    from aidevtools.formats.custom.bfp import golden
    mantissas, exps, meta = golden.fp32_to_bfp(data)

    # C++ Golden (需要编译)
    from aidevtools.formats.custom.bfp import wrapper
    mantissas, exps = wrapper.fp32_to_bfp(data)
"""
from aidevtools.formats.custom.bfp import golden
from aidevtools.formats.custom.bfp.wrapper import is_cpp_available, register_bfp_golden

__all__ = ["golden", "register_bfp_golden", "is_cpp_available"]
