"""GFloat 自定义格式模块

提供 Python Golden 和 C++ Golden 实现。

使用方式:
    from aidevtools.formats.custom.gfloat import golden
    from aidevtools.formats.custom.gfloat import register_gfloat_golden

    # 注册后使用
    register_gfloat_golden()
"""
from aidevtools.formats.custom.gfloat import golden
from aidevtools.formats.custom.gfloat.wrapper import is_cpp_available, register_gfloat_golden

__all__ = ["golden", "register_gfloat_golden", "is_cpp_available"]
