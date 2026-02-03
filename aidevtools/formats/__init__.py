"""数据格式模块

导入顺序很重要，避免循环导入：
1. 先加载 base (定义 FormatBase 基类)
2. 再加载内置格式 (numpy_fmt, raw) 触发注册
3. 再加载 quantize (定义量化相关函数)
4. 最后加载自定义格式 (gfloat, bfp) 触发注册
"""
# 1. 加载基类和注册机制
from aidevtools.formats import base  # noqa: F401
from aidevtools.formats._registry import get, list_formats, register  # noqa: F401
from aidevtools.formats.base import FormatBase, load, save  # noqa: F401

# 2. 加载内置格式以触发注册
from aidevtools.formats import numpy_fmt  # noqa: F401
from aidevtools.formats import raw  # noqa: F401

# 3. 加载量化模块
from aidevtools.formats import quantize  # noqa: F401

# 4. 加载自定义格式以触发注册 (必须在 quantize 之后)
from aidevtools.formats.custom import gfloat  # noqa: F401
from aidevtools.formats.custom.bfp import golden as _bfp_golden  # noqa: F401
