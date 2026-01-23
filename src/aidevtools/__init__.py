"""AI Dev Tools

PyTorch 风格的算子 API:
    from aidevtools import F

    y = F.linear(x, weight, bias)
    y = F.relu(y)
    y = F.softmax(y, dim=-1)

工具函数:
    from aidevtools import ops

    ops.seed(42)
    ops.clear()
    # ... 执行算子 ...
    ops.dump("./workspace")
"""
__version__ = "0.1.0"

# 模块级导入
from aidevtools import ops

# PyTorch 风格 API: from aidevtools import F
from aidevtools.ops import functional as F

# 便捷导出工具函数
from aidevtools.ops import seed, clear, dump

__all__ = ["ops", "F", "seed", "clear", "dump"]
