"""AI Dev Tools

推荐用法 - 通过 PyTorch 劫持:
    import aidevtools.golden  # 导入即启用劫持

    import torch.nn.functional as F
    y = F.linear(x, w)  # 自动走 golden

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

# 便捷导出工具函数
from aidevtools.ops import clear, dump, seed

__all__ = ["ops", "seed", "clear", "dump"]
