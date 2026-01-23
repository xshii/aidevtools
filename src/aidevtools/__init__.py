"""AI Dev Tools"""
__version__ = "0.1.0"

# 简化 API: from aidevtools import ops
from aidevtools import ops

# PyTorch 风格 API: from aidevtools import F
from aidevtools.ops import functional as F

__all__ = ["ops", "F"]
