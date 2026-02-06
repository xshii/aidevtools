"""
Tiling 策略模块

设计模式:
- Strategy 模式: 可插拔的 Tiling 策略
- Template Method 模式: 基类定义流程
"""

from .base import TileConfig, TilingStrategy
from .baseline import BaselineStrategy
from .efficiency_aware import EfficiencyAwareStrategy
from .fuse_speedup import FuseSpeedupStrategy

__all__ = [
    "TileConfig",
    "TilingStrategy",
    "BaselineStrategy",
    "EfficiencyAwareStrategy",
    "FuseSpeedupStrategy",
]
