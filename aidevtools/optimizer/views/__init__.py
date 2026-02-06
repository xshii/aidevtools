"""
可视化视图模块

设计模式:
- Template Method 模式: 基类定义渲染流程
- Strategy 模式: 不同视图策略
"""

from .base import View, ViewResult
from .memory import MemoryFlowView
from .compute import ComputeView
from .roofline import RooflineView
from .bandwidth import BandwidthPipelineView
from .echarts import (
    ChartType,
    EChartsOption,
    EChartsConverter,
    to_echarts,
)

__all__ = [
    "View",
    "ViewResult",
    "MemoryFlowView",
    "ComputeView",
    "RooflineView",
    "BandwidthPipelineView",
    # ECharts
    "ChartType",
    "EChartsOption",
    "EChartsConverter",
    "to_echarts",
]
