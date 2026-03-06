"""
比对报告模块

包含文字报告、可视化报告、模型级可视化。
"""

from .text_report import (
    print_strategy_table,
    format_strategy_results,
    generate_strategy_json,
    print_joint_report,
    visualize_joint_report,
)
from .visualizer import Visualizer, ChartConfig
from .model_visualizer import (
    ModelVisualizer,
    ModelCompareResult,
    OpCompareResult,
    OpStatus,
)

__all__ = [
    # 文字报告
    "print_strategy_table",
    "format_strategy_results",
    "generate_strategy_json",
    "print_joint_report",
    "visualize_joint_report",
    # 可视化基础
    "Visualizer",
    "ChartConfig",
    # 模型级可视化
    "ModelVisualizer",
    "ModelCompareResult",
    "OpCompareResult",
    "OpStatus",
]
