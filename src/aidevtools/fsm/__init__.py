"""功能仿真模型 (FSM) Golden Comparison 嵌入模块

本模块提供将 Golden Comparison (GC) 嵌入到功能仿真器的接口和实现。

核心组件：
- FSMGoldenHook: GC 嵌入的基类，定义回调接口
- StepHook: 单步比对 Hook（每条指令）
- SegmentHook: 片段级比对 Hook（每层/每算子）
- ModelHook: 全模型比对 Hook（端到端）
- GoldenValidator: 双 Golden 交叉验证器
- DUTAdapter: DUT 接口适配层

使用示例：
    from aidevtools.fsm import StepHook, SegmentHook, ModelHook

    # 单步比对
    hook = StepHook(sample_rate=0.1)  # 只比对 10% 的指令
    simulator.register_hook(hook)

    # 片段级比对
    hook = SegmentHook(compare_points=["matmul_0", "softmax_0"])
    simulator.register_hook(hook)

    # 全模型比对
    hook = ModelHook(golden_source="external")
    simulator.register_hook(hook)
"""

from aidevtools.fsm.hooks import (
    FSMGoldenHook,
    StepHook,
    SegmentHook,
    ModelHook,
    CompareGranularity,
)
from aidevtools.fsm.validator import GoldenValidator, ValidationResult
from aidevtools.fsm.dut_adapter import DUTAdapter, DUTConfig
from aidevtools.fsm.report import FSMCompareReport, generate_report

__all__ = [
    # Hooks
    "FSMGoldenHook",
    "StepHook",
    "SegmentHook",
    "ModelHook",
    "CompareGranularity",
    # Validator
    "GoldenValidator",
    "ValidationResult",
    # DUT Adapter
    "DUTAdapter",
    "DUTConfig",
    # Report
    "FSMCompareReport",
    "generate_report",
]
