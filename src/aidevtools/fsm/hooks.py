"""功能仿真 Golden Comparison Hook 实现

本模块实现三种粒度的 GC 嵌入：
1. StepHook: 单步比对（每条指令）
2. SegmentHook: 片段级比对（每层/每算子）
3. ModelHook: 全模型比对（端到端）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import numpy as np

from aidevtools.core.log import logger
from aidevtools.tools.compare.diff import (
    DiffResult,
    ExactResult,
    FullCompareResult,
    CompareThresholds,
    compare_full,
    compare_exact,
    compare_3col,
    calc_qsnr,
    calc_cosine,
)


class CompareGranularity(Enum):
    """比对粒度"""
    STEP = "step"           # 单步（每条指令）
    SEGMENT = "segment"     # 片段级（每层/每算子）
    MODEL = "model"         # 全模型（端到端）


@dataclass
class OpContext:
    """算子执行上下文"""
    op_name: str                          # 算子名称 (如 "matmul")
    op_id: int                            # 算子序号
    full_name: str                        # 完整名称 (如 "matmul_0")
    inputs: Dict[str, np.ndarray]         # 输入数据 {name: data}
    shapes: Dict[str, Tuple[int, ...]]    # 形状信息 {name: shape}
    dtype: str = "fp16"                   # 数据类型
    extra: Dict[str, Any] = field(default_factory=dict)  # 额外参数


@dataclass
class StepResult:
    """单步比对结果"""
    insn_id: int              # 指令序号
    op_name: str              # 算子名称
    passed: bool              # 是否通过
    diff: DiffResult          # 比对详情
    dut_output: Optional[np.ndarray] = None   # DUT 输出（可选保存）
    golden_output: Optional[np.ndarray] = None  # Golden 输出（可选保存）
    timestamp: float = 0.0    # 时间戳


@dataclass
class SegmentResult:
    """片段级比对结果"""
    segment_name: str         # 片段名称 (如 "layer_0", "matmul_0")
    op_count: int             # 包含的算子数量
    full_result: FullCompareResult  # 三列比对结果
    step_results: List[StepResult] = field(default_factory=list)  # 包含的单步结果
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class ModelResult:
    """全模型比对结果"""
    model_name: str                         # 模型名称
    e2e_diff: DiffResult                    # 端到端比对结果
    segment_results: List[SegmentResult]    # 各片段结果
    error_trace: List[Dict[str, Any]]       # 误差传播追踪
    total_ops: int                          # 总算子数
    passed_ops: int                         # 通过的算子数
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def pass_rate(self) -> float:
        """通过率"""
        return self.passed_ops / self.total_ops if self.total_ops > 0 else 0.0


class FSMGoldenHook(ABC):
    """功能仿真 Golden Comparison Hook 基类

    仿真器通过调用 Hook 的回调方法触发 GC 比对。

    使用方式：
        1. 继承此类实现具体的 Hook
        2. 将 Hook 注册到仿真器
        3. 仿真器在执行过程中调用 Hook 的回调方法

    Example:
        class MySimulator:
            def __init__(self):
                self.hooks = []

            def register_hook(self, hook: FSMGoldenHook):
                self.hooks.append(hook)

            def execute_op(self, op, inputs):
                output = self._compute(op, inputs)
                for hook in self.hooks:
                    hook.on_op_complete(op, inputs, output)
                return output
    """

    def __init__(
        self,
        thresholds: Optional[CompareThresholds] = None,
        enabled: bool = True,
        verbose: bool = False,
    ):
        """
        Args:
            thresholds: 比对阈值配置
            enabled: 是否启用
            verbose: 是否打印详细日志
        """
        self.thresholds = thresholds or CompareThresholds()
        self.enabled = enabled
        self.verbose = verbose
        self._results: List[Any] = []

    @property
    @abstractmethod
    def granularity(self) -> CompareGranularity:
        """返回比对粒度"""
        pass

    @abstractmethod
    def on_op_complete(
        self,
        ctx: OpContext,
        dut_output: np.ndarray,
    ) -> Optional[Any]:
        """算子执行完成回调

        Args:
            ctx: 算子上下文
            dut_output: DUT 计算结果

        Returns:
            比对结果（如果有）
        """
        pass

    def on_model_start(self, model_name: str, model_input: np.ndarray):
        """模型推理开始回调（可选实现）"""
        pass

    def on_model_complete(self, model_name: str, model_output: np.ndarray) -> Optional[Any]:
        """模型推理完成回调（可选实现）"""
        pass

    def get_results(self) -> List[Any]:
        """获取所有比对结果"""
        return self._results.copy()

    def clear(self):
        """清空结果"""
        self._results.clear()

    def _get_golden_fn(self, op_name: str) -> Optional[Callable]:
        """获取 Golden 计算函数

        优先级：
        1. CPU Golden (C++ 实现)
        2. Torch Reference
        """
        from aidevtools.ops.cpu_golden import run_cpu_golden

        # 使用 CPU Golden
        def golden_fn(**inputs):
            return run_cpu_golden(op_name, inputs)

        return golden_fn


class StepHook(FSMGoldenHook):
    """单步比对 Hook

    每条指令执行后进行比对，用于精确定位问题。

    Example:
        hook = StepHook(sample_rate=0.1)  # 采样 10%
        simulator.register_hook(hook)

        # 运行仿真
        simulator.run(model)

        # 获取结果
        for result in hook.get_results():
            if not result.passed:
                print(f"Mismatch at {result.op_name}: QSNR={result.diff.qsnr}")
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        save_data: bool = False,
        stop_on_fail: bool = False,
        **kwargs,
    ):
        """
        Args:
            sample_rate: 采样率 (0.0-1.0)，用于控制比对开销
            save_data: 是否保存 DUT/Golden 数据（调试用）
            stop_on_fail: 遇到不匹配是否停止
        """
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.save_data = save_data
        self.stop_on_fail = stop_on_fail
        self._insn_counter = 0
        self._rng = np.random.default_rng(42)

    @property
    def granularity(self) -> CompareGranularity:
        return CompareGranularity.STEP

    def on_op_complete(
        self,
        ctx: OpContext,
        dut_output: np.ndarray,
    ) -> Optional[StepResult]:
        if not self.enabled:
            return None

        self._insn_counter += 1

        # 采样控制
        if self.sample_rate < 1.0:
            if self._rng.random() > self.sample_rate:
                return None

        # 获取 Golden 计算函数
        golden_fn = self._get_golden_fn(ctx.op_name)
        if golden_fn is None:
            logger.warning(f"No golden function for {ctx.op_name}")
            return None

        # 计算 Golden
        try:
            golden_output = golden_fn(**ctx.inputs)
        except Exception as e:
            logger.error(f"Golden computation failed for {ctx.full_name}: {e}")
            return None

        # 比对
        diff = compare_full(
            golden_output,
            dut_output,
            atol=self.thresholds.fuzzy_atol,
            rtol=self.thresholds.fuzzy_rtol,
        )

        result = StepResult(
            insn_id=self._insn_counter,
            op_name=ctx.full_name,
            passed=diff.passed,
            diff=diff,
            dut_output=dut_output.copy() if self.save_data else None,
            golden_output=golden_output.copy() if self.save_data else None,
            timestamp=time.time(),
        )

        self._results.append(result)

        if self.verbose:
            status = "PASS" if diff.passed else "FAIL"
            logger.info(
                f"[Step {self._insn_counter}] {ctx.full_name}: {status} "
                f"(QSNR={diff.qsnr:.1f}dB, cos={diff.cosine:.6f})"
            )

        if not diff.passed and self.stop_on_fail:
            raise RuntimeError(f"Step comparison failed at {ctx.full_name}")

        return result

    def clear(self):
        super().clear()
        self._insn_counter = 0


class SegmentHook(FSMGoldenHook):
    """片段级比对 Hook

    在指定的比对点进行比对，平衡精度和性能。

    Example:
        hook = SegmentHook(
            compare_points=["matmul_0", "layernorm_0", "softmax_0"],
            compare_quantized=True,  # 同时比对量化 golden
        )
        simulator.register_hook(hook)
    """

    def __init__(
        self,
        compare_points: Optional[List[str]] = None,
        compare_quantized: bool = True,
        quantize_dtype: str = "gfp16",
        **kwargs,
    ):
        """
        Args:
            compare_points: 需要比对的算子名列表，None 表示全部比对
            compare_quantized: 是否进行量化感知比对
            quantize_dtype: 量化数据类型 (gfp16/gfp8/bfp16/bfp8)
        """
        super().__init__(**kwargs)
        self.compare_points = set(compare_points) if compare_points else None
        self.compare_quantized = compare_quantized
        self.quantize_dtype = quantize_dtype
        self._current_segment: Optional[str] = None
        self._segment_ops: List[StepResult] = []

    @property
    def granularity(self) -> CompareGranularity:
        return CompareGranularity.SEGMENT

    def _should_compare(self, full_name: str) -> bool:
        """判断是否需要比对"""
        if self.compare_points is None:
            return True
        return full_name in self.compare_points

    def _get_quantized_golden(
        self,
        op_name: str,
        inputs: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """获取量化感知的 Golden"""
        from aidevtools.formats.quantize import simulate_quantize

        # 对输入进行量化/反量化
        quantized_inputs = {}
        for name, data in inputs.items():
            quantized_inputs[name] = simulate_quantize(data, self.quantize_dtype)

        # 使用量化后的输入计算 Golden
        golden_fn = self._get_golden_fn(op_name)
        if golden_fn is None:
            return np.zeros_like(list(inputs.values())[0])

        return golden_fn(**quantized_inputs)

    def on_op_complete(
        self,
        ctx: OpContext,
        dut_output: np.ndarray,
    ) -> Optional[SegmentResult]:
        if not self.enabled:
            return None

        if not self._should_compare(ctx.full_name):
            return None

        # 获取 Golden
        golden_fn = self._get_golden_fn(ctx.op_name)
        if golden_fn is None:
            return None

        try:
            golden_pure = golden_fn(**ctx.inputs)
        except Exception as e:
            logger.error(f"Golden computation failed: {e}")
            return None

        # 获取量化感知 Golden
        if self.compare_quantized:
            golden_qnt = self._get_quantized_golden(ctx.op_name, ctx.inputs)
        else:
            golden_qnt = golden_pure

        # 三列比对
        full_result = compare_3col(
            op_name=ctx.op_name,
            op_id=ctx.op_id,
            result=dut_output,
            golden_pure=golden_pure,
            golden_qnt=golden_qnt,
            thresholds=self.thresholds,
        )

        segment_result = SegmentResult(
            segment_name=ctx.full_name,
            op_count=1,
            full_result=full_result,
            start_time=time.time(),
            end_time=time.time(),
        )

        self._results.append(segment_result)

        if self.verbose:
            logger.info(
                f"[Segment] {ctx.full_name}: {full_result.status} "
                f"(exact={full_result.exact.passed}, "
                f"fuzzy_pure={full_result.fuzzy_pure.passed}, "
                f"fuzzy_qnt={full_result.fuzzy_qnt.passed})"
            )

        return segment_result


class ModelHook(FSMGoldenHook):
    """全模型比对 Hook

    端到端验证，包含误差累积分析。

    Example:
        hook = ModelHook(
            golden_source="external",  # 使用外部仿真器的 golden
            track_error_propagation=True,
        )
        simulator.register_hook(hook)

        # 运行
        hook.on_model_start("bert", input_data)
        simulator.run(model, input_data)
        result = hook.on_model_complete("bert", output_data)

        # 分析误差传播
        for trace in result.error_trace:
            print(f"{trace['op']}: error={trace['error']:.4f}")
    """

    def __init__(
        self,
        golden_source: str = "cpp",
        external_golden_fn: Optional[Callable] = None,
        track_error_propagation: bool = True,
        error_sample_ops: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Args:
            golden_source: Golden 来源
                - "cpp": 使用本框架的 C++ Golden
                - "torch": 使用 PyTorch 计算
                - "external": 使用外部提供的 Golden
            external_golden_fn: 外部 Golden 计算函数 (当 golden_source="external" 时)
            track_error_propagation: 是否追踪误差传播
            error_sample_ops: 用于误差采样的算子列表
        """
        super().__init__(**kwargs)
        self.golden_source = golden_source
        self.external_golden_fn = external_golden_fn
        self.track_error_propagation = track_error_propagation
        self.error_sample_ops = set(error_sample_ops) if error_sample_ops else None

        # 运行时状态
        self._model_name: Optional[str] = None
        self._model_input: Optional[np.ndarray] = None
        self._layer_trace: List[Dict[str, Any]] = []
        self._op_results: List[Tuple[str, DiffResult]] = []
        self._start_time: float = 0.0

    @property
    def granularity(self) -> CompareGranularity:
        return CompareGranularity.MODEL

    def on_model_start(self, model_name: str, model_input: np.ndarray):
        """模型推理开始"""
        self._model_name = model_name
        self._model_input = model_input.copy()
        self._layer_trace.clear()
        self._op_results.clear()
        self._start_time = time.time()

        if self.verbose:
            logger.info(f"[Model] Start: {model_name}, input_shape={model_input.shape}")

    def on_op_complete(
        self,
        ctx: OpContext,
        dut_output: np.ndarray,
    ) -> Optional[DiffResult]:
        if not self.enabled:
            return None

        # 计算当前算子的 Golden 并比对
        golden_fn = self._get_golden_fn(ctx.op_name)
        if golden_fn is None:
            return None

        try:
            golden_output = golden_fn(**ctx.inputs)
        except Exception as e:
            logger.warning(f"Golden failed for {ctx.full_name}: {e}")
            return None

        diff = compare_full(golden_output, dut_output)
        self._op_results.append((ctx.full_name, diff))

        # 误差追踪
        if self.track_error_propagation:
            should_track = (
                self.error_sample_ops is None or
                ctx.full_name in self.error_sample_ops
            )
            if should_track:
                self._layer_trace.append({
                    "op": ctx.full_name,
                    "op_type": ctx.op_name,
                    "error": diff.max_abs,
                    "qsnr": diff.qsnr,
                    "cosine": diff.cosine,
                    "passed": diff.passed,
                })

        return diff

    def on_model_complete(
        self,
        model_name: str,
        model_output: np.ndarray,
    ) -> Optional[ModelResult]:
        """模型推理完成，返回完整结果"""
        if not self.enabled:
            return None

        end_time = time.time()

        # 计算端到端 Golden
        e2e_golden = self._compute_e2e_golden(model_output)
        if e2e_golden is None:
            e2e_golden = model_output  # fallback

        # 端到端比对
        e2e_diff = compare_full(e2e_golden, model_output)

        # 统计
        total_ops = len(self._op_results)
        passed_ops = sum(1 for _, diff in self._op_results if diff.passed)

        # 误差传播分析
        error_trace = self._analyze_error_propagation()

        result = ModelResult(
            model_name=model_name,
            e2e_diff=e2e_diff,
            segment_results=[],  # 可以从 _op_results 构建
            error_trace=error_trace,
            total_ops=total_ops,
            passed_ops=passed_ops,
            start_time=self._start_time,
            end_time=end_time,
        )

        self._results.append(result)

        if self.verbose:
            logger.info(
                f"[Model] Complete: {model_name}, "
                f"E2E QSNR={e2e_diff.qsnr:.1f}dB, "
                f"pass_rate={result.pass_rate:.1%} ({passed_ops}/{total_ops})"
            )

        return result

    def _compute_e2e_golden(self, model_output: np.ndarray) -> Optional[np.ndarray]:
        """计算端到端 Golden"""
        if self.golden_source == "external" and self.external_golden_fn:
            try:
                return self.external_golden_fn(self._model_input)
            except Exception as e:
                logger.error(f"External golden failed: {e}")
                return None

        # 对于 "cpp" 和 "torch"，端到端 golden 需要逐层累积
        # 这里返回 None，表示使用各层累积的结果
        return None

    def _analyze_error_propagation(self) -> List[Dict[str, Any]]:
        """分析误差传播"""
        if not self._layer_trace:
            return []

        analysis = []
        prev_error = 0.0

        for i, trace in enumerate(self._layer_trace):
            error_growth = trace["error"] - prev_error if i > 0 else trace["error"]

            analysis.append({
                **trace,
                "error_growth": error_growth,
                "cumulative_idx": i,
            })
            prev_error = trace["error"]

        return analysis

    def clear(self):
        super().clear()
        self._model_name = None
        self._model_input = None
        self._layer_trace.clear()
        self._op_results.clear()


# ============================================================
# 便捷函数
# ============================================================

def create_hook(
    granularity: Union[str, CompareGranularity],
    **kwargs,
) -> FSMGoldenHook:
    """创建 Hook 的工厂函数

    Args:
        granularity: 比对粒度 ("step", "segment", "model")
        **kwargs: 传递给具体 Hook 的参数

    Returns:
        对应粒度的 Hook 实例
    """
    if isinstance(granularity, str):
        granularity = CompareGranularity(granularity)

    hook_map = {
        CompareGranularity.STEP: StepHook,
        CompareGranularity.SEGMENT: SegmentHook,
        CompareGranularity.MODEL: ModelHook,
    }

    hook_cls = hook_map.get(granularity)
    if hook_cls is None:
        raise ValueError(f"Unknown granularity: {granularity}")

    return hook_cls(**kwargs)
