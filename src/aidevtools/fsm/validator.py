"""Golden 验证器 - 双 Golden 交叉验证

本模块实现双 Golden 交叉验证，确保 Golden 实现的正确性。

验证流程:
1. Golden_A vs Golden_B: 确认两个 Golden 实现一致
2. DUT vs Golden_A: 验证 DUT 功能正确性
3. DUT vs Golden_B: 交叉验证
4. 量化误差分析: 区分功能问题和量化问题

典型场景:
- Golden_A: 已有功能仿真器的 Golden（可信参考）
- Golden_B: 本框架的 C++ Golden 或 PyTorch Reference
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from aidevtools.core.log import logger
from aidevtools.tools.compare.diff import (
    DiffResult,
    FullCompareResult,
    CompareThresholds,
    compare_full,
    compare_exact,
    compare_3col,
    calc_qsnr,
)


class GoldenSource(Enum):
    """Golden 来源类型"""
    EXTERNAL_SIM = "external_sim"     # 已有功能仿真器
    CPP_GOLDEN = "cpp_golden"         # 本框架 C++ 实现
    TORCH_REF = "torch_ref"           # PyTorch Reference
    CUSTOM = "custom"                 # 自定义


class ValidationStatus(Enum):
    """验证状态"""
    GOLDEN_CONSISTENT = "golden_consistent"     # Golden 一致
    GOLDEN_MISMATCH = "golden_mismatch"         # Golden 不一致（需要排查）
    DUT_CORRECT = "dut_correct"                 # DUT 正确
    DUT_FUNCTIONAL_ERROR = "dut_functional_error"  # DUT 功能错误
    QUANTIZATION_ERROR = "quantization_error"   # 量化引入的误差
    UNKNOWN = "unknown"


@dataclass
class GoldenCompareResult:
    """Golden vs Golden 比对结果"""
    golden_a_source: GoldenSource
    golden_b_source: GoldenSource
    diff: DiffResult
    consistent: bool                  # Golden 是否一致

    @property
    def status_msg(self) -> str:
        if self.consistent:
            return f"Golden consistent (QSNR={self.diff.qsnr:.1f}dB)"
        return f"Golden MISMATCH (QSNR={self.diff.qsnr:.1f}dB)"


@dataclass
class ValidationResult:
    """完整验证结果"""
    op_name: str
    op_id: int

    # 三方数据
    dut_output: Optional[np.ndarray] = None
    golden_a: Optional[np.ndarray] = None
    golden_b: Optional[np.ndarray] = None
    golden_quantized: Optional[np.ndarray] = None

    # 比对结果
    golden_vs_golden: Optional[GoldenCompareResult] = None  # Golden_A vs Golden_B
    dut_vs_golden_a: Optional[DiffResult] = None            # DUT vs Golden_A
    dut_vs_golden_b: Optional[DiffResult] = None            # DUT vs Golden_B
    dut_vs_golden_qnt: Optional[DiffResult] = None          # DUT vs Quantized Golden

    # 状态
    status: ValidationStatus = ValidationStatus.UNKNOWN
    diagnosis: str = ""

    @property
    def passed(self) -> bool:
        """是否通过验证"""
        return self.status in (
            ValidationStatus.GOLDEN_CONSISTENT,
            ValidationStatus.DUT_CORRECT,
            ValidationStatus.QUANTIZATION_ERROR,  # 量化误差可能是可接受的
        )


@dataclass
class ValidatorConfig:
    """验证器配置"""
    # Golden 来源
    golden_a_source: GoldenSource = GoldenSource.EXTERNAL_SIM
    golden_b_source: GoldenSource = GoldenSource.CPP_GOLDEN

    # 比对阈值
    thresholds: CompareThresholds = field(default_factory=CompareThresholds)

    # Golden 一致性阈值
    golden_consistency_qsnr: float = 60.0   # Golden 要求 QSNR > 60dB
    golden_consistency_cosine: float = 0.9999  # 余弦相似度 > 0.9999

    # 量化相关
    quantize_dtype: str = "gfp16"
    compare_quantized: bool = True

    # 行为选项
    fail_on_golden_mismatch: bool = True    # Golden 不一致时是否报错
    save_mismatch_data: bool = True         # 保存不匹配的数据


class GoldenValidator:
    """Golden 验证器

    用于交叉验证不同 Golden 实现的一致性，以及验证 DUT 输出的正确性。

    核心功能:
    1. 验证两个 Golden 实现的一致性（确保 Golden 正确）
    2. 验证 DUT 输出与 Golden 的一致性
    3. 区分功能错误和量化误差
    4. 生成详细的诊断报告

    Example:
        validator = GoldenValidator(
            golden_a_fn=external_sim.run,  # 已有仿真器
            golden_b_fn=cpu_golden.run,    # 本框架
        )

        result = validator.validate(
            op_name="matmul",
            op_id=0,
            inputs={"a": a, "b": b},
            dut_output=dut_result,
        )

        if not result.passed:
            print(f"Validation failed: {result.diagnosis}")
    """

    def __init__(
        self,
        golden_a_fn: Optional[Callable] = None,
        golden_b_fn: Optional[Callable] = None,
        config: Optional[ValidatorConfig] = None,
    ):
        """
        Args:
            golden_a_fn: Golden A 计算函数 (inputs -> output)
            golden_b_fn: Golden B 计算函数 (inputs -> output)
            config: 验证器配置
        """
        self.config = config or ValidatorConfig()
        self._golden_a_fn = golden_a_fn
        self._golden_b_fn = golden_b_fn
        self._results: List[ValidationResult] = []

    def set_golden_a(
        self,
        fn: Callable,
        source: GoldenSource = GoldenSource.EXTERNAL_SIM,
    ):
        """设置 Golden A 计算函数"""
        self._golden_a_fn = fn
        self.config.golden_a_source = source

    def set_golden_b(
        self,
        fn: Callable,
        source: GoldenSource = GoldenSource.CPP_GOLDEN,
    ):
        """设置 Golden B 计算函数"""
        self._golden_b_fn = fn
        self.config.golden_b_source = source

    def validate(
        self,
        op_name: str,
        op_id: int,
        inputs: Dict[str, np.ndarray],
        dut_output: np.ndarray,
        golden_a: Optional[np.ndarray] = None,
        golden_b: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """执行验证

        Args:
            op_name: 算子名称
            op_id: 算子 ID
            inputs: 输入数据字典
            dut_output: DUT 输出
            golden_a: 预计算的 Golden A（如果有）
            golden_b: 预计算的 Golden B（如果有）

        Returns:
            ValidationResult 包含完整的验证结果和诊断
        """
        result = ValidationResult(op_name=op_name, op_id=op_id)
        result.dut_output = dut_output.copy() if self.config.save_mismatch_data else None

        # 1. 计算 Golden A
        if golden_a is not None:
            result.golden_a = golden_a
        elif self._golden_a_fn is not None:
            try:
                result.golden_a = self._golden_a_fn(**inputs)
            except Exception as e:
                logger.warning(f"Golden A computation failed: {e}")

        # 2. 计算 Golden B
        if golden_b is not None:
            result.golden_b = golden_b
        elif self._golden_b_fn is not None:
            try:
                result.golden_b = self._golden_b_fn(**inputs)
            except Exception as e:
                logger.warning(f"Golden B computation failed: {e}")

        # 3. 计算量化感知 Golden
        if self.config.compare_quantized:
            result.golden_quantized = self._compute_quantized_golden(op_name, inputs)

        # 4. 执行比对
        self._compare_all(result, dut_output)

        # 5. 诊断
        self._diagnose(result)

        self._results.append(result)
        return result

    def validate_golden_only(
        self,
        op_name: str,
        inputs: Dict[str, np.ndarray],
    ) -> GoldenCompareResult:
        """只验证两个 Golden 的一致性

        用于在没有 DUT 输出的情况下验证 Golden 实现。

        Args:
            op_name: 算子名称
            inputs: 输入数据

        Returns:
            GoldenCompareResult
        """
        golden_a, golden_b = None, None

        if self._golden_a_fn:
            try:
                golden_a = self._golden_a_fn(**inputs)
            except Exception as e:
                logger.error(f"Golden A failed: {e}")

        if self._golden_b_fn:
            try:
                golden_b = self._golden_b_fn(**inputs)
            except Exception as e:
                logger.error(f"Golden B failed: {e}")

        if golden_a is None or golden_b is None:
            return GoldenCompareResult(
                golden_a_source=self.config.golden_a_source,
                golden_b_source=self.config.golden_b_source,
                diff=DiffResult(
                    passed=False, max_abs=float('inf'), mean_abs=float('inf'),
                    max_rel=float('inf'), qsnr=0.0, cosine=0.0,
                    total_elements=0, exceed_count=0
                ),
                consistent=False,
            )

        diff = compare_full(golden_a, golden_b)
        consistent = (
            diff.qsnr >= self.config.golden_consistency_qsnr and
            diff.cosine >= self.config.golden_consistency_cosine
        )

        return GoldenCompareResult(
            golden_a_source=self.config.golden_a_source,
            golden_b_source=self.config.golden_b_source,
            diff=diff,
            consistent=consistent,
        )

    def _compute_quantized_golden(
        self,
        op_name: str,
        inputs: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """计算量化感知的 Golden"""
        from aidevtools.formats.quantize import simulate_quantize

        # 使用 Golden B 函数（通常是本框架的实现）
        golden_fn = self._golden_b_fn or self._golden_a_fn
        if golden_fn is None:
            return None

        try:
            # 对输入进行量化/反量化
            quantized_inputs = {}
            for name, data in inputs.items():
                quantized_inputs[name] = simulate_quantize(
                    data, self.config.quantize_dtype
                )

            return golden_fn(**quantized_inputs)
        except Exception as e:
            logger.warning(f"Quantized golden computation failed: {e}")
            return None

    def _compare_all(self, result: ValidationResult, dut_output: np.ndarray):
        """执行所有比对"""
        thresholds = self.config.thresholds

        # Golden A vs Golden B
        if result.golden_a is not None and result.golden_b is not None:
            diff = compare_full(result.golden_a, result.golden_b)
            consistent = (
                diff.qsnr >= self.config.golden_consistency_qsnr and
                diff.cosine >= self.config.golden_consistency_cosine
            )
            result.golden_vs_golden = GoldenCompareResult(
                golden_a_source=self.config.golden_a_source,
                golden_b_source=self.config.golden_b_source,
                diff=diff,
                consistent=consistent,
            )

        # DUT vs Golden A
        if result.golden_a is not None:
            result.dut_vs_golden_a = compare_full(
                result.golden_a, dut_output,
                atol=thresholds.fuzzy_atol,
                rtol=thresholds.fuzzy_rtol,
            )

        # DUT vs Golden B
        if result.golden_b is not None:
            result.dut_vs_golden_b = compare_full(
                result.golden_b, dut_output,
                atol=thresholds.fuzzy_atol,
                rtol=thresholds.fuzzy_rtol,
            )

        # DUT vs Quantized Golden
        if result.golden_quantized is not None:
            result.dut_vs_golden_qnt = compare_full(
                result.golden_quantized, dut_output,
                atol=thresholds.fuzzy_atol,
                rtol=thresholds.fuzzy_rtol,
            )

    def _diagnose(self, result: ValidationResult):
        """诊断验证结果"""
        # 检查 Golden 一致性
        if result.golden_vs_golden is not None:
            if not result.golden_vs_golden.consistent:
                result.status = ValidationStatus.GOLDEN_MISMATCH
                result.diagnosis = (
                    f"Golden implementations are inconsistent! "
                    f"QSNR={result.golden_vs_golden.diff.qsnr:.1f}dB < "
                    f"{self.config.golden_consistency_qsnr}dB. "
                    f"Please check Golden A ({self.config.golden_a_source.value}) "
                    f"and Golden B ({self.config.golden_b_source.value})."
                )
                if self.config.fail_on_golden_mismatch:
                    logger.error(result.diagnosis)
                return

        # 检查 DUT 正确性
        dut_vs_a = result.dut_vs_golden_a
        dut_vs_b = result.dut_vs_golden_b
        dut_vs_qnt = result.dut_vs_golden_qnt

        # 先用主要 Golden (A) 判断
        primary_result = dut_vs_a or dut_vs_b
        if primary_result is None:
            result.status = ValidationStatus.UNKNOWN
            result.diagnosis = "No Golden available for comparison"
            return

        if primary_result.passed:
            result.status = ValidationStatus.DUT_CORRECT
            result.diagnosis = (
                f"DUT output is correct. "
                f"QSNR={primary_result.qsnr:.1f}dB, "
                f"cosine={primary_result.cosine:.6f}"
            )
            return

        # DUT 与 fp32 Golden 不一致，检查是否是量化问题
        if dut_vs_qnt is not None and dut_vs_qnt.passed:
            result.status = ValidationStatus.QUANTIZATION_ERROR
            result.diagnosis = (
                f"DUT matches quantized Golden but not fp32 Golden. "
                f"This indicates quantization-induced error. "
                f"DUT vs fp32: QSNR={primary_result.qsnr:.1f}dB, "
                f"DUT vs quantized: QSNR={dut_vs_qnt.qsnr:.1f}dB"
            )
            return

        # 功能错误
        result.status = ValidationStatus.DUT_FUNCTIONAL_ERROR
        result.diagnosis = (
            f"DUT output is INCORRECT. "
            f"QSNR={primary_result.qsnr:.1f}dB (expected >{self.config.thresholds.fuzzy_min_qsnr}dB), "
            f"cosine={primary_result.cosine:.6f} (expected >{self.config.thresholds.fuzzy_min_cosine})"
        )
        logger.warning(f"[{result.op_name}_{result.op_id}] {result.diagnosis}")

    def get_results(self) -> List[ValidationResult]:
        """获取所有验证结果"""
        return self._results.copy()

    def clear(self):
        """清空结果"""
        self._results.clear()

    def summary(self) -> Dict[str, Any]:
        """生成汇总统计"""
        status_counts = {s: 0 for s in ValidationStatus}
        for r in self._results:
            status_counts[r.status] += 1

        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)

        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "by_status": {s.value: c for s, c in status_counts.items() if c > 0},
        }


# ============================================================
# 便捷函数
# ============================================================

def create_validator_with_external_golden(
    external_golden_fn: Callable,
    quantize_dtype: str = "gfp16",
) -> GoldenValidator:
    """创建使用外部 Golden 的验证器

    Args:
        external_golden_fn: 外部仿真器的 Golden 计算函数
        quantize_dtype: 量化数据类型

    Returns:
        配置好的 GoldenValidator
    """
    from aidevtools.ops.cpu_golden import run_cpu_golden

    def cpp_golden(**inputs):
        # 从 inputs 推断算子名称（需要在调用时传入）
        op_name = inputs.pop("_op_name", "unknown")
        return run_cpu_golden(op_name, inputs)

    config = ValidatorConfig(
        golden_a_source=GoldenSource.EXTERNAL_SIM,
        golden_b_source=GoldenSource.CPP_GOLDEN,
        quantize_dtype=quantize_dtype,
    )

    validator = GoldenValidator(
        golden_a_fn=external_golden_fn,
        golden_b_fn=cpp_golden,
        config=config,
    )

    return validator


def validate_golden_consistency(
    golden_a_fn: Callable,
    golden_b_fn: Callable,
    test_inputs: List[Dict[str, np.ndarray]],
    op_names: Optional[List[str]] = None,
) -> List[GoldenCompareResult]:
    """批量验证 Golden 一致性

    Args:
        golden_a_fn: Golden A 计算函数
        golden_b_fn: Golden B 计算函数
        test_inputs: 测试输入列表
        op_names: 对应的算子名称列表

    Returns:
        比对结果列表
    """
    validator = GoldenValidator(golden_a_fn, golden_b_fn)
    results = []

    for i, inputs in enumerate(test_inputs):
        op_name = op_names[i] if op_names else f"op_{i}"
        result = validator.validate_golden_only(op_name, inputs)
        results.append(result)

        if not result.consistent:
            logger.warning(
                f"Golden mismatch at {op_name}: {result.status_msg}"
            )

    return results
