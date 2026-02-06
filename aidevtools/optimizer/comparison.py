"""
理论分析 vs 工程化方法对比

两种预测方法:
1. 理论分析: 基于 ChipSpec + OpProfile + CostParameters
2. 工程化: 基于 FusionHyperParams (ML 校准)

对比维度:
- 预测精度 (vs 实测)
- 误差分布
- 各场景表现
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import json

import numpy as np


class PredictMethod(Enum):
    """预测方法"""
    THEORETICAL = "theoretical"  # 理论分析
    EMPIRICAL = "empirical"      # 工程化 (ML)
    HYBRID = "hybrid"            # 混合


@dataclass
class PredictionResult:
    """单次预测结果"""
    benchmark_name: str
    method: PredictMethod
    predicted_latency_us: float
    actual_latency_us: Optional[float] = None

    @property
    def error(self) -> Optional[float]:
        """绝对误差"""
        if self.actual_latency_us is None or self.actual_latency_us <= 0:
            return None
        return abs(self.predicted_latency_us - self.actual_latency_us)

    @property
    def error_pct(self) -> Optional[float]:
        """百分比误差"""
        if self.actual_latency_us is None or self.actual_latency_us <= 0:
            return None
        return self.error / self.actual_latency_us * 100


@dataclass
class ComparisonMetrics:
    """对比指标"""
    method: PredictMethod
    sample_count: int

    # 误差指标
    mae: float = 0.0          # Mean Absolute Error (us)
    mape: float = 0.0         # Mean Absolute Percentage Error (%)
    rmse: float = 0.0         # Root Mean Square Error (us)
    max_error: float = 0.0    # 最大误差 (us)
    max_error_pct: float = 0.0  # 最大百分比误差

    # 相关性
    r_squared: float = 0.0    # R² 决定系数

    # 分布
    error_std: float = 0.0    # 误差标准差
    error_percentiles: Dict[int, float] = field(default_factory=dict)  # P50, P90, P99

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "sample_count": self.sample_count,
            "mae_us": self.mae,
            "mape_pct": self.mape,
            "rmse_us": self.rmse,
            "max_error_us": self.max_error,
            "max_error_pct": self.max_error_pct,
            "r_squared": self.r_squared,
            "error_std": self.error_std,
            "error_percentiles": self.error_percentiles,
        }


@dataclass
class MethodComparison:
    """方法对比结果"""
    theoretical: ComparisonMetrics
    empirical: ComparisonMetrics

    # 各 benchmark 的详细结果
    details: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """生成对比摘要"""
        lines = [
            "=" * 70,
            "理论分析 vs 工程化方法对比",
            "=" * 70,
            "",
            f"{'指标':<20} {'理论分析':>15} {'工程化(ML)':>15} {'差异':>15}",
            "-" * 70,
        ]

        t, e = self.theoretical, self.empirical

        def fmt_diff(t_val, e_val, lower_better=True):
            diff = e_val - t_val
            if lower_better:
                winner = "← 更优" if diff > 0 else "→ 更优" if diff < 0 else "相同"
            else:
                winner = "→ 更优" if diff > 0 else "← 更优" if diff < 0 else "相同"
            return f"{diff:+.2f} {winner}"

        lines.append(f"{'样本数':<20} {t.sample_count:>15} {e.sample_count:>15}")
        lines.append(f"{'MAE (us)':<20} {t.mae:>15.2f} {e.mae:>15.2f} {fmt_diff(t.mae, e.mae)}")
        lines.append(f"{'MAPE (%)':<20} {t.mape:>15.2f} {e.mape:>15.2f} {fmt_diff(t.mape, e.mape)}")
        lines.append(f"{'RMSE (us)':<20} {t.rmse:>15.2f} {e.rmse:>15.2f} {fmt_diff(t.rmse, e.rmse)}")
        lines.append(f"{'R²':<20} {t.r_squared:>15.4f} {e.r_squared:>15.4f} {fmt_diff(t.r_squared, e.r_squared, False)}")
        lines.append(f"{'最大误差 (%)':<20} {t.max_error_pct:>15.2f} {e.max_error_pct:>15.2f} {fmt_diff(t.max_error_pct, e.max_error_pct)}")

        lines.append("")
        lines.append("误差百分位:")
        for p in [50, 90, 99]:
            t_p = t.error_percentiles.get(p, 0)
            e_p = e.error_percentiles.get(p, 0)
            lines.append(f"  P{p:<18} {t_p:>15.2f} {e_p:>15.2f} {fmt_diff(t_p, e_p)}")

        lines.append("")
        lines.append("=" * 70)

        # 结论
        t_score = t.mape + t.rmse / 10
        e_score = e.mape + e.rmse / 10

        if e_score < t_score * 0.9:
            lines.append("结论: 工程化方法明显更优 (建议使用 ML 校准后的参数)")
        elif t_score < e_score * 0.9:
            lines.append("结论: 理论分析更优 (可能需要更多校准数据)")
        else:
            lines.append("结论: 两种方法表现接近 (可根据场景选择)")

        return "\n".join(lines)

    def to_echarts(self):
        """生成 ECharts 对比图配置"""
        from .views.echarts import EChartsConverter

        # 柱状图: 各项指标对比
        metrics_chart = EChartsConverter.bar_chart(
            x_data=["MAE", "MAPE", "RMSE", "Max Error%"],
            series_data={
                "理论分析": [
                    self.theoretical.mae,
                    self.theoretical.mape,
                    self.theoretical.rmse,
                    self.theoretical.max_error_pct,
                ],
                "工程化(ML)": [
                    self.empirical.mae,
                    self.empirical.mape,
                    self.empirical.rmse,
                    self.empirical.max_error_pct,
                ],
            },
            title="预测方法对比",
            y_label="误差值",
        )

        return metrics_chart


class MethodComparator:
    """
    方法对比器

    对比理论分析和工程化方法的预测精度
    """

    def __init__(self):
        self._theoretical_predictor = None
        self._empirical_predictor = None

    def set_theoretical_predictor(self, predictor):
        """
        设置理论分析预测器

        Args:
            predictor: 接受 Benchmark 返回 latency_us 的函数
        """
        self._theoretical_predictor = predictor

    def set_empirical_predictor(self, predictor):
        """
        设置工程化预测器

        Args:
            predictor: 接受 Benchmark 返回 latency_us 的函数
        """
        self._empirical_predictor = predictor

    def compare(self,
                benchmarks: List,
                actual_latencies: Dict[str, float]) -> MethodComparison:
        """
        执行对比

        Args:
            benchmarks: Benchmark 列表
            actual_latencies: {benchmark_name: actual_latency_us}

        Returns:
            MethodComparison
        """
        theoretical_results = []
        empirical_results = []
        details = []

        for bm in benchmarks:
            actual = actual_latencies.get(bm.name)
            if actual is None:
                continue

            # 理论分析预测
            if self._theoretical_predictor:
                t_pred = self._theoretical_predictor(bm)
                theoretical_results.append(PredictionResult(
                    benchmark_name=bm.name,
                    method=PredictMethod.THEORETICAL,
                    predicted_latency_us=t_pred,
                    actual_latency_us=actual,
                ))
            else:
                t_pred = None

            # 工程化预测
            if self._empirical_predictor:
                e_pred = self._empirical_predictor(bm)
                empirical_results.append(PredictionResult(
                    benchmark_name=bm.name,
                    method=PredictMethod.EMPIRICAL,
                    predicted_latency_us=e_pred,
                    actual_latency_us=actual,
                ))
            else:
                e_pred = None

            details.append({
                "benchmark": bm.name,
                "actual_us": actual,
                "theoretical_us": t_pred,
                "empirical_us": e_pred,
                "theoretical_error_pct": abs(t_pred - actual) / actual * 100 if t_pred else None,
                "empirical_error_pct": abs(e_pred - actual) / actual * 100 if e_pred else None,
            })

        # 计算指标
        t_metrics = self._compute_metrics(theoretical_results, PredictMethod.THEORETICAL)
        e_metrics = self._compute_metrics(empirical_results, PredictMethod.EMPIRICAL)

        return MethodComparison(
            theoretical=t_metrics,
            empirical=e_metrics,
            details=details,
        )

    def _compute_metrics(self, results: List[PredictionResult],
                         method: PredictMethod) -> ComparisonMetrics:
        """计算对比指标"""
        if not results:
            return ComparisonMetrics(method=method, sample_count=0)

        actuals = np.array([r.actual_latency_us for r in results])
        predictions = np.array([r.predicted_latency_us for r in results])
        errors = np.abs(predictions - actuals)
        error_pcts = errors / actuals * 100

        # MAE, MAPE, RMSE
        mae = np.mean(errors)
        mape = np.mean(error_pcts)
        rmse = np.sqrt(np.mean(errors ** 2))

        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # 百分位
        percentiles = {
            50: float(np.percentile(error_pcts, 50)),
            90: float(np.percentile(error_pcts, 90)),
            99: float(np.percentile(error_pcts, 99)),
        }

        return ComparisonMetrics(
            method=method,
            sample_count=len(results),
            mae=float(mae),
            mape=float(mape),
            rmse=float(rmse),
            max_error=float(np.max(errors)),
            max_error_pct=float(np.max(error_pcts)),
            r_squared=float(r_squared),
            error_std=float(np.std(error_pcts)),
            error_percentiles=percentiles,
        )

    @classmethod
    def from_calibration(cls,
                         archive,
                         calibration_result,
                         benchmarks: Optional[Dict[str, Any]] = None) -> "MethodComparison":
        """
        从校准结果自动对比

        自动对比:
        - 理论分析: 使用校准前的默认参数
        - 工程化: 使用校准后的参数

        Args:
            archive: MeasurementArchive (包含实测数据)
            calibration_result: CalibrationResult (校准结果)
            benchmarks: {sample_id: Benchmark} 映射 (可选)

        Returns:
            MethodComparison
        """
        from .fusion_rules import FusionHyperParams
        from .hyper_calibrator import HyperCalibrator
        from .measurement_archive import MeasurementSample, LabelVector, MeasurementSource

        old_params = calibration_result.old_params  # 理论/默认参数
        new_params = calibration_result.new_params  # 工程化/校准后参数

        theoretical_results = []
        empirical_results = []
        details = []

        # 创建预测器
        calibrator = HyperCalibrator(archive)

        for sample in archive:
            actual = sample.labels.latency_us

            # 理论预测 (使用旧参数)
            t_pred = calibrator._predict_latency(old_params, sample)

            # 工程化预测 (使用新参数)
            e_pred = calibrator._predict_latency(new_params, sample)

            theoretical_results.append(PredictionResult(
                benchmark_name=sample.id,
                method=PredictMethod.THEORETICAL,
                predicted_latency_us=t_pred,
                actual_latency_us=actual,
            ))

            empirical_results.append(PredictionResult(
                benchmark_name=sample.id,
                method=PredictMethod.EMPIRICAL,
                predicted_latency_us=e_pred,
                actual_latency_us=actual,
            ))

            details.append({
                "benchmark": sample.id,
                "actual_us": actual,
                "theoretical_us": t_pred,
                "empirical_us": e_pred,
                "theoretical_error_pct": abs(t_pred - actual) / actual * 100 if actual > 0 else 0,
                "empirical_error_pct": abs(e_pred - actual) / actual * 100 if actual > 0 else 0,
                "improvement_pct": (abs(t_pred - actual) - abs(e_pred - actual)) / actual * 100 if actual > 0 else 0,
            })

        comparator = cls()
        t_metrics = comparator._compute_metrics(theoretical_results, PredictMethod.THEORETICAL)
        e_metrics = comparator._compute_metrics(empirical_results, PredictMethod.EMPIRICAL)

        return MethodComparison(
            theoretical=t_metrics,
            empirical=e_metrics,
            details=details,
        )


# 便捷函数
def compare_methods(theoretical_predictions: Dict[str, float],
                    empirical_predictions: Dict[str, float],
                    actual_latencies: Dict[str, float]) -> MethodComparison:
    """
    简化对比接口

    Args:
        theoretical_predictions: {bm_name: predicted_us} 理论预测
        empirical_predictions: {bm_name: predicted_us} 工程化预测
        actual_latencies: {bm_name: actual_us} 实测值

    Returns:
        MethodComparison

    示例:
        theoretical = {"bm1": 100.0, "bm2": 200.0}
        empirical = {"bm1": 95.0, "bm2": 210.0}
        actual = {"bm1": 98.0, "bm2": 205.0}

        result = compare_methods(theoretical, empirical, actual)
        print(result.summary())
    """
    theoretical_results = []
    empirical_results = []
    details = []

    for bm_name, actual in actual_latencies.items():
        t_pred = theoretical_predictions.get(bm_name)
        e_pred = empirical_predictions.get(bm_name)

        if t_pred is not None:
            theoretical_results.append(PredictionResult(
                benchmark_name=bm_name,
                method=PredictMethod.THEORETICAL,
                predicted_latency_us=t_pred,
                actual_latency_us=actual,
            ))

        if e_pred is not None:
            empirical_results.append(PredictionResult(
                benchmark_name=bm_name,
                method=PredictMethod.EMPIRICAL,
                predicted_latency_us=e_pred,
                actual_latency_us=actual,
            ))

        details.append({
            "benchmark": bm_name,
            "actual_us": actual,
            "theoretical_us": t_pred,
            "empirical_us": e_pred,
            "theoretical_error_pct": abs(t_pred - actual) / actual * 100 if t_pred else None,
            "empirical_error_pct": abs(e_pred - actual) / actual * 100 if e_pred else None,
        })

    comparator = MethodComparator()
    t_metrics = comparator._compute_metrics(theoretical_results, PredictMethod.THEORETICAL)
    e_metrics = comparator._compute_metrics(empirical_results, PredictMethod.EMPIRICAL)

    return MethodComparison(
        theoretical=t_metrics,
        empirical=e_metrics,
        details=details,
    )


@dataclass
class CalibrateAndCompareResult:
    """校准并对比的完整结果"""
    calibration: Any  # CalibrationResult
    comparison: MethodComparison
    calibrated_predictions: Dict[str, float]  # {bm_name: calibrated_latency_us}

    def summary(self) -> str:
        """生成完整摘要"""
        lines = [
            "=" * 70,
            "校准 + 对比 结果",
            "=" * 70,
            "",
            f"校准方法: {self.calibration.method.value}",
            f"校准改进: {self.calibration.improvement():.1f}%",
            f"R²: {self.calibration.r_squared:.4f}",
            "",
            self.comparison.summary(),
            "",
            "校准后时延预测:",
            "-" * 40,
        ]

        for bm_name, pred in list(self.calibrated_predictions.items())[:10]:
            lines.append(f"  {bm_name}: {pred:.2f} us")

        if len(self.calibrated_predictions) > 10:
            lines.append(f"  ... 共 {len(self.calibrated_predictions)} 条")

        return "\n".join(lines)

    def export_predictions(self, path: str) -> None:
        """导出校准后的预测结果"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "method": "empirical_calibrated",
                "r_squared": self.calibration.r_squared,
                "mape": self.comparison.empirical.mape,
                "predictions": self.calibrated_predictions,
            }, f, indent=2, ensure_ascii=False)


def calibrate_and_compare(archive,
                          method: str = "differential_evolution",
                          max_iterations: int = 100,
                          benchmarks: Optional[Dict[str, Any]] = None) -> CalibrateAndCompareResult:
    """
    一键校准并对比

    流程:
    1. 使用默认参数 (理论分析)
    2. 执行 ML 校准
    3. 对比优化前后精度
    4. 输出校准后的时延预测

    Args:
        archive: MeasurementArchive (包含实测数据)
        method: 优化方法 ("differential_evolution", "bayesian", "random_search")
        max_iterations: 最大迭代次数
        benchmarks: {sample_id: Benchmark} 映射 (可选，用于预测新数据)

    Returns:
        CalibrateAndCompareResult

    示例:
        from aidevtools.optimizer import MeasurementArchive, calibrate_and_compare

        # 导入实测数据
        archive = MeasurementArchive()
        archive.import_from_benchmarks(results, benchmarks)

        # 一键校准并对比
        result = calibrate_and_compare(archive)
        print(result.summary())

        # 导出校准后的预测
        result.export_predictions("calibrated_predictions.json")
    """
    from .hyper_calibrator import HyperCalibrator, OptimizeMethod

    # 方法映射
    method_map = {
        "differential_evolution": OptimizeMethod.DIFFERENTIAL_EVOLUTION,
        "bayesian": OptimizeMethod.BAYESIAN,
        "random_search": OptimizeMethod.RANDOM_SEARCH,
        "grid_search": OptimizeMethod.GRID_SEARCH,
        "gradient": OptimizeMethod.GRADIENT,
    }

    opt_method = method_map.get(method, OptimizeMethod.DIFFERENTIAL_EVOLUTION)

    # 1. 执行校准
    calibrator = HyperCalibrator(archive)
    calibration_result = calibrator.calibrate(
        method=opt_method,
        max_iterations=max_iterations,
    )

    # 2. 对比
    comparison = MethodComparator.from_calibration(archive, calibration_result, benchmarks)

    # 3. 生成校准后的预测
    calibrated_predictions = {}
    new_params = calibration_result.new_params

    for sample in archive:
        pred = calibrator._predict_latency(new_params, sample)
        calibrated_predictions[sample.id] = pred

    # 4. 应用新参数
    calibrator.apply(calibration_result, validate=False)

    return CalibrateAndCompareResult(
        calibration=calibration_result,
        comparison=comparison,
        calibrated_predictions=calibrated_predictions,
    )
