"""
超参数校准器

用于从实测数据学习并更新超参数

流程:
1. 从 MeasurementArchive 加载训练数据
2. 使用优化算法拟合超参数
3. 验证拟合效果
4. 更新到 FusionRules

支持的优化方法:
- 网格搜索 (Grid Search)
- 贝叶斯优化 (Bayesian Optimization)
- 差分进化 (Differential Evolution)
- 梯度下降 (Gradient Descent)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
import json
import numpy as np
from datetime import datetime

from .fusion_rules import FusionHyperParams, FusionRules, get_fusion_rules
from .measurement_archive import MeasurementArchive, MeasurementSample


class OptimizeMethod(Enum):
    """优化方法"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BAYESIAN = "bayesian"
    GRADIENT = "gradient"


@dataclass
class CalibrationResult:
    """校准结果"""
    method: OptimizeMethod
    old_params: FusionHyperParams
    new_params: FusionHyperParams

    # 拟合指标
    train_loss: float
    val_loss: Optional[float] = None
    r_squared: float = 0.0
    rmse: float = 0.0
    mape: float = 0.0  # Mean Absolute Percentage Error

    # 优化过程
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)

    # 时间
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    def improvement(self) -> float:
        """相比旧参数的改进百分比"""
        if self.train_loss == 0:
            return 0.0
        # 假设 old_loss 在 convergence_history[0]
        if self.convergence_history:
            old_loss = self.convergence_history[0]
            return (old_loss - self.train_loss) / old_loss * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method.value,
            "old_params": self.old_params.to_dict(),
            "new_params": self.new_params.to_dict(),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "mape": self.mape,
            "iterations": self.iterations,
            "improvement_pct": self.improvement(),
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }


class HyperCalibrator:
    """
    超参数校准器

    从实测数据学习超参数
    """

    def __init__(self,
                 archive: Optional[MeasurementArchive] = None,
                 rules: Optional[FusionRules] = None):
        """
        Args:
            archive: 测量数据归档
            rules: 融合规则库 (None 使用全局)
        """
        self.archive = archive or MeasurementArchive()
        self.rules = rules or get_fusion_rules()
        self._cost_fn: Optional[Callable] = None

    def set_cost_function(self, cost_fn: Callable[[FusionHyperParams, List[MeasurementSample]], float]):
        """
        设置自定义代价函数

        Args:
            cost_fn: (params, samples) -> loss
        """
        self._cost_fn = cost_fn

    def _default_cost_function(self, params: FusionHyperParams,
                               samples: List[MeasurementSample]) -> float:
        """默认代价函数: MAPE (基于 latency_us)"""
        if not samples:
            return float('inf')

        errors = []
        for sample in samples:
            predicted = self._predict_latency(params, sample)
            actual = sample.labels.latency_us  # 使用 latency_us 作为主要标签

            if actual > 0:
                error = abs(predicted - actual) / actual
                errors.append(error)

        return np.mean(errors) if errors else float('inf')

    def _predict_latency(self, params: FusionHyperParams,
                        sample: MeasurementSample) -> float:
        """
        使用超参数预测时延 (latency_us)

        Returns:
            预测的时延 (微秒)
        """
        f = sample.features
        frequency_mhz = f.frequency_mhz if f.frequency_mhz > 0 else 1000

        # 基础计算时间 (简化估算)
        # 假设 1 GFLOP/s 的基准性能
        base_us = f.total_flops / 1e9 * 1000  # GFLOPS -> us

        # 融合收益
        if f.is_fused and f.fusion_depth > 1:
            # 使用超参数计算组合加速比
            decay = params.decay_base
            speedup = params.speedup_base

            for i in range(f.fusion_depth - 1):
                gain = (1.2 - 1.0) * params.speedup_scale * decay  # 假设基础 speedup=1.2
                speedup *= (1.0 + gain)
                decay *= params.decay_rate

            # 深度惩罚
            speedup *= (1.0 - params.depth_penalty * (f.fusion_depth - 2))
            speedup = min(speedup, params.pairwise_ceiling)

            base_us /= max(1.0, speedup)

        # 底噪开销 (cycles -> us)
        overhead_cycles = params.op_submit_base * f.num_ops
        overhead_cycles += params.op_launch_latency * f.num_ops

        if f.is_fused:
            overhead_cycles *= (1.0 - params.fuse_submit_save)

        # DMA 开销 (cycles)
        dma_cycles = params.dma_submit_base * (f.total_bytes / 1024)
        dma_cycles += params.dma_setup_per_kb * (f.total_bytes / 1024)

        if f.is_fused:
            dma_cycles *= (1.0 - params.fuse_dma_save)

        # cycles -> us
        overhead_us = (overhead_cycles + dma_cycles) / frequency_mhz

        return base_us + overhead_us

    # ==================== 优化方法 ====================

    def calibrate(self,
                 method: OptimizeMethod = OptimizeMethod.DIFFERENTIAL_EVOLUTION,
                 train_ratio: float = 0.8,
                 max_iterations: int = 100,
                 **kwargs) -> CalibrationResult:
        """
        执行校准

        Args:
            method: 优化方法
            train_ratio: 训练集比例
            max_iterations: 最大迭代次数
            **kwargs: 方法特定参数

        Returns:
            CalibrationResult
        """
        import time
        start_time = time.time()

        # 准备数据
        samples = list(self.archive)
        np.random.shuffle(samples)

        split_idx = int(len(samples) * train_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:] if split_idx < len(samples) else []

        # 保存旧参数
        old_params = FusionHyperParams.from_dict(self.rules.hyper_params.to_dict())

        # 代价函数
        cost_fn = self._cost_fn or self._default_cost_function

        # 执行优化
        if method == OptimizeMethod.GRID_SEARCH:
            new_params, history = self._grid_search(cost_fn, train_samples, **kwargs)
        elif method == OptimizeMethod.RANDOM_SEARCH:
            new_params, history = self._random_search(cost_fn, train_samples, max_iterations, **kwargs)
        elif method == OptimizeMethod.DIFFERENTIAL_EVOLUTION:
            new_params, history = self._differential_evolution(cost_fn, train_samples, max_iterations, **kwargs)
        elif method == OptimizeMethod.BAYESIAN:
            new_params, history = self._bayesian_optimize(cost_fn, train_samples, max_iterations, **kwargs)
        elif method == OptimizeMethod.GRADIENT:
            new_params, history = self._gradient_descent(cost_fn, train_samples, max_iterations, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 计算指标
        train_loss = cost_fn(new_params, train_samples)
        val_loss = cost_fn(new_params, val_samples) if val_samples else None

        # 计算 R², RMSE
        r_squared, rmse, mape = self._compute_metrics(new_params, samples)

        duration = time.time() - start_time

        return CalibrationResult(
            method=method,
            old_params=old_params,
            new_params=new_params,
            train_loss=train_loss,
            val_loss=val_loss,
            r_squared=r_squared,
            rmse=rmse,
            mape=mape,
            iterations=len(history),
            convergence_history=history,
            duration_seconds=duration,
        )

    def _grid_search(self, cost_fn: Callable,
                    samples: List[MeasurementSample],
                    grid_points: int = 5,
                    **kwargs) -> Tuple[FusionHyperParams, List[float]]:
        """网格搜索"""
        bounds = FusionHyperParams.param_bounds()
        param_names = FusionHyperParams.param_names()

        # 选择关键参数进行搜索
        key_params = ["decay_base", "speedup_scale", "depth_penalty",
                     "op_submit_base", "dma_submit_base"]
        key_indices = [param_names.index(p) for p in key_params if p in param_names]

        best_params = self.rules.hyper_params
        best_loss = cost_fn(best_params, samples)
        history = [best_loss]

        # 生成网格
        grids = []
        for idx in key_indices:
            low, high = bounds[idx]
            grids.append(np.linspace(low, high, grid_points))

        # 遍历网格
        from itertools import product
        for values in product(*grids):
            vec = best_params.to_vector()
            for i, idx in enumerate(key_indices):
                vec[idx] = values[i]

            params = FusionHyperParams.from_vector(vec)
            loss = cost_fn(params, samples)
            history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = params

        return best_params, history

    def _random_search(self, cost_fn: Callable,
                      samples: List[MeasurementSample],
                      max_iterations: int,
                      **kwargs) -> Tuple[FusionHyperParams, List[float]]:
        """随机搜索"""
        bounds = FusionHyperParams.param_bounds()

        best_params = self.rules.hyper_params
        best_loss = cost_fn(best_params, samples)
        history = [best_loss]

        for _ in range(max_iterations):
            # 随机采样
            vec = []
            for low, high in bounds:
                vec.append(np.random.uniform(low, high))

            params = FusionHyperParams.from_vector(vec)
            loss = cost_fn(params, samples)
            history.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_params = params

        return best_params, history

    def _differential_evolution(self, cost_fn: Callable,
                               samples: List[MeasurementSample],
                               max_iterations: int,
                               **kwargs) -> Tuple[FusionHyperParams, List[float]]:
        """差分进化"""
        from scipy.optimize import differential_evolution

        bounds = FusionHyperParams.param_bounds()
        history = []

        def objective(vec):
            params = FusionHyperParams.from_vector(list(vec))
            loss = cost_fn(params, samples)
            history.append(loss)
            return loss

        result = differential_evolution(
            objective,
            bounds,
            maxiter=max_iterations,
            seed=42,
            polish=True,
            **kwargs
        )

        best_params = FusionHyperParams.from_vector(list(result.x))
        return best_params, history

    def _bayesian_optimize(self, cost_fn: Callable,
                          samples: List[MeasurementSample],
                          max_iterations: int,
                          **kwargs) -> Tuple[FusionHyperParams, List[float]]:
        """贝叶斯优化 (需要 scikit-optimize)"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real
        except ImportError:
            print("Warning: scikit-optimize not installed, falling back to random search")
            return self._random_search(cost_fn, samples, max_iterations)

        bounds = FusionHyperParams.param_bounds()
        space = [Real(low, high) for low, high in bounds]
        history = []

        def objective(vec):
            params = FusionHyperParams.from_vector(vec)
            loss = cost_fn(params, samples)
            history.append(loss)
            return loss

        result = gp_minimize(
            objective,
            space,
            n_calls=max_iterations,
            random_state=42,
            **kwargs
        )

        best_params = FusionHyperParams.from_vector(result.x)
        return best_params, history

    def _gradient_descent(self, cost_fn: Callable,
                         samples: List[MeasurementSample],
                         max_iterations: int,
                         learning_rate: float = 0.01,
                         **kwargs) -> Tuple[FusionHyperParams, List[float]]:
        """数值梯度下降"""
        bounds = FusionHyperParams.param_bounds()
        vec = np.array(self.rules.hyper_params.to_vector())
        history = []
        epsilon = 1e-5

        for _ in range(max_iterations):
            params = FusionHyperParams.from_vector(list(vec))
            loss = cost_fn(params, samples)
            history.append(loss)

            # 数值梯度
            grad = np.zeros_like(vec)
            for i in range(len(vec)):
                vec_plus = vec.copy()
                vec_plus[i] += epsilon
                params_plus = FusionHyperParams.from_vector(list(vec_plus))
                loss_plus = cost_fn(params_plus, samples)
                grad[i] = (loss_plus - loss) / epsilon

            # 更新
            vec = vec - learning_rate * grad

            # 裁剪到边界
            for i, (low, high) in enumerate(bounds):
                vec[i] = np.clip(vec[i], low, high)

        best_params = FusionHyperParams.from_vector(list(vec))
        return best_params, history

    def _compute_metrics(self, params: FusionHyperParams,
                        samples: List[MeasurementSample]) -> Tuple[float, float, float]:
        """计算 R², RMSE, MAPE (基于 latency_us)"""
        if not samples:
            return 0.0, 0.0, 0.0

        actuals = []
        predictions = []

        for sample in samples:
            actual = sample.labels.latency_us  # 使用 latency_us
            predicted = self._predict_latency(params, sample)
            actuals.append(actual)
            predictions.append(predicted)

        actuals = np.array(actuals)
        predictions = np.array(predictions)

        # R²
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # RMSE (单位: us)
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

        # MAPE (%)
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1e-6))) * 100

        return r_squared, rmse, mape

    # ==================== 更新接口 ====================

    def apply(self, result: CalibrationResult, validate: bool = True) -> bool:
        """
        应用校准结果到规则库

        Args:
            result: 校准结果
            validate: 是否验证改进

        Returns:
            是否成功应用
        """
        if validate:
            # 检查是否有改进
            if result.val_loss is not None and result.val_loss >= result.train_loss * 1.5:
                print(f"Warning: Validation loss ({result.val_loss:.4f}) much higher than "
                      f"training loss ({result.train_loss:.4f}), possible overfitting")

            if result.improvement() < 0:
                print(f"Warning: New params are worse than old ({result.improvement():.1f}%)")
                return False

        # 更新
        self.rules.hyper_params = result.new_params
        print(f"Applied new hyperparameters (improvement: {result.improvement():.1f}%)")
        return True

    def rollback(self, result: CalibrationResult) -> None:
        """回滚到旧参数"""
        self.rules.hyper_params = result.old_params
        print("Rolled back to old hyperparameters")

    def save_result(self, result: CalibrationResult, path: str) -> None:
        """保存校准结果"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def update_and_save(self, result: CalibrationResult,
                       rules_path: str,
                       result_path: Optional[str] = None) -> bool:
        """
        应用更新并保存

        Args:
            result: 校准结果
            rules_path: 规则文件路径
            result_path: 结果文件路径 (可选)

        Returns:
            是否成功
        """
        if not self.apply(result):
            return False

        # 保存规则
        self.rules.save(rules_path)
        print(f"Saved updated rules to {rules_path}")

        # 保存结果
        if result_path:
            self.save_result(result, result_path)
            print(f"Saved calibration result to {result_path}")

        return True


# ==================== 便捷函数 ====================


def calibrate_from_archive(archive_path: str,
                          method: OptimizeMethod = OptimizeMethod.DIFFERENTIAL_EVOLUTION,
                          output_path: Optional[str] = None) -> CalibrationResult:
    """
    从归档文件校准超参数

    Args:
        archive_path: 归档文件路径
        method: 优化方法
        output_path: 输出规则路径

    Returns:
        CalibrationResult
    """
    archive = MeasurementArchive(archive_path)
    calibrator = HyperCalibrator(archive)

    result = calibrator.calibrate(method=method)

    if output_path:
        calibrator.update_and_save(result, output_path)

    return result


def quick_calibrate(samples: List[MeasurementSample],
                   method: OptimizeMethod = OptimizeMethod.RANDOM_SEARCH,
                   max_iterations: int = 50) -> FusionHyperParams:
    """
    快速校准

    Args:
        samples: 样本列表
        method: 优化方法
        max_iterations: 最大迭代

    Returns:
        优化后的超参数
    """
    archive = MeasurementArchive()
    for s in samples:
        archive.add(s)

    calibrator = HyperCalibrator(archive)
    result = calibrator.calibrate(method=method, max_iterations=max_iterations)
    calibrator.apply(result, validate=False)

    return result.new_params
