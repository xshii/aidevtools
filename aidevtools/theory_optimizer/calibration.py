"""
ML 校准模块

设计模式:
- Observer 模式: 参数更新通知
- Strategy 模式: 不同的拟合策略
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .cost_model import CostParameters, CostResult, ParameterizedCostModel


# ============================================================
# 数据结构
# ============================================================


@dataclass
class MeasurementRecord:
    """单条实测记录"""

    op_type: str
    shapes: Dict[str, int]
    dtype: str = "fp16"
    compute_unit: str = "cube"
    tile_config: Dict[str, int] = field(default_factory=dict)
    memory_pattern: str = "sequential"
    is_fused: bool = False
    fuse_partner: Optional[str] = None
    measured_us: float = 0.0
    predicted_us: Optional[float] = None
    timestamp: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

    @property
    def error_ratio(self) -> float:
        """误差比例"""
        if self.predicted_us and self.measured_us > 0:
            return (self.predicted_us - self.measured_us) / self.measured_us
        return 0.0

    def to_dict(self) -> dict:
        return {
            "op_type": self.op_type,
            "shapes": self.shapes,
            "dtype": self.dtype,
            "compute_unit": self.compute_unit,
            "tile_config": self.tile_config,
            "memory_pattern": self.memory_pattern,
            "is_fused": self.is_fused,
            "fuse_partner": self.fuse_partner,
            "measured_us": self.measured_us,
            "predicted_us": self.predicted_us,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MeasurementRecord":
        return cls(**d)


@dataclass
class FitResult:
    """拟合结果"""

    params: CostParameters
    r2_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    max_error: float = 0.0
    n_samples: int = 0
    convergence_history: List[float] = field(default_factory=list)
    method: str = ""

    def to_dict(self) -> dict:
        return {
            "r2_score": self.r2_score,
            "mse": self.mse,
            "mae": self.mae,
            "mape": self.mape,
            "max_error": self.max_error,
            "n_samples": self.n_samples,
            "method": self.method,
        }


@dataclass
class EvalMetrics:
    """评估指标"""

    overall_mape: float = 0.0
    overall_mae: float = 0.0
    per_op_mape: Dict[str, float] = field(default_factory=dict)
    per_op_bias: Dict[str, float] = field(default_factory=dict)
    outlier_count: int = 0
    outlier_records: List[MeasurementRecord] = field(default_factory=list)


@dataclass
class CVResult:
    """交叉验证结果"""

    mean_mape: float = 0.0
    std_mape: float = 0.0
    fold_results: List[FitResult] = field(default_factory=list)


# ============================================================
# 拟合策略 (Strategy 模式)
# ============================================================


class FitStrategy:
    """拟合策略基类"""

    name: str = "base"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: CostParameters,
        **kwargs,
    ) -> CostParameters:
        """执行拟合"""
        raise NotImplementedError


class LinearFitStrategy(FitStrategy):
    """线性回归拟合"""

    name = "linear"

    def fit(self, X, y, params, **kwargs):
        """
        线性回归拟合效率参数

        模型: T = FLOPs/(η_c × P) + Bytes/(η_m × BW) + overhead
        """
        # X 的列: [flops/P, bytes/BW, n_tiles, n_dma, 1]
        # 拟合: y = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b

        try:
            coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            return params

        # 从系数恢复效率参数
        # w1 = 1/eta_compute, w2 = 1/eta_memory
        if len(coef) >= 5:
            if coef[0] > 0:
                params.compute.eta_cube = min(1.0, max(0.1, 1.0 / coef[0]))
            if coef[1] > 0:
                params.memory.eta_sequential = min(1.0, max(0.1, 1.0 / coef[1]))
            params.overhead.alpha_tile = max(0, coef[2])
            params.overhead.beta_dma = max(0, coef[3])
            params.overhead.gamma_base = max(0, coef[4])

        return params


class LayeredFitStrategy(FitStrategy):
    """分层拟合策略 (推荐)"""

    name = "layered"

    def fit(self, X, y, params, **kwargs):
        """
        分层拟合:
        1. 拟合全局效率参数
        2. 拟合开销参数
        3. 拟合算子特定修正系数
        """
        chip = kwargs.get("chip")
        records = kwargs.get("records", [])

        if chip is None or len(records) == 0:
            return params

        # Step 1: 分离 Cube 和 Vector 数据
        cube_mask = np.array([r.compute_unit == "cube" for r in records])
        vec_mask = ~cube_mask

        # 拟合 Cube 效率
        if np.sum(cube_mask) > 3:
            params.compute.eta_cube = self._fit_efficiency(
                X[cube_mask], y[cube_mask], chip, "cube"
            )

        # 拟合 Vector 效率
        if np.sum(vec_mask) > 3:
            params.compute.eta_vector = self._fit_efficiency(
                X[vec_mask], y[vec_mask], chip, "vector"
            )

        # Step 2: 计算残差，拟合开销参数
        predicted_roofline = self._predict_roofline(X, params, chip, records)
        residual = y - predicted_roofline

        n_tiles = X[:, 2] if X.shape[1] > 2 else np.ones(len(y))
        n_dma = X[:, 3] if X.shape[1] > 3 else np.ones(len(y))

        X_overhead = np.column_stack([n_tiles, n_dma, np.ones(len(y))])
        try:
            coef, _, _, _ = np.linalg.lstsq(X_overhead, residual, rcond=None)
            params.overhead.alpha_tile = max(0, coef[0])
            params.overhead.beta_dma = max(0, coef[1])
            params.overhead.gamma_base = max(0, coef[2])
        except np.linalg.LinAlgError:
            pass

        # Step 3: 拟合算子特定修正系数
        predicted_with_overhead = predicted_roofline + (
            params.overhead.alpha_tile * n_tiles
            + params.overhead.beta_dma * n_dma
            + params.overhead.gamma_base
        )

        op_types = [r.op_type for r in records]
        unique_ops = set(op_types)

        for op_type in unique_ops:
            mask = np.array([r.op_type == op_type for r in records])
            if np.sum(mask) > 0:
                ratios = y[mask] / predicted_with_overhead[mask]
                # 使用中位数更稳健
                params.theta[op_type] = float(np.median(ratios[np.isfinite(ratios)]))

        return params

    def _fit_efficiency(self, X, y, chip, unit):
        """拟合单个计算单元的效率"""
        if unit == "cube":
            peak = chip.cube.fp16_tflops * 1e12
        else:
            peak = chip.vector.fp16_gflops * 1e9

        hbm_bw = chip.memory.hbm.bandwidth_gbps * 1e9

        # 理论时间
        flops = X[:, 0]
        bytes_col = X[:, 1]

        t_compute_ideal = flops / peak * 1e6
        t_memory_ideal = bytes_col / hbm_bw * 1e6
        t_roofline_ideal = np.maximum(t_compute_ideal, t_memory_ideal)

        # 效率 = 理想时间 / 实测时间
        valid = (y > 0) & (t_roofline_ideal > 0)
        if np.sum(valid) > 0:
            eff = t_roofline_ideal[valid] / y[valid]
            return float(np.clip(np.median(eff), 0.1, 1.0))
        return 0.85

    def _predict_roofline(self, X, params, chip, records):
        """预测 Roofline 时间"""
        flops = X[:, 0]
        bytes_col = X[:, 1]

        predicted = np.zeros(len(flops))
        for i, r in enumerate(records):
            if r.compute_unit == "cube":
                eta = params.compute.eta_cube
                peak = chip.cube.fp16_tflops * 1e12
            else:
                eta = params.compute.eta_vector
                peak = chip.vector.fp16_gflops * 1e9

            eta_mem = params.memory.eta_sequential
            hbm_bw = chip.memory.hbm.bandwidth_gbps * 1e9

            t_compute = flops[i] / (eta * peak) * 1e6 if peak > 0 else 0
            t_memory = bytes_col[i] / (eta_mem * hbm_bw) * 1e6 if hbm_bw > 0 else 0
            predicted[i] = max(t_compute, t_memory)

        return predicted


class GradientFitStrategy(FitStrategy):
    """梯度下降拟合"""

    name = "gradient"

    def fit(self, X, y, params, **kwargs):
        """梯度下降全参数优化"""
        lr = kwargs.get("lr", 0.01)
        epochs = kwargs.get("epochs", 1000)
        patience = kwargs.get("patience", 50)

        param_vec = params.to_vector()
        bounds = params.get_param_bounds()

        best_loss = float("inf")
        best_vec = param_vec.copy()
        no_improve = 0

        for epoch in range(epochs):
            # 计算 loss (MAPE)
            predicted = self._predict_with_vec(X, param_vec, kwargs)
            loss = np.mean(np.abs(predicted - y) / np.maximum(y, 1e-6))

            if loss < best_loss:
                best_loss = loss
                best_vec = param_vec.copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

            # 数值梯度
            grad = self._compute_gradient(X, y, param_vec, kwargs)
            param_vec = param_vec - lr * grad

            # 投影到边界
            for i, (lo, hi) in enumerate(bounds):
                param_vec[i] = np.clip(param_vec[i], lo, hi)

        params.from_vector(best_vec)
        return params

    def _predict_with_vec(self, X, vec, kwargs):
        """用参数向量预测"""
        # 简化实现: 假设 X 列为 [flops/P, bytes/BW, ...]
        eta_c = vec[0]
        eta_m = vec[2]
        alpha = vec[5]
        beta = vec[6]
        gamma = vec[7]

        t_compute = X[:, 0] / eta_c if eta_c > 0 else X[:, 0]
        t_memory = X[:, 1] / eta_m if eta_m > 0 else X[:, 1]
        t_roofline = np.maximum(t_compute, t_memory)

        n_tiles = X[:, 2] if X.shape[1] > 2 else np.ones(len(X))
        n_dma = X[:, 3] if X.shape[1] > 3 else np.ones(len(X))

        return t_roofline + alpha * n_tiles + beta * n_dma + gamma

    def _compute_gradient(self, X, y, vec, kwargs, eps=1e-6):
        """数值梯度"""
        grad = np.zeros_like(vec)
        base_loss = np.mean(np.abs(self._predict_with_vec(X, vec, kwargs) - y) / np.maximum(y, 1e-6))

        for i in range(len(vec)):
            vec_perturbed = vec.copy()
            vec_perturbed[i] += eps
            loss = np.mean(np.abs(self._predict_with_vec(X, vec_perturbed, kwargs) - y) / np.maximum(y, 1e-6))
            grad[i] = (loss - base_loss) / eps

        return grad


# ============================================================
# ML 校准器
# ============================================================


class MLCalibrator:
    """
    机器学习校准器

    支持:
    - 多种数据导入方式 (CSV, JSON, API)
    - 多种拟合策略 (linear, layered, gradient)
    - 交叉验证
    - 持久化
    """

    # 拟合策略注册表
    _strategies: Dict[str, FitStrategy] = {
        "linear": LinearFitStrategy(),
        "layered": LayeredFitStrategy(),
        "gradient": GradientFitStrategy(),
    }

    def __init__(self, cost_model: ParameterizedCostModel):
        """
        Args:
            cost_model: 要校准的 Cost Model
        """
        self.cost_model = cost_model
        self.measurements: List[MeasurementRecord] = []
        self.fitted = False
        self._fit_result: Optional[FitResult] = None

    # ==================== 数据管理 ====================

    def add_measurement(self, record: MeasurementRecord):
        """添加单条实测记录"""
        # 计算预测值
        record.predicted_us = self._predict_single(record)
        self.measurements.append(record)
        self.fitted = False

    def add_measurements_batch(self, records: List[MeasurementRecord]):
        """批量添加"""
        for record in records:
            self.add_measurement(record)

    def import_csv(self, path: str):
        """从 CSV 导入"""
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = MeasurementRecord(
                    op_type=row["op_type"],
                    shapes=self._parse_shapes(row),
                    dtype=row.get("dtype", "fp16"),
                    compute_unit=row.get("compute_unit", "cube"),
                    tile_config=self._parse_tile_config(row),
                    memory_pattern=row.get("mem_pattern", "sequential"),
                    is_fused=row.get("is_fused", "false").lower() == "true",
                    fuse_partner=row.get("fuse_partner") or None,
                    measured_us=float(row["measured_us"]),
                )
                self.add_measurement(record)

    def import_json(self, path: str):
        """从 JSON 导入"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("measurements", data if isinstance(data, list) else [])
        for item in records:
            record = MeasurementRecord.from_dict(item)
            self.add_measurement(record)

    def export_measurements(self, path: str):
        """导出测量数据"""
        data = {
            "chip": self.cost_model.chip.name,
            "timestamp": datetime.now().isoformat(),
            "measurements": [r.to_dict() for r in self.measurements],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def clear_measurements(self):
        """清空测量数据"""
        self.measurements.clear()
        self.fitted = False

    def _parse_shapes(self, row: dict) -> Dict[str, int]:
        """解析 shapes"""
        shapes = {}
        for key in ["M", "K", "N", "C", "H", "W"]:
            if key in row and row[key]:
                shapes[key] = int(row[key])
        return shapes

    def _parse_tile_config(self, row: dict) -> Dict[str, int]:
        """解析 tile config"""
        config = {}
        for key in ["tile_m", "tile_n", "tile_k"]:
            if key in row and row[key]:
                config[key.replace("tile_", "")] = int(row[key])
        return config

    # ==================== 特征工程 ====================

    def extract_features(self, record: MeasurementRecord) -> np.ndarray:
        """提取单条记录的特征"""
        from aidevtools.analysis.profile import OpProfile

        # 计算 FLOPs 和 bytes
        profile = self._record_to_profile(record)

        flops = profile.flops
        io_bytes = profile.total_bytes
        n_tiles = self._count_tiles(record)
        n_dma = n_tiles  # 简化: 每个 tile 一次 DMA

        return np.array([flops, io_bytes, n_tiles, n_dma, 1.0])

    def extract_features_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """批量提取特征"""
        X = np.array([self.extract_features(r) for r in self.measurements])
        y = np.array([r.measured_us for r in self.measurements])
        return X, y

    def _record_to_profile(self, record: MeasurementRecord) -> "OpProfile":
        """将记录转换为 OpProfile"""
        from aidevtools.analysis.profile import OpProfile

        # 根据 op_type 和 shapes 计算
        from .benchmark import OpSpec, OpType

        try:
            op_type = OpType(record.op_type)
        except ValueError:
            op_type = OpType.MATMUL

        op = OpSpec(
            name="temp",
            op_type=op_type,
            shapes=record.shapes,
            dtype=record.dtype,
        )
        return op.to_profile()

    def _count_tiles(self, record: MeasurementRecord) -> int:
        """计算 tile 数量"""
        if not record.tile_config:
            return 1

        count = 1
        for dim, full_size in record.shapes.items():
            tile_size = record.tile_config.get(dim.lower(), full_size)
            count *= (full_size + tile_size - 1) // tile_size
        return count

    def _predict_single(self, record: MeasurementRecord) -> float:
        """预测单条记录"""
        profile = self._record_to_profile(record)
        n_tiles = self._count_tiles(record)

        result = self.cost_model.compute_cost(
            profile,
            n_tiles=n_tiles,
            memory_pattern=record.memory_pattern,
        )
        return result.calibrated_us

    # ==================== 模型拟合 ====================

    def fit(self, method: str = "layered", **kwargs) -> FitResult:
        """
        执行参数拟合

        Args:
            method: "linear" | "layered" | "gradient"
        """
        if len(self.measurements) < 5:
            raise ValueError("需要至少 5 条测量数据")

        X, y = self.extract_features_batch()

        strategy = self._strategies.get(method)
        if strategy is None:
            raise ValueError(f"Unknown fit method: {method}")

        # 传递额外参数
        kwargs["chip"] = self.cost_model.chip
        kwargs["records"] = self.measurements

        # 执行拟合
        new_params = strategy.fit(X, y, self.cost_model.params.clone(), **kwargs)
        self.cost_model.set_params(new_params)

        # 评估拟合效果
        metrics = self.evaluate()
        self._fit_result = FitResult(
            params=new_params.clone(),
            mape=metrics.overall_mape,
            mae=metrics.overall_mae,
            n_samples=len(self.measurements),
            method=method,
        )
        self.fitted = True

        return self._fit_result

    def fit_per_op_type(self) -> Dict[str, FitResult]:
        """按算子类型分别拟合"""
        results = {}
        op_types = set(r.op_type for r in self.measurements)

        for op_type in op_types:
            records = [r for r in self.measurements if r.op_type == op_type]
            if len(records) >= 3:
                # 创建临时校准器
                temp_calibrator = MLCalibrator(
                    ParameterizedCostModel(self.cost_model.chip)
                )
                temp_calibrator.measurements = records

                try:
                    result = temp_calibrator.fit(method="layered")
                    results[op_type] = result

                    # 更新主模型的 theta
                    self.cost_model.params.theta[op_type] = (
                        temp_calibrator.cost_model.params.theta.get(op_type, 1.0)
                    )
                except Exception:
                    pass

        return results

    # ==================== 评估与验证 ====================

    def evaluate(self) -> EvalMetrics:
        """评估当前模型"""
        metrics = EvalMetrics()

        if not self.measurements:
            return metrics

        errors = []
        per_op_errors: Dict[str, List[float]] = {}
        per_op_bias: Dict[str, List[float]] = {}

        for record in self.measurements:
            predicted = self._predict_single(record)
            actual = record.measured_us

            if actual > 0:
                error = abs(predicted - actual) / actual
                bias = (predicted - actual) / actual
                errors.append(error)

                if record.op_type not in per_op_errors:
                    per_op_errors[record.op_type] = []
                    per_op_bias[record.op_type] = []
                per_op_errors[record.op_type].append(error)
                per_op_bias[record.op_type].append(bias)

                # 离群点检测
                if error > 0.3:
                    metrics.outlier_count += 1
                    metrics.outlier_records.append(record)

        metrics.overall_mape = float(np.mean(errors)) if errors else 0
        metrics.overall_mae = float(np.mean([abs(e) for e in errors])) if errors else 0

        for op_type, errs in per_op_errors.items():
            metrics.per_op_mape[op_type] = float(np.mean(errs))
        for op_type, biases in per_op_bias.items():
            metrics.per_op_bias[op_type] = float(np.mean(biases))

        return metrics

    def cross_validate(self, k_folds: int = 5) -> CVResult:
        """K-fold 交叉验证"""
        if len(self.measurements) < k_folds * 2:
            raise ValueError(f"需要至少 {k_folds * 2} 条数据进行 {k_folds}-fold CV")

        indices = np.arange(len(self.measurements))
        np.random.shuffle(indices)
        folds = np.array_split(indices, k_folds)

        fold_mapes = []
        fold_results = []

        original_measurements = self.measurements.copy()
        original_params = self.cost_model.params.clone()

        for i in range(k_folds):
            # 分割训练/测试
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            self.measurements = [original_measurements[j] for j in train_idx]
            self.cost_model.set_params(original_params.clone())

            # 拟合
            result = self.fit(method="layered")
            fold_results.append(result)

            # 在测试集评估
            test_records = [original_measurements[j] for j in test_idx]
            test_errors = []
            for record in test_records:
                predicted = self._predict_single(record)
                if record.measured_us > 0:
                    test_errors.append(abs(predicted - record.measured_us) / record.measured_us)

            fold_mapes.append(float(np.mean(test_errors)) if test_errors else 0)

        # 恢复
        self.measurements = original_measurements
        self.cost_model.set_params(original_params)

        return CVResult(
            mean_mape=float(np.mean(fold_mapes)),
            std_mape=float(np.std(fold_mapes)),
            fold_results=fold_results,
        )

    def predict(self, record: MeasurementRecord) -> float:
        """预测单条记录"""
        return self._predict_single(record)

    def get_residuals(self) -> np.ndarray:
        """获取残差"""
        residuals = []
        for record in self.measurements:
            predicted = self._predict_single(record)
            residuals.append(predicted - record.measured_us)
        return np.array(residuals)

    # ==================== 持久化 ====================

    def save(self, path: str):
        """保存校准结果"""
        data = {
            "version": "1.0",
            "chip": self.cost_model.chip.name,
            "timestamp": datetime.now().isoformat(),
            "params": self.cost_model.params.to_dict(),
            "measurements": [r.to_dict() for r in self.measurements],
            "fit_result": self._fit_result.to_dict() if self._fit_result else None,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """加载校准结果"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 加载参数
        if "params" in data:
            self.cost_model.set_params(CostParameters.from_dict(data["params"]))

        # 加载测量数据
        self.measurements = []
        for item in data.get("measurements", []):
            self.measurements.append(MeasurementRecord.from_dict(item))

        self.fitted = data.get("fit_result") is not None

    def export_params_only(self, path: str):
        """仅导出参数"""
        data = {
            "version": "1.0",
            "chip": self.cost_model.chip.name,
            "params": self.cost_model.params.to_dict(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ==================== 策略注册 ====================

    @classmethod
    def register_strategy(cls, name: str, strategy: FitStrategy):
        """注册自定义拟合策略"""
        cls._strategies[name] = strategy
