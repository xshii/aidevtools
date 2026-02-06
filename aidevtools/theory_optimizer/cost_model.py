"""
参数化 Cost Model

设计模式:
- 组合模式: CostParameters 参数组合
- 策略模式: 不同的 Cost 计算策略
- 观察者模式: 参数更新通知

与 analysis 模块归一:
- 复用 analysis.chip.ChipSpec
- 复用 analysis.profile.OpProfile
- 复用 analysis.passes.roofline.RooflinePass 的计算逻辑
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from aidevtools.analysis.chip import ChipSpec, load_chip_spec
from aidevtools.analysis.profile import OpProfile

from .benchmark import Benchmark, OpSpec


# ============================================================
# 参数定义 (Composite 模式)
# ============================================================


@dataclass
class ComputeEfficiencyParams:
    """计算效率参数"""

    eta_cube: float = 0.85  # Cube 单元效率
    eta_vector: float = 0.70  # Vector 单元效率

    # 混合精度效率
    eta_mixed_fp16_int8: float = 0.80
    eta_mixed_fp16_int4: float = 0.75


@dataclass
class MemoryEfficiencyParams:
    """访存效率参数"""

    eta_sequential: float = 0.88  # 顺序访存
    eta_strided: float = 0.55  # 跨步访存
    eta_random: float = 0.30  # 随机访存

    # 各层级效率
    eta_l1: float = 0.95
    eta_l2: float = 0.90
    eta_hbm: float = 0.85


@dataclass
class OverheadParams:
    """开销参数"""

    alpha_tile: float = 0.1  # 每 Tile 切换开销 (us)
    beta_dma: float = 0.5  # 每 DMA 发起开销 (us)
    gamma_base: float = 1.0  # 算子基础开销 (us)
    delta_sync: float = 0.2  # 同步开销 (us)


@dataclass
class FuseParams:
    """融合参数"""

    kappa_io_save: float = 0.8  # 融合 IO 节省系数
    kappa_overhead_save: float = 0.5  # 融合开销节省系数


@dataclass
class CostParameters:
    """
    完整参数集合 (Composite 模式)

    包含所有可调参数，支持序列化和向量化
    """

    compute: ComputeEfficiencyParams = field(default_factory=ComputeEfficiencyParams)
    memory: MemoryEfficiencyParams = field(default_factory=MemoryEfficiencyParams)
    overhead: OverheadParams = field(default_factory=OverheadParams)
    fuse: FuseParams = field(default_factory=FuseParams)

    # 算子特定修正系数
    theta: Dict[str, float] = field(default_factory=lambda: {
        "matmul": 1.0,
        "conv": 1.0,
        "gelu": 1.0,
        "silu": 1.0,
        "relu": 1.0,
        "softmax": 1.0,
        "layernorm": 1.0,
        "rmsnorm": 1.0,
        "add": 1.0,
        "mul": 1.0,
    })

    # 观察者列表
    _observers: List[Callable] = field(default_factory=list, repr=False)

    def add_observer(self, callback: Callable):
        """添加观察者"""
        self._observers.append(callback)

    def notify_observers(self):
        """通知所有观察者"""
        for callback in self._observers:
            callback(self)

    def to_vector(self) -> np.ndarray:
        """转为参数向量 (用于优化)"""
        vec = [
            # 计算效率
            self.compute.eta_cube,
            self.compute.eta_vector,
            # 访存效率
            self.memory.eta_sequential,
            self.memory.eta_strided,
            self.memory.eta_random,
            # 开销
            self.overhead.alpha_tile,
            self.overhead.beta_dma,
            self.overhead.gamma_base,
            self.overhead.delta_sync,
            # 融合
            self.fuse.kappa_io_save,
            self.fuse.kappa_overhead_save,
        ]
        # 添加算子特定系数
        for op_type in sorted(self.theta.keys()):
            vec.append(self.theta[op_type])
        return np.array(vec)

    def from_vector(self, vec: np.ndarray):
        """从向量恢复参数"""
        idx = 0
        self.compute.eta_cube = vec[idx]; idx += 1
        self.compute.eta_vector = vec[idx]; idx += 1
        self.memory.eta_sequential = vec[idx]; idx += 1
        self.memory.eta_strided = vec[idx]; idx += 1
        self.memory.eta_random = vec[idx]; idx += 1
        self.overhead.alpha_tile = vec[idx]; idx += 1
        self.overhead.beta_dma = vec[idx]; idx += 1
        self.overhead.gamma_base = vec[idx]; idx += 1
        self.overhead.delta_sync = vec[idx]; idx += 1
        self.fuse.kappa_io_save = vec[idx]; idx += 1
        self.fuse.kappa_overhead_save = vec[idx]; idx += 1

        for op_type in sorted(self.theta.keys()):
            self.theta[op_type] = vec[idx]; idx += 1

        self.notify_observers()

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "compute": {
                "eta_cube": self.compute.eta_cube,
                "eta_vector": self.compute.eta_vector,
                "eta_mixed_fp16_int8": self.compute.eta_mixed_fp16_int8,
                "eta_mixed_fp16_int4": self.compute.eta_mixed_fp16_int4,
            },
            "memory": {
                "eta_sequential": self.memory.eta_sequential,
                "eta_strided": self.memory.eta_strided,
                "eta_random": self.memory.eta_random,
                "eta_l1": self.memory.eta_l1,
                "eta_l2": self.memory.eta_l2,
                "eta_hbm": self.memory.eta_hbm,
            },
            "overhead": {
                "alpha_tile": self.overhead.alpha_tile,
                "beta_dma": self.overhead.beta_dma,
                "gamma_base": self.overhead.gamma_base,
                "delta_sync": self.overhead.delta_sync,
            },
            "fuse": {
                "kappa_io_save": self.fuse.kappa_io_save,
                "kappa_overhead_save": self.fuse.kappa_overhead_save,
            },
            "theta": self.theta.copy(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CostParameters":
        """从字典反序列化"""
        params = cls()
        if "compute" in d:
            for k, v in d["compute"].items():
                setattr(params.compute, k, v)
        if "memory" in d:
            for k, v in d["memory"].items():
                setattr(params.memory, k, v)
        if "overhead" in d:
            for k, v in d["overhead"].items():
                setattr(params.overhead, k, v)
        if "fuse" in d:
            for k, v in d["fuse"].items():
                setattr(params.fuse, k, v)
        if "theta" in d:
            params.theta.update(d["theta"])
        return params

    def clone(self) -> "CostParameters":
        """克隆参数"""
        return CostParameters.from_dict(self.to_dict())

    def get_param_names(self) -> List[str]:
        """获取所有参数名"""
        names = [
            "compute.eta_cube", "compute.eta_vector",
            "memory.eta_sequential", "memory.eta_strided", "memory.eta_random",
            "overhead.alpha_tile", "overhead.beta_dma", "overhead.gamma_base", "overhead.delta_sync",
            "fuse.kappa_io_save", "fuse.kappa_overhead_save",
        ]
        for op_type in sorted(self.theta.keys()):
            names.append(f"theta.{op_type}")
        return names

    def get_param_bounds(self) -> List[Tuple[float, float]]:
        """获取参数边界 (用于约束优化)"""
        bounds = [
            (0.1, 1.0),  # eta_cube
            (0.1, 1.0),  # eta_vector
            (0.1, 1.0),  # eta_sequential
            (0.1, 1.0),  # eta_strided
            (0.1, 1.0),  # eta_random
            (0.0, 1.0),  # alpha_tile
            (0.0, 2.0),  # beta_dma
            (0.0, 5.0),  # gamma_base
            (0.0, 1.0),  # delta_sync
            (0.5, 1.0),  # kappa_io_save
            (0.3, 1.0),  # kappa_overhead_save
        ]
        # 算子特定系数边界
        for _ in self.theta:
            bounds.append((0.5, 2.0))
        return bounds


# ============================================================
# Cost 计算结果
# ============================================================


@dataclass
class CostResult:
    """单算子 Cost 计算结果"""

    op_name: str
    op_type: str

    # 时间分解 (us)
    compute_us: float = 0.0
    memory_us: float = 0.0
    roofline_us: float = 0.0
    overhead_us: float = 0.0
    total_us: float = 0.0

    # 融合调整
    fuse_speedup: float = 1.0
    fused_us: float = 0.0

    # 校准调整
    calibrated_us: float = 0.0

    # 瓶颈分析
    bottleneck: str = "memory"  # "compute" | "memory"
    arithmetic_intensity: float = 0.0
    ridge_point: float = 0.0

    # 效率
    compute_efficiency: float = 0.0
    memory_efficiency: float = 0.0

    # Tile 信息
    n_tiles: int = 1
    n_dma: int = 1

    # 详情
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 参数化 Cost Model
# ============================================================


class ParameterizedCostModel:
    """
    参数化 Cost Model

    与 analysis 模块归一:
    - 复用 ChipSpec 获取硬件规格
    - 计算逻辑与 RooflinePass 一致
    - 扩展了参数化能力
    """

    def __init__(
        self,
        chip: str | ChipSpec = "npu_910",
        params: CostParameters = None,
    ):
        """
        Args:
            chip: 芯片名称或 ChipSpec 对象
            params: Cost 参数 (默认使用默认值)
        """
        if isinstance(chip, str):
            self.chip = load_chip_spec(chip)
        else:
            self.chip = chip

        self.params = params or CostParameters()

        # 将 model 注册为参数的观察者
        self.params.add_observer(self._on_params_updated)

    def _on_params_updated(self, params: CostParameters):
        """参数更新回调 (Observer 模式)"""
        # 可以在这里做一些缓存清理等操作
        pass

    def compute_cost(
        self,
        op: OpSpec | OpProfile,
        n_tiles: int = 1,
        n_dma: int = 1,
        fuse_speedup: float = 1.0,
        memory_pattern: str = "sequential",
    ) -> CostResult:
        """
        计算单算子 Cost

        Args:
            op: 算子规格或 profile
            n_tiles: Tile 数量
            n_dma: DMA 次数
            fuse_speedup: 融合加速系数
            memory_pattern: 访存模式
        """
        # 转换为 OpProfile
        if isinstance(op, OpSpec):
            profile = op.to_profile()
            op_type = op.op_type.value
        else:
            profile = op
            op_type = profile.op_type

        # 获取计算单元效率
        if profile.compute_unit == "cube":
            eta_compute = self.params.compute.eta_cube
            peak_tflops = self.chip.cube.fp16_tflops
        else:
            eta_compute = self.params.compute.eta_vector
            peak_tflops = self.chip.vector.fp16_gflops / 1000

        # 获取访存效率
        eta_memory = self._get_memory_efficiency(memory_pattern)

        # 计算时间 (与 RooflinePass 逻辑一致)
        compute_us = 0.0
        if peak_tflops > 0 and profile.flops > 0:
            compute_us = profile.flops / (eta_compute * peak_tflops * 1e12) * 1e6

        # 访存时间
        hbm_bw = self.chip.memory.hbm.bandwidth_gbps
        memory_us = 0.0
        if hbm_bw > 0 and profile.total_bytes > 0:
            memory_us = profile.total_bytes / (eta_memory * hbm_bw * 1e9) * 1e6

        # Roofline
        roofline_us = max(compute_us, memory_us)

        # 开销
        overhead_us = (
            self.params.overhead.alpha_tile * n_tiles
            + self.params.overhead.beta_dma * n_dma
            + self.params.overhead.gamma_base
        )

        # 总时间
        total_us = roofline_us + overhead_us

        # 融合调整
        fused_us = total_us / fuse_speedup

        # 算子特定修正
        theta = self.params.theta.get(op_type, 1.0)
        calibrated_us = fused_us * theta

        # 瓶颈分析
        ai = profile.arithmetic_intensity
        ridge = (peak_tflops * 1e12) / (hbm_bw * 1e9) if hbm_bw > 0 else 0
        bottleneck = "compute" if ai >= ridge else "memory"

        # 效率
        compute_eff = min(1.0, ai / ridge) if ridge > 0 else 0
        memory_eff = min(1.0, ridge / ai) if ai > 0 else 0

        return CostResult(
            op_name=profile.name,
            op_type=op_type,
            compute_us=compute_us,
            memory_us=memory_us,
            roofline_us=roofline_us,
            overhead_us=overhead_us,
            total_us=total_us,
            fuse_speedup=fuse_speedup,
            fused_us=fused_us,
            calibrated_us=calibrated_us,
            bottleneck=bottleneck,
            arithmetic_intensity=ai,
            ridge_point=ridge,
            compute_efficiency=compute_eff,
            memory_efficiency=memory_eff,
            n_tiles=n_tiles,
            n_dma=n_dma,
            details={
                "eta_compute": eta_compute,
                "eta_memory": eta_memory,
                "peak_tflops": peak_tflops,
                "hbm_bw_gbps": hbm_bw,
                "theta": theta,
            },
        )

    def compute_benchmark_cost(
        self,
        benchmark: Benchmark,
        tile_counts: Dict[str, int] = None,
    ) -> List[CostResult]:
        """
        计算 Benchmark 的所有算子 Cost

        Args:
            benchmark: Benchmark 用例
            tile_counts: 各算子的 Tile 数量
        """
        tile_counts = tile_counts or {}
        results = []

        for op in benchmark.ops:
            n_tiles = tile_counts.get(op.name, 1)
            fuse_speedup = benchmark.get_fuse_speedup(op.name)

            cost = self.compute_cost(
                op,
                n_tiles=n_tiles,
                fuse_speedup=fuse_speedup,
            )
            results.append(cost)

        return results

    def get_total_cost(self, results: List[CostResult]) -> Dict[str, float]:
        """汇总 Cost 结果"""
        return {
            "total_us": sum(r.total_us for r in results),
            "fused_us": sum(r.fused_us for r in results),
            "calibrated_us": sum(r.calibrated_us for r in results),
            "compute_us": sum(r.compute_us for r in results),
            "memory_us": sum(r.memory_us for r in results),
            "overhead_us": sum(r.overhead_us for r in results),
        }

    def _get_memory_efficiency(self, pattern: str) -> float:
        """根据访存模式获取效率"""
        if pattern == "sequential":
            return self.params.memory.eta_sequential
        elif pattern == "strided":
            return self.params.memory.eta_strided
        elif pattern == "random":
            return self.params.memory.eta_random
        return self.params.memory.eta_sequential

    def set_params(self, params: CostParameters):
        """设置参数"""
        self.params = params
        params.add_observer(self._on_params_updated)

    def get_params(self) -> CostParameters:
        """获取参数"""
        return self.params


# ============================================================
# Cost Model 工厂
# ============================================================


class CostModelFactory:
    """Cost Model 工厂"""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type):
        """注册 Cost Model 类型"""
        cls._registry[name] = model_class

    @classmethod
    def create(
        cls,
        model_type: str = "parameterized",
        chip: str = "npu_910",
        **kwargs,
    ) -> ParameterizedCostModel:
        """创建 Cost Model"""
        if model_type == "parameterized":
            return ParameterizedCostModel(chip, **kwargs)

        if model_type in cls._registry:
            return cls._registry[model_type](chip, **kwargs)

        raise ValueError(f"Unknown model type: {model_type}")


# 注册默认模型
CostModelFactory.register("parameterized", ParameterizedCostModel)
