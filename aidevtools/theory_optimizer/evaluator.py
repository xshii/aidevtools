"""
融合策略评估器

设计模式:
- Facade 模式: 统一入口
- Observer 模式: 评估事件通知
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .benchmark import Benchmark, BenchmarkSuite
from .cost_model import CostParameters, CostResult, ParameterizedCostModel
from .calibration import MLCalibrator, MeasurementRecord
from .strategy.base import TilingStrategy, TilingResult, StrategyRegistry
from .views.base import View, ViewResult, ViewFormat, ViewRegistry
from .memory_plan import MemoryPlan, MemoryPlanBuilder


@dataclass
class EvalResult:
    """评估结果"""
    benchmark: Benchmark
    strategy_name: str
    tiling_result: TilingResult
    cost_result: CostResult
    views: Dict[str, ViewResult] = field(default_factory=dict)
    memory_plan: Optional[MemoryPlan] = None

    def summary(self) -> str:
        """生成摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Evaluation Result: {self.benchmark.name}")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"Strategy: {self.strategy_name}")
        lines.append(f"Operators: {len(self.benchmark.ops)}")
        lines.append(f"Fusion pairs: {len(self.benchmark.fuse_pairs)}")
        lines.append("")

        lines.append("Tiling Result:")
        lines.append(f"  Fusion groups: {len(self.tiling_result.fusion_configs)}")
        lines.append(f"  Unfused ops: {len(self.tiling_result.unfused_configs)}")
        lines.append(f"  Estimated cycles: {self.tiling_result.estimated_cycles:,}")
        lines.append(f"  Compute utilization: {self.tiling_result.compute_utilization:.1%}")
        lines.append(f"  Memory utilization: {self.tiling_result.memory_utilization:.1%}")
        lines.append("")

        lines.append("Cost Analysis:")
        lines.append(f"  Compute cycles: {self.cost_result.compute_cycles:,}")
        lines.append(f"  Memory cycles: {self.cost_result.memory_cycles:,}")
        lines.append(f"  Total cycles: {self.cost_result.total_cycles:,}")
        lines.append(f"  Bottleneck: {self.cost_result.bottleneck}")

        if self.views:
            lines.append("")
            lines.append(f"Generated views: {', '.join(self.views.keys())}")

        return "\n".join(lines)


@dataclass
class CompareResult:
    """策略比较结果"""
    benchmark: Benchmark
    results: Dict[str, EvalResult]  # strategy_name -> result
    baseline: Optional[str] = None  # 基准策略

    def summary(self) -> str:
        """生成比较摘要"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"Strategy Comparison: {self.benchmark.name}")
        lines.append("=" * 70)
        lines.append("")

        # 表头
        header = f"{'Strategy':<20} {'Cycles':>12} {'Compute':>10} {'Memory':>10} {'Speedup':>10}"
        lines.append(header)
        lines.append("-" * 70)

        baseline_cycles = None
        if self.baseline and self.baseline in self.results:
            baseline_cycles = self.results[self.baseline].cost_result.total_cycles

        for name, result in self.results.items():
            cycles = result.cost_result.total_cycles
            compute = result.tiling_result.compute_utilization
            memory = result.tiling_result.memory_utilization

            if baseline_cycles:
                speedup = baseline_cycles / cycles if cycles > 0 else 0
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "-"

            if name == self.baseline:
                name += " (baseline)"

            lines.append(
                f"{name:<20} {cycles:>12,} {compute:>9.1%} {memory:>9.1%} {speedup_str:>10}"
            )

        # 最佳策略
        lines.append("")
        best = min(self.results.items(), key=lambda x: x[1].cost_result.total_cycles)
        lines.append(f"Best strategy: {best[0]} ({best[1].cost_result.total_cycles:,} cycles)")

        return "\n".join(lines)


class FusionEvaluator:
    """
    融合策略评估器 (Facade 模式)

    提供统一的评估接口，协调各个子系统
    """

    def __init__(self,
                 cost_params: Optional[CostParameters] = None,
                 chip_spec: Optional[Any] = None):
        """
        Args:
            cost_params: 成本模型参数
            chip_spec: 芯片规格 (来自 analysis 模块)
        """
        self.cost_model = ParameterizedCostModel(
            params=cost_params,
            chip_spec=chip_spec
        )
        self.calibrator = MLCalibrator(self.cost_model)

        # 事件回调
        self._callbacks: Dict[str, List[Callable]] = {
            "before_eval": [],
            "after_eval": [],
            "before_compare": [],
            "after_compare": [],
        }

    # ==================== 评估接口 ====================

    def evaluate(self, benchmark: Benchmark,
                strategy: str = "baseline",
                generate_views: Optional[List[str]] = None,
                generate_memory_plan: bool = False,
                constraints: Optional[Dict] = None) -> EvalResult:
        """
        评估单个策略

        Args:
            benchmark: 待评估的 benchmark
            strategy: 策略名称
            generate_views: 需要生成的视图列表
            generate_memory_plan: 是否生成内存规划
            constraints: 约束条件

        Returns:
            EvalResult: 评估结果
        """
        self._notify("before_eval", benchmark=benchmark, strategy=strategy)

        # 1. 获取策略
        strategy_instance = StrategyRegistry.create(strategy)
        if not strategy_instance:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 2. 执行 tiling
        tiling_result = strategy_instance.optimize(benchmark, constraints)

        # 3. 成本估算
        cost_result = self.cost_model.estimate_benchmark(benchmark, tiling_result)

        # 4. 生成视图
        views = {}
        if generate_views:
            for view_name in generate_views:
                view = ViewRegistry.create(view_name)
                if view:
                    views[view_name] = view.generate(tiling_result)

        # 5. 生成内存规划
        memory_plan = None
        if generate_memory_plan:
            memory_plan = self._generate_memory_plan(benchmark, tiling_result)

        result = EvalResult(
            benchmark=benchmark,
            strategy_name=strategy,
            tiling_result=tiling_result,
            cost_result=cost_result,
            views=views,
            memory_plan=memory_plan
        )

        self._notify("after_eval", result=result)
        return result

    def compare(self, benchmark: Benchmark,
               strategies: Optional[List[str]] = None,
               baseline: str = "baseline",
               generate_views: Optional[List[str]] = None,
               constraints: Optional[Dict] = None) -> CompareResult:
        """
        比较多个策略

        Args:
            benchmark: 待评估的 benchmark
            strategies: 策略列表 (None 表示所有已注册策略)
            baseline: 基准策略
            generate_views: 需要生成的视图列表
            constraints: 约束条件

        Returns:
            CompareResult: 比较结果
        """
        if strategies is None:
            strategies = StrategyRegistry.list_strategies()

        self._notify("before_compare", benchmark=benchmark, strategies=strategies)

        results = {}
        for strategy in strategies:
            try:
                result = self.evaluate(
                    benchmark,
                    strategy=strategy,
                    generate_views=generate_views,
                    constraints=constraints
                )
                results[strategy] = result
            except Exception as e:
                print(f"Warning: Strategy {strategy} failed: {e}")

        compare_result = CompareResult(
            benchmark=benchmark,
            results=results,
            baseline=baseline if baseline in results else None
        )

        self._notify("after_compare", result=compare_result)
        return compare_result

    def evaluate_suite(self, suite_name: str,
                      strategy: str = "baseline",
                      **suite_args) -> EvalResult:
        """
        评估预定义的 benchmark suite

        Args:
            suite_name: suite 名称 (bert_ffn, llama_ffn, attention)
            strategy: 策略名称
            **suite_args: suite 参数

        Returns:
            EvalResult: 评估结果
        """
        # 获取 suite
        suite_method = getattr(BenchmarkSuite, suite_name, None)
        if not suite_method:
            raise ValueError(f"Unknown suite: {suite_name}")

        benchmark = suite_method(**suite_args)
        return self.evaluate(benchmark, strategy=strategy)

    # ==================== 校准接口 ====================

    def add_measurement(self, op_name: str, measured_cycles: int,
                       shapes: Dict[str, int],
                       op_type: str = "matmul") -> None:
        """添加实测数据"""
        self.calibrator.add_measurement(MeasurementRecord(
            op_name=op_name,
            op_type=op_type,
            shapes=shapes,
            measured_cycles=measured_cycles
        ))

    def import_measurements(self, path: str) -> int:
        """导入实测数据"""
        if path.endswith(".csv"):
            return self.calibrator.import_csv(path)
        elif path.endswith(".json"):
            return self.calibrator.import_json(path)
        else:
            raise ValueError(f"Unsupported format: {path}")

    def calibrate(self, method: str = "layered") -> Dict[str, Any]:
        """
        执行校准

        Args:
            method: 校准方法 (linear, layered, gradient)

        Returns:
            校准结果
        """
        result = self.calibrator.fit(method=method)
        return {
            "method": result.method,
            "r_squared": result.r_squared,
            "rmse": result.rmse,
            "parameters": result.params.to_vector().tolist(),
        }

    def evaluate_calibration(self) -> Dict[str, float]:
        """评估校准质量"""
        metrics = self.calibrator.evaluate()
        return {
            "mae": metrics.mae,
            "rmse": metrics.rmse,
            "r_squared": metrics.r_squared,
            "max_error": metrics.max_error,
        }

    def save_calibration(self, path: str) -> None:
        """保存校准结果"""
        self.calibrator.save(path)

    def load_calibration(self, path: str) -> None:
        """加载校准结果"""
        self.calibrator.load(path)

    # ==================== 视图接口 ====================

    def generate_view(self, tiling_result: TilingResult,
                     view_name: str,
                     format: ViewFormat = ViewFormat.TEXT) -> ViewResult:
        """
        生成单个视图

        Args:
            tiling_result: tiling 结果
            view_name: 视图名称
            format: 输出格式

        Returns:
            ViewResult: 视图结果
        """
        view = ViewRegistry.create(view_name)
        if not view:
            raise ValueError(f"Unknown view: {view_name}")

        return view.generate(tiling_result, format=format)

    def list_views(self) -> List[str]:
        """列出所有可用视图"""
        return ViewRegistry.list_views()

    # ==================== 内存规划接口 ====================

    def generate_memory_plan(self, benchmark: Benchmark,
                            tiling_result: TilingResult,
                            l1_size: int = 256 * 1024,
                            l2_size: int = 2 * 1024 * 1024) -> MemoryPlan:
        """
        生成内存规划

        Args:
            benchmark: benchmark
            tiling_result: tiling 结果
            l1_size: L1 大小
            l2_size: L2 大小

        Returns:
            MemoryPlan: 内存规划
        """
        return self._generate_memory_plan(benchmark, tiling_result, l1_size, l2_size)

    def generate_dma_code(self, memory_plan: MemoryPlan,
                         language: str = "pseudo") -> str:
        """
        生成 DMA 代码

        Args:
            memory_plan: 内存规划
            language: 目标语言

        Returns:
            生成的代码
        """
        return memory_plan.generate_code(language)

    # ==================== 策略接口 ====================

    def list_strategies(self) -> List[str]:
        """列出所有可用策略"""
        return StrategyRegistry.list_strategies()

    def register_strategy(self, name: str, strategy_class: type) -> None:
        """注册自定义策略"""
        StrategyRegistry._strategies[name] = strategy_class

    # ==================== 事件接口 ====================

    def on(self, event: str, callback: Callable) -> None:
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _notify(self, event: str, **kwargs) -> None:
        """触发事件"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                print(f"Warning: Callback error: {e}")

    # ==================== 内部方法 ====================

    def _generate_memory_plan(self, benchmark: Benchmark,
                             tiling_result: TilingResult,
                             l1_size: int = 256 * 1024,
                             l2_size: int = 2 * 1024 * 1024) -> MemoryPlan:
        """生成内存规划"""
        builder = MemoryPlanBuilder()
        builder.create_plan(l1_size=l1_size, l2_size=l2_size)

        # 为每个算子分配内存
        all_configs = {}
        for fc in tiling_result.fusion_configs:
            all_configs.update(fc.tile_configs)
        all_configs.update(tiling_result.unfused_configs)

        for op_name, config in all_configs.items():
            op_spec = benchmark.ops.get(op_name)
            if not op_spec:
                continue

            builder.for_op(op_name)

            # 根据 tile 大小分配 buffer
            if op_spec.op_type.value == "matmul":
                M = config.tile_sizes.get("M", 64)
                N = config.tile_sizes.get("N", 64)
                K = config.tile_sizes.get("K", 64)

                builder.add_input("A", (M, K))
                builder.add_input("B", (K, N))
                builder.add_output("C", (M, N))
            else:
                # 通用处理
                shape = tuple(config.tile_sizes.values())
                builder.add_input("input", shape)
                builder.add_output("output", shape)

        # 生成 DMA 操作
        builder.generate_dma()

        return builder.build()


# 便捷函数
def quick_evaluate(benchmark: Benchmark,
                  strategy: str = "baseline",
                  views: Optional[List[str]] = None) -> EvalResult:
    """快速评估"""
    evaluator = FusionEvaluator()
    return evaluator.evaluate(benchmark, strategy=strategy, generate_views=views)


def quick_compare(benchmark: Benchmark,
                 strategies: Optional[List[str]] = None) -> CompareResult:
    """快速比较"""
    evaluator = FusionEvaluator()
    return evaluator.compare(benchmark, strategies=strategies)
