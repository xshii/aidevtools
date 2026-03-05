"""
组合策略

将多个策略组合，按顺序执行。
"""
from typing import List, Dict, Any
from .base import CompareStrategy, CompareContext


class CompositeStrategy(CompareStrategy):
    """
    组合策略

    将多个子策略组合，按顺序执行。
    """

    def __init__(self, strategies: List[CompareStrategy], name: str = "composite"):
        self.strategies = strategies
        self._name = name

    def prepare(self, ctx: CompareContext) -> None:
        for strategy in self.strategies:
            strategy.prepare(ctx)

    def run(self, ctx: CompareContext) -> Dict[str, Any]:
        results = {}
        for strategy in self.strategies:
            results[strategy.name] = strategy.run(ctx)
        return results

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        strategy_names = [s.name for s in self.strategies]
        return f"CompositeStrategy(name={self.name!r}, strategies={strategy_names})"
