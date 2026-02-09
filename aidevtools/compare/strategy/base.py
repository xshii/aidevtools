"""
比对策略基类

使用策略模式重构compare模块，提供统一的比对接口。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

from ..types import CompareConfig, _PreparedPair


@dataclass
class CompareContext:
    """
    比对上下文

    包含所有策略共享的数据，避免重复传递参数。

    Attributes:
        golden: Golden数据（纯FP32/FP64）
        dut: DUT数据（待测数据）
        config: 比对配置（阈值、格式等）
        golden_qnt: Golden量化数据（可选，用于量化感知比对）
        prepared: 预处理数据缓存（避免重复flatten/astype）
        metadata: 额外元数据（供策略使用）
    """
    golden: np.ndarray
    dut: np.ndarray
    config: CompareConfig
    golden_qnt: Optional[np.ndarray] = None
    prepared: Optional[_PreparedPair] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CompareStrategy(ABC):
    """
    比对策略基类

    所有比对策略都应继承此类，实现统一的compare接口。

    策略模式优点：
    - 易于扩展新的比对策略
    - 策略可组合、可复用
    - 易于单元测试
    - 解耦比对逻辑和引擎逻辑
    """

    @abstractmethod
    def run(self, ctx: CompareContext) -> Any:
        """
        执行比对（策略模式入口）

        Args:
            ctx: 比对上下文（包含golden/dut/config等）

        Returns:
            比对结果（类型取决于具体策略）

        Note:
            这是实例方法，用于策略模式。
            各策略的主比对逻辑应实现为静态方法 compare()，
            然后在 run() 中调用。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称（用于日志、报告等）"""
        pass

    def prepare(self, ctx: CompareContext) -> None:
        """
        预处理（可选）

        某些策略可能需要预处理数据，如创建PreparedPair缓存。
        默认实现为空，子类可覆盖。

        Args:
            ctx: 比对上下文
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
