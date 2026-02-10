"""Compare 可视化测试

测试三层架构的可视化功能
"""

import pytest
import numpy as np


class TestVisualizer:
    """基础底座测试"""

    def test_create_page(self):
        """创建 Page"""
        try:
            from aidevtools.compare.visualizer import Visualizer
        except ImportError:
            pytest.skip("pyecharts not installed")

        page = Visualizer.create_page(title="Test Report")
        assert page.page_title == "Test Report"

    def test_create_pie(self):
        """创建饼图"""
        try:
            from aidevtools.compare.visualizer import Visualizer
        except ImportError:
            pytest.skip("pyecharts not installed")

        pie = Visualizer.create_pie(
            data={"A": 100, "B": 50},
            title="Test Pie"
        )
        assert pie is not None

    def test_create_bar(self):
        """创建柱状图"""
        try:
            from aidevtools.compare.visualizer import Visualizer
        except ImportError:
            pytest.skip("pyecharts not installed")

        bar = Visualizer.create_bar(
            x_data=["X1", "X2"],
            series_data={"S1": [10, 20]},
            title="Test Bar"
        )
        assert bar is not None


class TestBitAnalysisVisualize:
    """BitAnalysis 策略级可视化测试"""

    def test_visualize(self):
        """生成可视化报告"""
        try:
            from aidevtools.compare.visualizer import Visualizer
        except ImportError:
            pytest.skip("pyecharts not installed")

        from aidevtools.compare.strategy import BitAnalysisStrategy, FP32

        # 构造测试数据
        golden = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = np.array([-1.0, 2.0, 3.0], dtype=np.float32)  # sign flip

        res = BitAnalysisStrategy.compare(golden, result, fmt=FP32)

        # 可视化
        page = BitAnalysisStrategy.visualize(res)
        assert page is not None
        assert page.page_title == "Bit Analysis Report"


class TestBlockedVisualize:
    """Blocked 策略级可视化测试"""

    def test_visualize(self):
        """生成可视化报告"""
        try:
            from aidevtools.compare.visualizer import Visualizer
        except ImportError:
            pytest.skip("pyecharts not installed")

        from aidevtools.compare.strategy import BlockedStrategy

        # 构造测试数据
        golden = np.random.randn(1000).astype(np.float32)
        result = golden + np.random.randn(1000) * 0.01

        blocks = BlockedStrategy.compare(golden, result, block_size=256)

        # 可视化
        page = BlockedStrategy.visualize(blocks, threshold=20.0, cols=4)
        assert page is not None
        assert page.page_title == "Blocked Analysis Report"


class TestModelVisualizer:
    """模型级可视化测试"""

    def test_visualize(self):
        """生成模型级报告"""
        try:
            from aidevtools.compare.visualizer import Visualizer
        except ImportError:
            pytest.skip("pyecharts not installed")

        from aidevtools.compare.model_visualizer import (
            ModelVisualizer,
            ModelCompareResult,
            OpCompareResult,
            OpStatus,
        )

        # 构建测试数据
        ops = [
            OpCompareResult("op1", 0, OpStatus.HAS_DATA, qsnr=45.2, passed=True),
            OpCompareResult("op2", 1, OpStatus.HAS_DATA, qsnr=12.3, passed=False),
            OpCompareResult("op3", 2, OpStatus.MISSING_DUT),
        ]

        model_result = ModelCompareResult(
            model_name="TestModel",
            ops=ops,
            total_ops=3,
            ops_with_data=2,
            ops_missing_dut=1,
            passed_ops=1,
            failed_ops=1,
        )

        # 可视化
        page = ModelVisualizer.visualize(model_result)
        assert page is not None
        assert "TestModel" in page.page_title
