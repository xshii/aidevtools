"""
模型级可视化 - 误差传播分析

核心功能：
1. 跨算子误差传播
2. 误差源头定位
3. 处理 DUT 部分算子缺失
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class OpStatus(Enum):
    """算子状态"""
    HAS_DATA = "has_data"      # 有完整数据
    MISSING_DUT = "missing_dut"  # 缺少 DUT 输出
    SKIPPED = "skipped"        # 未比对


@dataclass
class OpCompareResult:
    """单算子比对结果"""
    op_name: str
    op_id: int
    status: OpStatus

    # 比对指标（如有数据）
    qsnr: Optional[float] = None
    cosine: Optional[float] = None
    max_abs: Optional[float] = None
    passed: Optional[bool] = None


@dataclass
class ModelCompareResult:
    """模型级比对结果"""
    model_name: str
    ops: List[OpCompareResult]

    # 统计
    total_ops: int
    ops_with_data: int
    ops_missing_dut: int
    passed_ops: int
    failed_ops: int


class ModelVisualizer:
    """
    模型级可视化

    核心：误差传播分析
    """

    @staticmethod
    def visualize(result: ModelCompareResult) -> "Page":
        """
        生成模型级完整报告

        包含:
        1. 误差传播 Sankey 图
        2. 算子 QSNR 排序
        3. 误差累积曲线
        4. 数据完整性
        """
        from aidevtools.compare.visualizer import Visualizer

        page = Visualizer.create_page(title=f"Model Analysis: {result.model_name}")

        # 1. 误差传播 Sankey 图
        sankey = ModelVisualizer._create_error_propagation(result)
        if sankey:
            page.add(sankey)

        # 2. 算子 QSNR 排序（找瓶颈）
        qsnr_chart = ModelVisualizer._create_op_qsnr_ranking(result)
        if qsnr_chart:
            page.add(qsnr_chart)

        # 3. 误差累积曲线
        accumulation = ModelVisualizer._create_error_accumulation(result)
        if accumulation:
            page.add(accumulation)

        # 4. 数据完整性
        completeness = ModelVisualizer._create_data_completeness(result)
        page.add(completeness)

        return page

    @staticmethod
    def _create_error_propagation(result: ModelCompareResult):
        """
        误差传播 Sankey 图

        展示误差在算子间流动:
        Input → Conv1 → BN1 → Conv2 → Output
               (15%)   (12%)   (25%)
        """
        from aidevtools.compare.visualizer import Visualizer

        # 构建节点和链接
        nodes = []
        links = []

        # 只包含有数据的算子
        valid_ops = [op for op in result.ops if op.status == OpStatus.HAS_DATA and op.qsnr is not None]

        if len(valid_ops) < 2:
            return None  # 数据不足

        # 节点：Input + 各算子
        nodes.append("Input")
        for op in valid_ops:
            nodes.append(op.op_name)

        # 链接：误差流动
        for i, op in enumerate(valid_ops):
            # 误差值 = 100 - QSNR (简化)
            error_val = max(0, 100 - op.qsnr)

            links.append({
                "source": nodes[i] if i < len(nodes) - 1 else nodes[-2],
                "target": op.op_name,
                "value": error_val,
            })

        sankey = Visualizer.create_sankey(
            nodes=nodes,
            links=links,
            title="Error Propagation Flow",
        )
        return sankey

    @staticmethod
    def _create_op_qsnr_ranking(result: ModelCompareResult):
        """
        算子 QSNR 排序

        找出误差最大的算子（瓶颈定位）
        """
        from aidevtools.compare.visualizer import Visualizer

        # 提取有 QSNR 的算子
        ops_with_qsnr = [
            (op.op_name, op.qsnr)
            for op in result.ops
            if op.status == OpStatus.HAS_DATA and op.qsnr is not None
        ]

        if not ops_with_qsnr:
            return None

        # 按 QSNR 升序排序（误差大的在前）
        ops_with_qsnr.sort(key=lambda x: x[1])

        # 取前 20
        top_ops = ops_with_qsnr[:min(20, len(ops_with_qsnr))]

        x_data = [name for name, _ in top_ops]
        series = {"QSNR (dB)": [qsnr for _, qsnr in top_ops]}

        bar = Visualizer.create_bar(
            x_data,
            series,
            title="Op QSNR Ranking (Lower = Worse)",
            horizontal=True,
        )
        return bar

    @staticmethod
    def _create_error_accumulation(result: ModelCompareResult):
        """
        误差累积曲线

        展示误差随算子层数的累积趋势
        """
        from aidevtools.compare.visualizer import Visualizer

        # 提取有数据的算子
        valid_ops = [
            op for op in result.ops
            if op.status == OpStatus.HAS_DATA and op.qsnr is not None
        ]

        if not valid_ops:
            return None

        x_data = [f"{op.op_id}:{op.op_name}" for op in valid_ops]
        y_qsnr = [op.qsnr for op in valid_ops]

        # 如果有 cosine 数据
        y_cosine = [op.cosine * 100 if op.cosine else 0 for op in valid_ops]

        series = {"QSNR (dB)": y_qsnr}
        if any(y_cosine):
            series["Cosine (%)"] = y_cosine

        line = Visualizer.create_line(
            x_data,
            series,
            title="Error Accumulation Across Ops",
        )
        return line

    @staticmethod
    def _create_data_completeness(result: ModelCompareResult):
        """
        数据完整性摘要

        展示有多少算子缺失 DUT 输出
        """
        from aidevtools.compare.visualizer import Visualizer

        data = {
            "✅ Has Data": result.ops_with_data,
            "❌ Missing DUT": result.ops_missing_dut,
        }

        skipped = result.total_ops - result.ops_with_data - result.ops_missing_dut
        if skipped > 0:
            data["⏭️ Skipped"] = skipped

        pie = Visualizer.create_pie(
            data,
            title=f"Data Completeness ({result.ops_with_data}/{result.total_ops} ops)",
        )
        return pie


__all__ = [
    "OpStatus",
    "OpCompareResult",
    "ModelCompareResult",
    "ModelVisualizer",
]
